from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import yaml

from .config import OptimizerConfig, SchedulerConfig, TrainerConfig
from ..blocks.conv_block import build_conv_dense_drn_block, build_conv_drn_block
from ..models.digital_modules import (
    DigitalClassifierHead,
    DigitalModuleBlock,
    build_head_from_spec,
    build_layers_from_spec,
)
from ..models.network import DigitalDRNNet, SequentialDigitalDRNNet
from ..models.network_digital_analog import DigitalAnalogNet
from .trainer import BPTrainer, HybridEPTrainer
from ..utils.data import build_image_transforms


@dataclass
class ExperimentBundle:
    config: dict[str, Any]
    model: DigitalDRNNet
    trainer: BPTrainer
    train_loader: DataLoader | None
    eval_loader: DataLoader | None


def _config_root(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    return _package_root() / "hydra_conf"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(
    *,
    config_name: str = "config",
    config_dir: str | Path | None = None,
    algorithm: str | None = None,
    trainer: str | None = None,
    data: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    root = _config_root(config_dir)
    base = _load_yaml(root / f"{config_name}.yaml")
    defaults = list(base.pop("defaults", []))
    if config_name != "config":
        shared_root = _load_yaml(root / "config.yaml")
        inherited_root = {
            key: value
            for key, value in shared_root.items()
            if key in {"config", "exp_type"}
        }
        base = _deep_merge(inherited_root, base)

    overrides = {
        "algorithm": algorithm,
        "trainer": trainer,
        "data": data,
        "model": model,
    }
    composed: dict[str, Any] = {}
    selected: dict[str, str] = {}

    for entry in defaults:
        if entry == "_self_":
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"Unsupported defaults entry {entry!r} in {root / f'{config_name}.yaml'}.")
        if len(entry) != 1:
            raise ValueError(f"Defaults entries must contain exactly one key, got {entry!r}.")
        section, choice = next(iter(entry.items()))
        choice = overrides.get(section) or choice
        composed[section] = _load_yaml(root / section / f"{choice}.yaml")
        selected[section] = str(choice)

    composed["_selection"] = selected

    return _deep_merge(composed, base)


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def _resolve_output_root(output_dir: str | Path) -> Path:
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = _package_root() / output_dir_path
    return output_dir_path


def _resolve_data_root(data_root: str | Path | None) -> Path:
    if data_root is None:
        data_root = "~/data"
    expanded = os.path.expandvars(str(data_root))
    return Path(expanded).expanduser()


def _build_timestamped_run_dir(output_root: str | Path, config: Mapping[str, Any]) -> Path:
    selection = dict(config.get("_selection", {}))
    model_dir_name = selection.get("model") or config.get("model", {}).get("name") or "model"
    model_dir_name = _sanitize_path_component(str(model_dir_name))
    run_root = _resolve_output_root(output_root) / model_dir_name
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    hostname = _sanitize_path_component(socket.gethostname())
    run_name = f"{timestamp}-{hostname}"
    run_dir = run_root / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"{run_name}-{suffix:02d}"
        suffix += 1
    return run_dir


def _flatten_block_config(block_config: Mapping[str, Any], model_defaults: Mapping[str, Any]) -> dict[str, Any]:
    if "ff" in block_config or "drn" in block_config:
        ff_cfg = dict(block_config.get("ff", {}))
        drn_cfg = dict(block_config.get("drn", {}))
    else:
        ff_cfg = {}
        drn_cfg = dict(block_config)

    layer_dims = drn_cfg.get("layer_dims")
    if layer_dims is None:
        raise ValueError("Each block config must define drn.layer_dims.")

    flat: dict[str, Any] = {
        "layer_dims": layer_dims,
        "ff_bias": _coalesce(ff_cfg.get("bias"), model_defaults.get("ff_bias"), True),
        "ff_activation": _coalesce(ff_cfg.get("activation"), model_defaults.get("ff_activation"), "tanh"),
        "signed_drive": _coalesce(ff_cfg.get("signed_drive"), model_defaults.get("signed_drive"), False),
        "ff_learning_rate": _coalesce(ff_cfg.get("lr"), ff_cfg.get("learning_rate")),
        "num_iterations": _coalesce(drn_cfg.get("num_iterations"), model_defaults.get("num_iterations"), 6),
        "mode": _coalesce(drn_cfg.get("mode"), model_defaults.get("mode"), "asynchronous"),
        "non_linearity": _coalesce(
            drn_cfg.get("drn_non_linearity"),
            drn_cfg.get("non_linearity"),
            model_defaults.get("drn_non_linearity"),
            "linear",
        ),
        "weight_gains": _coalesce(drn_cfg.get("weight_gains"), model_defaults.get("weight_gains"), 0.1),
        "bias_gain": _coalesce(drn_cfg.get("bias_gain"), model_defaults.get("bias_gain"), 0.0),
        "voltage_amp": _coalesce(drn_cfg.get("voltage_amp"), model_defaults.get("voltage_amp"), 1.0),
        "current_amp": _coalesce(drn_cfg.get("current_amp"), model_defaults.get("current_amp"), 1.0),
        "amplify_first_free_layer": _coalesce(
            drn_cfg.get("amplify_first_free_layer"),
            model_defaults.get("amplify_first_free_layer"),
            True,
        ),
        "weight_min": _coalesce(drn_cfg.get("weight_min"), model_defaults.get("weight_min")),
        "weight_max": _coalesce(drn_cfg.get("weight_max"), model_defaults.get("weight_max")),
        "weight_init_mode": _coalesce(drn_cfg.get("weight_init_mode"), model_defaults.get("weight_init_mode"), "kaiming_uniform"),
        "quadratic_diode_param": dict(_coalesce(drn_cfg.get("quadratic_diode_param"), model_defaults.get("quadratic_diode_param"), {})),
        "exponential_diode_param": dict(_coalesce(drn_cfg.get("exponential_diode_param"), model_defaults.get("exponential_diode_param"), {})),
        "hard_sigmoid_param": dict(_coalesce(drn_cfg.get("hard_sigmoid_param"), model_defaults.get("hard_sigmoid_param"), {})),
        "learn_drive_scale": _coalesce(drn_cfg.get("learn_drive_scale"), model_defaults.get("learn_drive_scale"), True),
        "init_drive_scale": _coalesce(drn_cfg.get("init_drive_scale"), model_defaults.get("init_drive_scale"), 1.0),
        "drn_learning_rate": _coalesce(drn_cfg.get("lr"), drn_cfg.get("learning_rate")),
    }

    ff_type = ff_cfg.get("type", "dense")
    if ff_type == "conv":
        flat.update(
            {
                "ff_input_shape": tuple(_coalesce(ff_cfg.get("input_shape"), model_defaults.get("input_shape")) or ()),
                "ff_conv_channels": ff_cfg.get("conv_channels"),
                "ff_conv_kernels": ff_cfg.get("conv_kernels"),
                "ff_conv_strides": ff_cfg.get("conv_strides"),
                "ff_conv_paddings": ff_cfg.get("conv_paddings"),
                "ff_conv_pool_kernels": ff_cfg.get("conv_pool_kernels"),
            }
        )
        if not flat["ff_input_shape"]:
            raise ValueError("Conv FF blocks must define ff.input_shape or model.config.input_shape.")
    elif ff_type != "dense":
        raise ValueError(f"Unsupported ff.type '{ff_type}'.")

    return flat


def _resolve_transition_layers(ff_cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    option = ff_cfg.get("transition_option", "A")
    if option == "A":
        return list(ff_cfg.get("layers", []))
    alternative = dict(ff_cfg.get("alternatives", {}).get(option, {}))
    return list(alternative.get("layers", ff_cfg.get("layers", [])))


def _build_conv_layer_shapes(drn_cfg: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    num_layers = int(drn_cfg.get("num_layers", 2))
    state_shape = tuple(drn_cfg["state_shape"])
    if len(state_shape) != 3:
        raise ValueError(f"Expected drn.state_shape as [channels, height, width], got {state_shape}.")
    channels = list(drn_cfg.get("channels", [state_shape[0]] * num_layers))
    if len(channels) != num_layers:
        raise ValueError(f"Expected {num_layers} channel entries for conv DRN block, got {len(channels)}.")
    _, height, width = state_shape
    return [(int(ch), int(height), int(width)) for ch in channels]


def _build_cifar10_v0_model(config: Mapping[str, Any]) -> DigitalAnalogNet:
    model_cfg = config["model"]
    model_defaults = dict(model_cfg.get("config", {}))
    blocks = []

    for block_cfg in model_cfg.get("blocks_config", []):
        ff_cfg = dict(block_cfg.get("ff", {}))
        drn_cfg = dict(block_cfg.get("drn", {}))
        ff_layers = _resolve_transition_layers(ff_cfg)
        ff_module = build_layers_from_spec(ff_layers)

        block = build_conv_drn_block(
            ff=ff_module,
            layer_shapes=_build_conv_layer_shapes(drn_cfg),
            kernel_sizes=drn_cfg.get("kernel_size", 3),
            strides=drn_cfg.get("stride", 1),
            paddings=drn_cfg.get("padding", 0),
            dilations=drn_cfg.get("dilation", 1),
            ff_learning_rate=_coalesce(ff_cfg.get("lr"), ff_cfg.get("learning_rate")),
            signed_drive=_coalesce(ff_cfg.get("signed_drive"), model_defaults.get("signed_drive"), False),
            num_iterations=_coalesce(drn_cfg.get("num_iterations"), model_defaults.get("num_iterations"), 6),
            mode=_coalesce(drn_cfg.get("mode"), model_defaults.get("mode"), "asynchronous"),
            non_linearity=_coalesce(
                drn_cfg.get("drn_non_linearity"),
                drn_cfg.get("non_linearity"),
                model_defaults.get("drn_non_linearity"),
                "linear",
            ),
            output_non_linearity=_coalesce(
                drn_cfg.get("output_non_linearity"),
                model_defaults.get("output_non_linearity"),
            ),
            weight_gains=_coalesce(drn_cfg.get("weight_gains"), model_defaults.get("weight_gains"), 0.1),
            bias_gain=_coalesce(drn_cfg.get("bias_gain"), model_defaults.get("bias_gain"), 0.0),
            voltage_amp=_coalesce(drn_cfg.get("voltage_amp"), model_defaults.get("voltage_amp"), 1.0),
            current_amp=_coalesce(drn_cfg.get("current_amp"), model_defaults.get("current_amp"), 1.0),
            amplify_first_free_layer=_coalesce(
                drn_cfg.get("amplify_first_free_layer"),
                model_defaults.get("amplify_first_free_layer"),
                True,
            ),
            weight_min=_coalesce(drn_cfg.get("weight_min"), model_defaults.get("weight_min")),
            weight_max=_coalesce(drn_cfg.get("weight_max"), model_defaults.get("weight_max")),
            weight_init_mode=_coalesce(drn_cfg.get("weight_init_mode"), model_defaults.get("weight_init_mode"), "kaiming_uniform"),
            quadratic_diode_param=dict(_coalesce(drn_cfg.get("quadratic_diode_param"), model_defaults.get("quadratic_diode_param"), {})),
            exponential_diode_param=dict(_coalesce(drn_cfg.get("exponential_diode_param"), model_defaults.get("exponential_diode_param"), {})),
            hard_sigmoid_param=dict(_coalesce(drn_cfg.get("hard_sigmoid_param"), model_defaults.get("hard_sigmoid_param"), {})),
            learn_drive_scale=_coalesce(drn_cfg.get("learn_drive_scale"), model_defaults.get("learn_drive_scale"), True),
            init_drive_scale=_coalesce(drn_cfg.get("init_drive_scale"), model_defaults.get("init_drive_scale"), 1.0),
            drn_learning_rate=_coalesce(drn_cfg.get("lr"), drn_cfg.get("learning_rate")),
        )
        blocks.append(block)

    head = None
    if model_cfg.get("head") is not None:
        head_cfg = dict(model_cfg["head"])
        head = build_head_from_spec(head_cfg)

    return DigitalAnalogNet(blocks, head=head)


def _build_cifar10_digital_mlp_model(config: Mapping[str, Any]) -> DigitalAnalogNet:
    model_cfg = config["model"]
    blocks = []

    for block_cfg in model_cfg.get("blocks_config", []):
        ff_cfg = dict(block_cfg.get("ff", {}))
        ff_layers = _resolve_transition_layers(ff_cfg)
        ff_module = build_layers_from_spec(ff_layers)
        block = DigitalModuleBlock(
            ff_module,
            learning_rate=_coalesce(ff_cfg.get("lr"), ff_cfg.get("learning_rate")),
        )
        blocks.append(block)

    head = None
    if model_cfg.get("head") is not None:
        head_cfg = dict(model_cfg["head"])
        head = build_head_from_spec(head_cfg)

    return DigitalAnalogNet(blocks, head=head)


def _build_cifar10_conv_dense_analog_model(config: Mapping[str, Any]) -> DigitalAnalogNet:
    model_cfg = config["model"]
    model_defaults = dict(model_cfg.get("config", {}))
    blocks = []

    for block_cfg in model_cfg.get("blocks_config", []):
        ff_cfg = dict(block_cfg.get("ff", {}))
        drn_cfg = dict(block_cfg.get("drn", {}))
        ff_layers = _resolve_transition_layers(ff_cfg)
        ff_module = build_layers_from_spec(ff_layers)

        block = build_conv_dense_drn_block(
            ff=ff_module,
            conv_state_shape=drn_cfg["state_shape"],
            output_dim=drn_cfg["output_dim"],
            ff_learning_rate=_coalesce(ff_cfg.get("lr"), ff_cfg.get("learning_rate")),
            signed_drive=_coalesce(ff_cfg.get("signed_drive"), model_defaults.get("signed_drive"), False),
            num_iterations=_coalesce(drn_cfg.get("num_iterations"), model_defaults.get("num_iterations"), 6),
            mode=_coalesce(drn_cfg.get("mode"), model_defaults.get("mode"), "asynchronous"),
            non_linearity=_coalesce(
                drn_cfg.get("drn_non_linearity"),
                drn_cfg.get("non_linearity"),
                model_defaults.get("drn_non_linearity"),
                "linear",
            ),
            weight_gains=_coalesce(drn_cfg.get("weight_gains"), model_defaults.get("weight_gains"), 0.1),
            bias_gain=_coalesce(drn_cfg.get("bias_gain"), model_defaults.get("bias_gain"), 0.0),
            voltage_amp=_coalesce(drn_cfg.get("voltage_amp"), model_defaults.get("voltage_amp"), 1.0),
            current_amp=_coalesce(drn_cfg.get("current_amp"), model_defaults.get("current_amp"), 1.0),
            amplify_first_free_layer=_coalesce(
                drn_cfg.get("amplify_first_free_layer"),
                model_defaults.get("amplify_first_free_layer"),
                True,
            ),
            weight_min=_coalesce(drn_cfg.get("weight_min"), model_defaults.get("weight_min")),
            weight_max=_coalesce(drn_cfg.get("weight_max"), model_defaults.get("weight_max")),
            weight_init_mode=_coalesce(drn_cfg.get("weight_init_mode"), model_defaults.get("weight_init_mode"), "kaiming_uniform"),
            quadratic_diode_param=dict(_coalesce(drn_cfg.get("quadratic_diode_param"), model_defaults.get("quadratic_diode_param"), {})),
            exponential_diode_param=dict(_coalesce(drn_cfg.get("exponential_diode_param"), model_defaults.get("exponential_diode_param"), {})),
            hard_sigmoid_param=dict(_coalesce(drn_cfg.get("hard_sigmoid_param"), model_defaults.get("hard_sigmoid_param"), {})),
            learn_drive_scale=_coalesce(drn_cfg.get("learn_drive_scale"), model_defaults.get("learn_drive_scale"), True),
            init_drive_scale=_coalesce(drn_cfg.get("init_drive_scale"), model_defaults.get("init_drive_scale"), 1.0),
            drn_learning_rate=_coalesce(drn_cfg.get("lr"), drn_cfg.get("learning_rate")),
        )
        blocks.append(block)

    return DigitalAnalogNet(blocks, head=None)


def _build_cifar10_mixed_block_model(config: Mapping[str, Any]) -> DigitalAnalogNet:
    model_cfg = config["model"]
    model_defaults = dict(model_cfg.get("config", {}))
    blocks = []

    common_defaults = {
        "mode": model_defaults.get("mode", "asynchronous"),
        "non_linearity": model_defaults.get("drn_non_linearity", "linear"),
        "weight_gains": model_defaults.get("weight_gains", 0.1),
        "bias_gain": model_defaults.get("bias_gain", 0.0),
        "voltage_amp": model_defaults.get("voltage_amp", 1.0),
        "current_amp": model_defaults.get("current_amp", 1.0),
        "amplify_first_free_layer": model_defaults.get("amplify_first_free_layer", True),
        "weight_min": model_defaults.get("weight_min"),
        "weight_max": model_defaults.get("weight_max"),
        "weight_init_mode": model_defaults.get("weight_init_mode", "kaiming_uniform"),
        "quadratic_diode_param": dict(model_defaults.get("quadratic_diode_param", {})),
        "exponential_diode_param": dict(model_defaults.get("exponential_diode_param", {})),
        "hard_sigmoid_param": dict(model_defaults.get("hard_sigmoid_param", {})),
        "learn_drive_scale": model_defaults.get("learn_drive_scale", True),
        "init_drive_scale": model_defaults.get("init_drive_scale", 1.0),
    }

    for block_cfg in model_cfg.get("blocks_config", []):
        ff_cfg = dict(block_cfg.get("ff", {}))
        drn_cfg = dict(block_cfg.get("drn", {}))
        ff_layers = _resolve_transition_layers(ff_cfg)
        ff_module = build_layers_from_spec(ff_layers)
        drn_type = drn_cfg.get("type", "conv_drn_block")

        kwargs = {
            "ff": ff_module,
            "ff_learning_rate": _coalesce(ff_cfg.get("lr"), ff_cfg.get("learning_rate")),
            "signed_drive": _coalesce(ff_cfg.get("signed_drive"), model_defaults.get("signed_drive"), False),
            "num_iterations": _coalesce(drn_cfg.get("num_iterations"), model_defaults.get("num_iterations"), 6),
            "mode": _coalesce(drn_cfg.get("mode"), common_defaults["mode"]),
            "non_linearity": _coalesce(
                drn_cfg.get("drn_non_linearity"),
                drn_cfg.get("non_linearity"),
                common_defaults["non_linearity"],
            ),
            "weight_gains": _coalesce(drn_cfg.get("weight_gains"), common_defaults["weight_gains"]),
            "bias_gain": _coalesce(drn_cfg.get("bias_gain"), common_defaults["bias_gain"]),
            "voltage_amp": _coalesce(drn_cfg.get("voltage_amp"), common_defaults["voltage_amp"]),
            "current_amp": _coalesce(drn_cfg.get("current_amp"), common_defaults["current_amp"]),
            "amplify_first_free_layer": _coalesce(
                drn_cfg.get("amplify_first_free_layer"),
                common_defaults["amplify_first_free_layer"],
            ),
            "weight_min": _coalesce(drn_cfg.get("weight_min"), common_defaults["weight_min"]),
            "weight_max": _coalesce(drn_cfg.get("weight_max"), common_defaults["weight_max"]),
            "weight_init_mode": _coalesce(drn_cfg.get("weight_init_mode"), common_defaults["weight_init_mode"]),
            "quadratic_diode_param": dict(_coalesce(drn_cfg.get("quadratic_diode_param"), common_defaults["quadratic_diode_param"])),
            "exponential_diode_param": dict(_coalesce(drn_cfg.get("exponential_diode_param"), common_defaults["exponential_diode_param"])),
            "hard_sigmoid_param": dict(_coalesce(drn_cfg.get("hard_sigmoid_param"), common_defaults["hard_sigmoid_param"])),
            "learn_drive_scale": _coalesce(drn_cfg.get("learn_drive_scale"), common_defaults["learn_drive_scale"]),
            "init_drive_scale": _coalesce(drn_cfg.get("init_drive_scale"), common_defaults["init_drive_scale"]),
            "drn_learning_rate": _coalesce(drn_cfg.get("lr"), drn_cfg.get("learning_rate")),
        }

        if drn_type == "conv_drn_block":
            block = build_conv_drn_block(
                layer_shapes=_build_conv_layer_shapes(drn_cfg),
                kernel_sizes=drn_cfg.get("kernel_size", 3),
                strides=drn_cfg.get("stride", 1),
                paddings=drn_cfg.get("padding", 0),
                dilations=drn_cfg.get("dilation", 1),
                output_non_linearity=_coalesce(
                    drn_cfg.get("output_non_linearity"),
                    model_defaults.get("output_non_linearity"),
                ),
                **kwargs,
            )
        elif drn_type == "conv_dense_drn_block":
            block = build_conv_dense_drn_block(
                conv_state_shape=drn_cfg["state_shape"],
                output_dim=drn_cfg["output_dim"],
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported drn.type '{drn_type}'.")

        blocks.append(block)

    head = None
    if model_cfg.get("head") is not None:
        head_cfg = dict(model_cfg["head"])
        head = build_head_from_spec(head_cfg)

    return DigitalAnalogNet(blocks, head=head)


def build_model_from_config(config: Mapping[str, Any]) -> DigitalDRNNet:
    model_cfg = config["model"] if "model" in config else config
    model_name = model_cfg.get("name")
    if isinstance(model_name, str) and model_name.startswith("cifar10_digital_analog_v0"):
        return _build_cifar10_v0_model(config)
    if isinstance(model_name, str) and model_name.startswith("cifar10_mixed_analog"):
        return _build_cifar10_mixed_block_model(config)
    if isinstance(model_name, str) and model_name.startswith("cifar10_drn_only"):
        return _build_cifar10_mixed_block_model(config)
    if isinstance(model_name, str) and model_name.startswith("cifar10_conv_dense_analog"):
        return _build_cifar10_conv_dense_analog_model(config)
    if isinstance(model_name, str) and model_name.startswith("cifar10_digital_mlp"):
        return _build_cifar10_digital_mlp_model(config)
    if model_name != "sequential_digital_drn":
        raise ValueError(f"Unsupported model '{model_name}'.")

    model_defaults = dict(model_cfg.get("config", {}))
    data_cfg = dict(config.get("data", {}).get("config", {}))
    input_dim = model_defaults.get("input_dim", data_cfg.get("input_dim"))
    if input_dim is None:
        raise ValueError("Model input_dim must be defined in model.config or data.config.")

    block_configs = [
        _flatten_block_config(block, model_defaults)
        for block in model_cfg.get("blocks_config", [])
    ]
    if not block_configs:
        raise ValueError("Model config must define at least one block in blocks_config.")

    return SequentialDigitalDRNNet(input_dim=input_dim, block_configs=block_configs)


def build_trainer_config_from_config(
    config: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
) -> TrainerConfig:
    trainer_cfg = dict(config["trainer"])
    run_cfg = dict(config.get("config", {}))

    optimizer_cfg = OptimizerConfig(**trainer_cfg.get("optimizer", {}))

    scheduler_raw = trainer_cfg.get("scheduler")
    scheduler_cfg = None
    if isinstance(scheduler_raw, Mapping):
        scheduler_name = scheduler_raw.get("name")
        if scheduler_name is not None:
            scheduler_config = dict(scheduler_raw.get("config", {}))
            if scheduler_name == "cosine" and scheduler_config.get("T_max") == "epochs":
                scheduler_config["T_max"] = int(trainer_cfg.get("epochs", 20))
            scheduler_cfg = SchedulerConfig(name=scheduler_name, config=scheduler_config)

    resolved_checkpoint_dir = checkpoint_dir
    if resolved_checkpoint_dir is None and run_cfg.get("save", True):
        output_dir = run_cfg.get("output_dir")
        if output_dir is not None:
            resolved_checkpoint_dir = _build_timestamped_run_dir(output_dir, config)

    return TrainerConfig(
        epochs=trainer_cfg.get("epochs", 20),
        criterion=trainer_cfg.get("criterion", "cross_entropy"),
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        grad_clip_norm=trainer_cfg.get("grad_clip_norm"),
        log_every=trainer_cfg.get("log_every", 50),
        eval_every=trainer_cfg.get("eval_every", 1),
        save_every=trainer_cfg.get("save_every", 0),
        save_best=trainer_cfg.get("save_best", True),
        save_last=trainer_cfg.get("save_last", False),
        checkpoint_dir=resolved_checkpoint_dir,
        save_events=trainer_cfg.get("save_events", True),
        device=run_cfg.get("device", "auto"),
        seed=run_cfg.get("seed", 0),
        deterministic=run_cfg.get("deterministic", False),
    )


def build_trainer_from_config(
    model: DigitalDRNNet,
    config: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
) -> BPTrainer:
    algorithm_cfg = config.get("algorithm", {})
    algorithm_name = algorithm_cfg.get("name", "bp")
    run_cfg = dict(config.get("config", {}))
    output_dir = run_cfg.get("output_dir")

    trainer_config = build_trainer_config_from_config(config, checkpoint_dir=checkpoint_dir)
    run_metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "cwd": str(Path.cwd()),
        "package_root": str(_package_root()),
        "algorithm_name": config.get("algorithm", {}).get("name"),
        "data_name": config.get("data", {}).get("name"),
        "model_name": config.get("model", {}).get("name"),
        "selection": dict(config.get("_selection", {})),
        "save_enabled": bool(run_cfg.get("save", True)),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "resolved_output_root": str(_resolve_output_root(output_dir)) if output_dir is not None else None,
        "checkpoint_dir": str(trainer_config.checkpoint_dir) if trainer_config.checkpoint_dir is not None else None,
    }
    if algorithm_name == "bp":
        return BPTrainer(
            model,
            trainer_config,
            experiment_config=dict(config),
            run_metadata=run_metadata,
        )
    if algorithm_name == "ep":
        algorithm_runtime_cfg = dict(algorithm_cfg.get("config", {}))
        return HybridEPTrainer(
            model,
            trainer_config,
            experiment_config=dict(config),
            run_metadata=run_metadata,
            beta=float(algorithm_runtime_cfg.get("beta", 1.0e-3)),
            nudging_mode=str(algorithm_runtime_cfg.get("nudging_mode", "current")),
            amp_gradient_compensation=bool(algorithm_runtime_cfg.get("ad_hoc_amp_gradient_scale", False)),
        )
    raise NotImplementedError(f"Algorithm '{algorithm_name}' is not implemented in digital_drn yet.")


def build_dataloaders_from_config(
    config: Mapping[str, Any],
    *,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    data_section = config.get("data", {})
    dataset_name = data_section.get("name")
    data_cfg = dict(data_section.get("config", {}))

    if dataset_name not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    batch_size = data_cfg.get("batch_size", 128)
    num_workers = data_cfg.get("num_workers", 0)
    data_root = _resolve_data_root(data_cfg.get("data_root"))
    train_subset = data_cfg.get("train_subset")
    test_subset = data_cfg.get("test_subset")
    seed = config.get("config", {}).get("seed")

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    train_transform = build_image_transforms(dataset_name, train=True)
    eval_transform = build_image_transforms(dataset_name, train=False)
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=str(data_root), train=True, download=download, transform=train_transform)
        test_dataset = datasets.MNIST(root=str(data_root), train=False, download=download, transform=eval_transform)
    else:
        train_dataset = datasets.CIFAR10(root=str(data_root), train=True, download=download, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=str(data_root), train=False, download=download, transform=eval_transform)

    if train_subset is not None:
        train_dataset = Subset(train_dataset, list(range(min(int(train_subset), len(train_dataset)))))
    if test_subset is not None:
        test_dataset = Subset(test_dataset, list(range(min(int(test_subset), len(test_dataset)))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )
    return train_loader, eval_loader


def build_experiment(
    *,
    config_name: str = "config",
    config_dir: str | Path | None = None,
    algorithm: str | None = None,
    trainer: str | None = None,
    data: str | None = None,
    model: str | None = None,
    checkpoint_dir: str | Path | None = None,
    build_data: bool = True,
    download: bool = False,
) -> ExperimentBundle:
    composed = load_experiment_config(
        config_name=config_name,
        config_dir=config_dir,
        algorithm=algorithm,
        trainer=trainer,
        data=data,
        model=model,
    )
    built_model = build_model_from_config(composed)
    built_trainer = build_trainer_from_config(built_model, composed, checkpoint_dir=checkpoint_dir)
    train_loader = eval_loader = None
    if build_data:
        train_loader, eval_loader = build_dataloaders_from_config(composed, download=download)
    return ExperimentBundle(
        config=composed,
        model=built_model,
        trainer=built_trainer,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
