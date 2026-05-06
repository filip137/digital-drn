from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import load_text_datasets
from .model import load_opt_causal_lm
from .teacher import collect_teacher_layer_activations, default_probe_layers, opt_num_layers


@dataclass
class RunningScalarStats:
    max_quantile_samples: int = 250_000
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    max_abs: float = 0.0
    _sample: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float32))

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float().reshape(-1).cpu()
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float(torch.sum(values * values).item())
        self.max_abs = max(self.max_abs, float(values.abs().max().item()))
        self._update_sample(values.abs())

    def as_dict(self) -> dict[str, float | int]:
        mean = self.total / max(1, self.count)
        mean_sq = self.total_sq / max(1, self.count)
        variance = max(0.0, mean_sq - mean * mean)
        if self._sample.numel() == 0:
            q999 = 0.0
        else:
            q999 = float(torch.quantile(self._sample, 0.999).item())
        return {
            "num_values": self.count,
            "mean": float(mean),
            "std": float(variance**0.5),
            "q0_999_abs": q999,
            "max_abs_observed": float(self.max_abs),
        }

    def _update_sample(self, values_abs: torch.Tensor) -> None:
        if self.max_quantile_samples <= 0:
            return
        if values_abs.numel() > self.max_quantile_samples:
            perm = torch.randperm(values_abs.numel())[: self.max_quantile_samples]
            values_abs = values_abs[perm]
        if self._sample.numel() == 0:
            self._sample = values_abs[: self.max_quantile_samples].clone()
            return
        sample = torch.cat((self._sample, values_abs), dim=0)
        if sample.numel() > self.max_quantile_samples:
            perm = torch.randperm(sample.numel())[: self.max_quantile_samples]
            sample = sample[perm]
        self._sample = sample.contiguous()


def calibrate_teacher(
    teacher: torch.nn.Module,
    loader: DataLoader,
    layer_indices: list[int],
    *,
    device: torch.device,
    max_batches: int,
    max_quantile_samples: int = 250_000,
    mlp_input_mode: str = "normalized",
) -> dict[str, Any]:
    stats = {
        int(layer_idx): {
            "z": RunningScalarStats(max_quantile_samples=max_quantile_samples),
            "r": RunningScalarStats(max_quantile_samples=max_quantile_samples),
        }
        for layer_idx in layer_indices
    }
    teacher.to(device)
    teacher.eval()

    batches_seen = 0
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        activations = collect_teacher_layer_activations(
            teacher,
            x.to(device),
            layer_indices,
            mlp_input_mode=mlp_input_mode,
        )
        for layer_idx, tensors in activations.items():
            stats[layer_idx]["z"].update(tensors.z)
            stats[layer_idx]["r"].update(tensors.r)
        batches_seen += 1

    return {
        "layers": {
            str(layer_idx): {
                "z": layer_stats["z"].as_dict(),
                "r": layer_stats["r"].as_dict(),
            }
            for layer_idx, layer_stats in stats.items()
        },
        "num_batches": batches_seen,
        "max_quantile_samples": int(max_quantile_samples),
        "mlp_input_mode": mlp_input_mode,
    }


def save_calibration(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scales_from_calibration(
    calibration: dict[str, Any] | None,
    layer_index: int,
    *,
    source: str = "q0_999_abs",
    default: float = 1.0,
) -> tuple[float, float]:
    if calibration is None:
        return default, default
    layer = calibration.get("layers", {}).get(str(layer_index))
    if layer is None:
        return default, default
    input_scale = float(layer.get("z", {}).get(source, default))
    output_scale = float(layer.get("r", {}).get(source, default))
    input_scale = input_scale if input_scale > 1.0e-12 else default
    output_scale = output_scale if output_scale > 1.0e-12 else default
    return input_scale, output_scale


def main() -> None:
    args = _parse_args()
    device = _get_device(args.device)
    tokenizer = "char" if args.debug and args.tokenizer == "auto" else args.tokenizer
    train_dataset, _val_dataset, _encode, _decode = load_text_datasets(
        args.data,
        model_name=args.model_name,
        tokenizer=tokenizer,
        train_frac=args.train_frac,
        block_size=args.block_size,
        vocab_size=128 if args.debug else None,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    teacher = _build_teacher(args).to(device)
    layer_indices = _parse_layers(args.layers, opt_num_layers(teacher))
    payload = calibrate_teacher(
        teacher,
        loader,
        layer_indices,
        device=device,
        max_batches=args.max_batches,
        max_quantile_samples=args.max_quantile_samples,
        mlp_input_mode=args.mlp_input_mode,
    )
    payload.update(
        {
            "model_name": args.model_name,
            "layer_indices": layer_indices,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "mlp_input_mode": args.mlp_input_mode,
        }
    )
    save_calibration(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_teacher(args: argparse.Namespace) -> torch.nn.Module:
    if not args.debug:
        return load_opt_causal_lm(args.model_name)
    try:
        from transformers import OPTConfig, OPTForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to construct debug OPT teachers.") from exc

    cfg = OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=args.block_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="relu",
        do_layer_norm_before=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        word_embed_proj_dim=32,
    )
    return OPTForCausalLM(cfg)


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "default":
        return default_probe_layers(num_layers)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    bad = [idx for idx in values if idx < 0 or idx >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}.")
    return sorted(set(values))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--layers", default="default")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--max_batches", type=int, default=128)
    parser.add_argument("--max_quantile_samples", type=int, default=250_000)
    parser.add_argument("--mlp_input_mode", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--output", type=Path, default=Path("runs/opt_mlp_drn_calibration.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max_batches must be positive.")
    return args


def _get_device(raw: str | None) -> torch.device:
    if raw is not None:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
