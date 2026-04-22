import json
from pathlib import Path
import socket

import torch.nn as nn
import pytest
from torchvision import transforms

from digital_drn import (
    BPTrainer,
    HybridEPTrainer,
    SequentialDigitalDRNNet,
    build_dataloaders_from_config,
    build_experiment,
    build_image_transforms,
    build_model_from_config,
    build_trainer_config_from_config,
    build_trainer_from_config,
    load_experiment_config,
)
from digital_drn.models.digital_modules import CrossEntropyReadoutHead
from digital_drn.models.network_digital_analog import DigitalAnalogNet


def test_load_experiment_config_composes_defaults_and_allows_model_override():
    cfg = load_experiment_config()
    assert cfg["algorithm"]["name"] == "bp"
    assert cfg["trainer"]["epochs"] == 200
    assert cfg["data"]["name"] == "mnist"
    assert cfg["model"]["name"] == "sequential_digital_drn"
    assert cfg["_selection"]["model"] == "dense_1block"

    overridden = load_experiment_config(model="dense_3block")
    assert len(overridden["model"]["blocks_config"]) == 3
    assert overridden["_selection"]["model"] == "dense_3block"


def test_custom_top_level_config_inherits_shared_run_config():
    cfg = load_experiment_config(config_name="cifar10_drn_only_signed_norm_readout_wider")
    assert cfg["config"]["device"] == "cuda"
    assert cfg["config"]["save"] is True
    assert cfg["config"]["output_dir"] == "simulation_results"


def test_build_model_from_nested_yaml_config_dense_and_conv():
    dense_cfg = load_experiment_config(model="dense_3block")
    dense_model = build_model_from_config(dense_cfg)
    assert isinstance(dense_model, SequentialDigitalDRNNet)
    assert len(dense_model.blocks) == 3
    assert dense_model.blocks[0].minimizer.mode == "asynchronous"
    assert dense_model.blocks[0].minimizer.num_iterations == 4
    assert dense_model.blocks[0].energy.dense_weights[0].min_cond == 1.0e-7
    assert dense_model.blocks[0].energy.dense_weights[0].max_cond == 10.0

    conv_cfg = load_experiment_config(model="conv_mnist_1block")
    conv_model = build_model_from_config(conv_cfg)
    assert isinstance(conv_model.blocks[0].ff[0], nn.Conv2d)
    assert conv_model.blocks[0].ff_activation_name == "tanh"

    cifar_debug_cfg = load_experiment_config(model="cifar10_digital_analog_v0_overfit_debug", data="cifar10")
    cifar_debug_model = build_model_from_config(cifar_debug_cfg)
    assert len(cifar_debug_model.blocks) == 2

    cifar_1block_cfg = load_experiment_config(model="cifar10_digital_analog_v0_1block", data="cifar10")
    cifar_1block_model = build_model_from_config(cifar_1block_cfg)
    assert len(cifar_1block_model.blocks) == 1

    conv_cifar_cfg = load_experiment_config(model="conv_cifar10_1block", data="cifar10")
    conv_cifar_model = build_model_from_config(conv_cifar_cfg)
    assert len(conv_cifar_model.blocks) == 1
    assert conv_cifar_model.blocks[0].output_dim == 10

    conv_cifar_single_cfg = load_experiment_config(model="conv_cifar10_1block_single_conv", data="cifar10")
    conv_cifar_single_model = build_model_from_config(conv_cifar_single_cfg)
    assert len(conv_cifar_single_model.blocks) == 1

    mixed_two_block_cfg = load_experiment_config(model="cifar10_mixed_analog_2block_readout", data="cifar10")
    mixed_two_block_model = build_model_from_config(mixed_two_block_cfg)
    assert len(mixed_two_block_model.blocks) == 2

    mixed_two_block_head_cfg = load_experiment_config(model="cifar10_mixed_analog_2block_plus_digital_head", data="cifar10")
    mixed_two_block_head_model = build_model_from_config(mixed_two_block_head_cfg)
    assert len(mixed_two_block_head_model.blocks) == 2
    assert mixed_two_block_head_model.head is not None
    assert isinstance(mixed_two_block_head_model.head, CrossEntropyReadoutHead)

    drn_only_signed_cfg = load_experiment_config(model="cifar10_drn_only_signed_norm_readout", data="cifar10")
    drn_only_signed_model = build_model_from_config(drn_only_signed_cfg)
    assert len(drn_only_signed_model.blocks) == 2
    assert drn_only_signed_model.head is not None
    assert drn_only_signed_model.blocks[0].signed_drive is True
    assert drn_only_signed_model.blocks[1].signed_drive is True
    assert drn_only_signed_model.blocks[0].layer_shapes == ((6, 32, 32), (32, 32, 32))
    assert drn_only_signed_model.blocks[1].layer_shapes == ((64, 16, 16), (64, 16, 16))
    assert drn_only_signed_model.blocks[0]._drive_scale_raw.requires_grad is False
    assert drn_only_signed_model.blocks[1]._drive_scale_raw.requires_grad is False
    assert isinstance(drn_only_signed_model.head.module[1], nn.Linear)
    assert drn_only_signed_model.head.module[1].in_features == 16384
    assert drn_only_signed_model.head.module[1].out_features == 256

    drn_only_signed_wider_cfg = load_experiment_config(
        config_name="cifar10_drn_only_signed_norm_readout_wider"
    )
    drn_only_signed_wider_model = build_model_from_config(drn_only_signed_wider_cfg)
    assert drn_only_signed_wider_cfg["algorithm"]["config"]["beta"] == pytest.approx(1.0e-2)
    assert drn_only_signed_wider_cfg["trainer"]["optimizer"]["lr"] == pytest.approx(1.0e-4)
    assert drn_only_signed_wider_cfg["model"]["blocks_config"][0]["ff"]["lr"] == pytest.approx(1.0e-3)
    assert drn_only_signed_wider_cfg["model"]["blocks_config"][0]["drn"]["lr"] == pytest.approx(1.0e-4)
    assert drn_only_signed_wider_cfg["model"]["blocks_config"][1]["ff"]["lr"] == pytest.approx(1.0e-3)
    assert drn_only_signed_wider_cfg["model"]["blocks_config"][1]["drn"]["lr"] == pytest.approx(1.0e-4)
    assert drn_only_signed_wider_cfg["model"]["head"]["lr"] == pytest.approx(3.0e-4)
    assert len(drn_only_signed_wider_model.blocks) == 2
    assert drn_only_signed_wider_model.head is not None
    assert drn_only_signed_wider_model.blocks[0].signed_drive is True
    assert drn_only_signed_wider_model.blocks[1].signed_drive is True
    assert drn_only_signed_wider_model.blocks[0].layer_shapes == ((6, 32, 32), (32, 32, 32), (64, 32, 32))
    assert drn_only_signed_wider_model.blocks[1].layer_shapes == ((128, 16, 16), (256, 16, 16), (512, 16, 16))
    assert drn_only_signed_wider_model.blocks[0]._drive_scale_raw.requires_grad is True
    assert drn_only_signed_wider_model.blocks[1]._drive_scale_raw.requires_grad is True
    assert isinstance(drn_only_signed_wider_model.head.module[0], nn.MaxPool2d)
    assert isinstance(drn_only_signed_wider_model.head.module[2], nn.Linear)
    assert drn_only_signed_wider_model.head.module[2].in_features == 32768
    assert drn_only_signed_wider_model.head.module[2].out_features == 256

    drn_only_signed_wider_hardsig_cfg = load_experiment_config(
        config_name="cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc"
    )
    drn_only_signed_wider_hardsig_model = build_model_from_config(drn_only_signed_wider_hardsig_cfg)
    assert drn_only_signed_wider_hardsig_cfg["algorithm"]["config"]["ad_hoc_amp_gradient_scale"] is True
    assert drn_only_signed_wider_hardsig_cfg["model"]["config"]["drn_non_linearity"] == "hard_sigmoid"
    assert drn_only_signed_wider_hardsig_cfg["model"]["config"]["voltage_amp"] == pytest.approx(4.0)
    assert drn_only_signed_wider_hardsig_cfg["model"]["config"]["current_amp"] == pytest.approx(1.0)
    assert len(drn_only_signed_wider_hardsig_model.blocks) == 2
    assert drn_only_signed_wider_hardsig_model.blocks[0].layer_shapes == ((6, 32, 32), (32, 32, 32), (64, 32, 32))
    assert drn_only_signed_wider_hardsig_model.blocks[1].layer_shapes == ((128, 16, 16), (256, 16, 16), (512, 16, 16))

    drn_only_signed_wider_cfg["model"]["blocks_config"][0]["drn"]["output_non_linearity"] = "linear"
    drn_only_signed_wider_cfg["model"]["blocks_config"][1]["drn"]["output_non_linearity"] = "linear"
    drn_only_signed_wider_output_linear_model = build_model_from_config(drn_only_signed_wider_cfg)
    assert drn_only_signed_wider_output_linear_model.blocks[0].energy.free_layers()[0].non_linearity == "perfect_diode"
    assert drn_only_signed_wider_output_linear_model.blocks[0].energy.free_layers()[-1].non_linearity == "linear"
    assert drn_only_signed_wider_output_linear_model.blocks[1].energy.free_layers()[0].non_linearity == "perfect_diode"
    assert drn_only_signed_wider_output_linear_model.blocks[1].energy.free_layers()[-1].non_linearity == "linear"

    drn_only_signed_wider_cfg["model"]["blocks_config"][0]["drn"]["amplify_first_free_layer"] = False
    drn_only_signed_wider_cfg["model"]["blocks_config"][1]["drn"]["amplify_first_free_layer"] = False
    drn_only_signed_wider_no_first_amp_model = build_model_from_config(drn_only_signed_wider_cfg)
    first_interaction = next(
        interaction
        for interaction in drn_only_signed_wider_no_first_amp_model.blocks[0].energy._interactions
        if interaction.__class__.__name__ == "ConvResistive"
    )
    assert first_interaction._amplify_first_free_layer is False

    v0_mlp_head_cfg = load_experiment_config(model="cifar10_digital_analog_v0_mlp_head", data="cifar10")
    v0_mlp_head_model = build_model_from_config(v0_mlp_head_cfg)
    assert len(v0_mlp_head_model.blocks) == 2
    assert v0_mlp_head_model.head is not None

    cifar_mlp_cfg = load_experiment_config(model="cifar10_digital_mlp_v0", data="cifar10")
    cifar_mlp_model = build_model_from_config(cifar_mlp_cfg)
    assert len(cifar_mlp_model.blocks) == 2
    assert isinstance(cifar_mlp_model.blocks[0].module[0], nn.Conv2d)

    cifar_mlp_head_baseline_cfg = load_experiment_config(model="cifar10_digital_mlp_head_baseline", data="cifar10")
    cifar_mlp_head_baseline_model = build_model_from_config(cifar_mlp_head_baseline_cfg)
    assert isinstance(cifar_mlp_head_baseline_model, DigitalAnalogNet)
    assert len(cifar_mlp_head_baseline_model.blocks) == 2
    assert isinstance(cifar_mlp_head_baseline_model.blocks[0].module[0], nn.Conv2d)
    assert isinstance(cifar_mlp_head_baseline_model.blocks[0].module[2], nn.Tanh)
    assert cifar_mlp_head_baseline_model.head is not None
    assert isinstance(cifar_mlp_head_baseline_model.head.module[1], nn.Linear)
    assert cifar_mlp_head_baseline_model.head.module[1].in_features == 16384
    assert cifar_mlp_head_baseline_model.head.module[1].out_features == 256
    assert isinstance(cifar_mlp_head_baseline_model.head.module[2], nn.Tanh)
    assert isinstance(cifar_mlp_head_baseline_model.head.module[3], nn.Linear)
    assert cifar_mlp_head_baseline_model.head.module[3].out_features == 10


def test_build_trainer_from_config_and_bundle(tmp_path: Path):
    cfg = load_experiment_config(model="dense_1block")
    trainer_cfg = build_trainer_config_from_config(cfg, checkpoint_dir=tmp_path)
    assert trainer_cfg.checkpoint_dir == tmp_path
    assert trainer_cfg.device == cfg["config"]["device"]
    assert trainer_cfg.scheduler is not None
    assert trainer_cfg.scheduler.name == "cosine"
    assert trainer_cfg.scheduler.config["T_max"] == trainer_cfg.epochs
    assert trainer_cfg.optimizer.config["weight_decay"] == 3.0e-4

    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg, checkpoint_dir=tmp_path)
    assert isinstance(trainer, BPTrainer)
    assert trainer.config.checkpoint_dir == tmp_path
    assert (tmp_path / "trainer_config.json").exists()
    assert (tmp_path / "experiment_config.json").exists()
    assert (tmp_path / "run_metadata.json").exists()
    run_metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert run_metadata["checkpoint_dir"] == str(tmp_path)
    assert run_metadata["output_dir"] == "simulation_results"
    assert run_metadata["resolved_output_root"].endswith("/simulation_results")

    bundle = build_experiment(model="dense_1block", checkpoint_dir=tmp_path, build_data=False)
    assert isinstance(bundle.model, SequentialDigitalDRNNet)
    assert isinstance(bundle.trainer, BPTrainer)
    assert bundle.train_loader is None
    assert bundle.eval_loader is None
    trainer.close()
    bundle.trainer.close()


def test_default_checkpoint_dir_uses_model_name_and_timestamped_run_dir():
    cfg = load_experiment_config(model="conv_mnist_1block")
    trainer_cfg = build_trainer_config_from_config(cfg)
    checkpoint_dir = Path(trainer_cfg.checkpoint_dir)
    hostname = socket.gethostname()

    assert checkpoint_dir.parent.name == "conv_mnist_1block"
    assert checkpoint_dir.parent.parent.name == "simulation_results"
    assert checkpoint_dir.name.endswith(hostname)
    assert len(checkpoint_dir.name) > len(hostname) + 1


def test_block_level_ff_and_drn_learning_rates_flow_into_optimizer_groups():
    cfg = load_experiment_config(model="dense_1block")
    cfg["trainer"]["optimizer"]["lr"] = 1.0e-4
    cfg["model"]["blocks_config"][0]["ff"]["lr"] = 3.0e-4
    cfg["model"]["blocks_config"][0]["drn"]["lr"] = 1.0e-5

    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)

    lrs = [group["lr"] for group in trainer.optimizer.param_groups]
    assert lrs == [3.0e-4, 1.0e-5]


def test_build_trainer_from_config_builds_ep_trainer():
    cfg = load_experiment_config(algorithm="ep", model="cifar10_digital_analog_v0_1block", data="cifar10")
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    assert isinstance(trainer, HybridEPTrainer)
    assert trainer.beta == pytest.approx(1.0e-3)
    trainer.close()

    adhoc_cfg = load_experiment_config(config_name="cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc")
    adhoc_model = build_model_from_config(adhoc_cfg)
    adhoc_trainer = build_trainer_from_config(adhoc_model, adhoc_cfg)
    assert isinstance(adhoc_trainer, HybridEPTrainer)
    assert adhoc_trainer.amp_gradient_compensation is True
    adhoc_trainer.close()


def test_build_trainer_from_config_builds_bp_trainer_for_pure_digital_cifar_baseline():
    cfg = load_experiment_config(model="cifar10_digital_mlp_head_baseline", data="cifar10")
    model = build_model_from_config(cfg)
    trainer = build_trainer_from_config(model, cfg)
    assert isinstance(trainer, BPTrainer)
    trainer.close()


def test_build_dataloaders_configures_cifar10_transforms():
    cfg = load_experiment_config(model="cifar10_digital_analog_v0_mlp_head", data="cifar10")
    train_transform = build_image_transforms("cifar10", train=True)
    eval_transform = build_image_transforms("cifar10", train=False)

    assert isinstance(train_transform.transforms[0], transforms.RandomHorizontalFlip)
    assert isinstance(train_transform.transforms[1], transforms.RandomCrop)
    assert train_transform.transforms[1].padding == 4
    assert train_transform.transforms[1].padding_mode == "edge"
    assert isinstance(train_transform.transforms[2], transforms.ToTensor)
    assert isinstance(train_transform.transforms[3], transforms.Normalize)
    assert isinstance(eval_transform.transforms[0], transforms.ToTensor)
    assert isinstance(eval_transform.transforms[1], transforms.Normalize)
