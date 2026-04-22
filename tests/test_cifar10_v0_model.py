import torch

from digital_drn import ConvDRNBlockEnergy, DigitalAnalogNet, build_model_from_config, load_experiment_config
from digital_drn.blocks import DigitalDRNBlock


def test_build_cifar10_v0_model_from_yaml():
    cfg = load_experiment_config(model="cifar10_digital_analog_v0", data="cifar10")
    model = build_model_from_config(cfg)

    assert isinstance(model, DigitalAnalogNet)
    assert len(model.blocks) == 2
    assert all(isinstance(block, DigitalDRNBlock) for block in model.blocks)
    assert all(isinstance(block.energy, ConvDRNBlockEnergy) for block in model.blocks)
    assert model.head is not None


def test_cifar10_v0_forward_produces_logits():
    cfg = load_experiment_config(model="cifar10_digital_analog_v0", data="cifar10")
    model = build_model_from_config(cfg)

    x = torch.randn(2, 3, 32, 32)
    logits = model(x, reset=True, num_iterations=1)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
