import torch
import torch.nn as nn
import pytest

from digital_drn import BlockEquilibriumProp, build_dense_drn_block
from digital_drn.core.parameter import Bias, DenseWeight


def _configure_stable_dense_block(block):
    with torch.no_grad():
        block.ff[1].weight.fill_(0.1)
        block.ff[1].bias.zero_()
        for weight in block.energy.dense_weights:
            weight.state.fill_(0.1)
        for bias in block.energy.biases:
            bias.state.zero_()


def _mean_state_distance(states, reference_states):
    distances = [
        (state.detach() - reference.detach()).float().square().mean().sqrt()
        for state, reference in zip(states, reference_states)
    ]
    return float(torch.stack(distances).mean().item())


def _block_ep_param_grads(block, head, criterion, inputs, targets, *, beta: float):
    h_free = block(inputs, reset=True, num_iterations=4)
    cache = block.capture_free_cache(inputs)

    h_leaf = h_free.detach().clone().requires_grad_(True)
    logits = head(h_leaf)
    loss = criterion(logits, targets)
    (delta_h,) = torch.autograd.grad(loss, (h_leaf,))

    return BlockEquilibriumProp(block, beta=beta).compute_gradients(
        free_cache=cache,
        output_cotangent=delta_h,
    ).param_grads


def test_block_free_cache_round_trip_restores_drive_and_state():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block)

    x = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.0, -0.5]], dtype=torch.float32)
    _ = block(x, reset=True, num_iterations=2)
    cache = block.capture_free_cache(x)

    with torch.no_grad():
        for layer in block.free_layers():
            layer.state.add_(1.0)
        block.energy.set_drive(torch.zeros_like(cache.drive))

    block.restore_free_cache(cache)

    assert torch.allclose(block.energy.drive.current, cache.drive)
    assert torch.allclose(block.output_state(), cache.output)
    assert all(
        torch.allclose(layer.state, cached)
        for layer, cached in zip(block.free_layers(), cache.free_state)
    )


def test_block_equilibrium_prop_returns_centered_grads_and_delta_drive():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=4,
        mode="asynchronous",
        non_linearity="linear",
        ff_learning_rate=1.0e-3,
        drn_learning_rate=1.0e-4,
        weight_gains=[0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block)

    head = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
            [-0.3, 0.2, 0.1],
            [0.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    h_free = block(inputs, reset=True, num_iterations=4)
    cache = block.capture_free_cache(inputs)

    h_leaf = h_free.detach().clone().requires_grad_(True)
    logits = head(h_leaf)
    loss = criterion(logits, targets)
    (delta_h,) = torch.autograd.grad(loss, (h_leaf,))

    ep = BlockEquilibriumProp(block, beta=0.1)
    result = ep.compute_gradients(free_cache=cache, output_cotangent=delta_h)

    assert len(result.param_grads) == len(block.resistive_params())
    assert all(
        grad.shape == param.state.shape
        for grad, param in zip(result.param_grads, block.resistive_params())
    )
    assert result.delta_drive.shape == cache.drive.shape
    assert any(torch.count_nonzero(grad).item() > 0 for grad in result.param_grads)
    assert torch.count_nonzero(result.delta_drive).item() > 0
    assert torch.max(torch.abs(result.plus_output - result.minus_output)).item() > 0.0
    assert torch.allclose(block.energy.drive.current, cache.drive)
    assert torch.allclose(block.output_state(), cache.output)



def test_block_equilibrium_prop_num_iterations_override_applies_to_nudged_phases():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=7,
        mode="asynchronous",
        non_linearity="linear",
        weight_gains=[0.1],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block)

    inputs = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.0, -0.5]], dtype=torch.float32)
    _ = block(inputs, reset=True, num_iterations=7)
    cache = block.capture_free_cache(inputs)
    output_cotangent = torch.ones_like(cache.output)

    seen_iterations = []
    original_compute = block.training_minimizer.compute_equilibrium

    def wrapped_compute_equilibrium():
        seen_iterations.append(block.training_minimizer.num_iterations)
        original_compute()

    block.training_minimizer.compute_equilibrium = wrapped_compute_equilibrium

    BlockEquilibriumProp(block, beta=1.0e-2, num_iterations=3).compute_gradients(
        free_cache=cache,
        output_cotangent=output_cotangent,
    )

    assert seen_iterations == [3, 3]
    assert block.training_minimizer.num_iterations == 7

def test_nudged_states_stay_close_to_free_state_for_small_beta():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=4,
        mode="asynchronous",
        non_linearity="linear",
        ff_learning_rate=1.0e-3,
        drn_learning_rate=1.0e-4,
        weight_gains=[0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block)

    head = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
            [-0.3, 0.2, 0.1],
            [0.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    h_free = block(inputs, reset=True, num_iterations=4)
    cache = block.capture_free_cache(inputs)

    h_leaf = h_free.detach().clone().requires_grad_(True)
    logits = head(h_leaf)
    loss = criterion(logits, targets)
    (delta_h,) = torch.autograd.grad(loss, (h_leaf,))

    small = BlockEquilibriumProp(block, beta=1.0e-3).compute_gradients(
        free_cache=cache,
        output_cotangent=delta_h,
    )
    large = BlockEquilibriumProp(block, beta=1.0e-1).compute_gradients(
        free_cache=cache,
        output_cotangent=delta_h,
    )

    small_plus_dist = _mean_state_distance(small.plus_state, cache.free_state)
    small_minus_dist = _mean_state_distance(small.minus_state, cache.free_state)
    large_plus_dist = _mean_state_distance(large.plus_state, cache.free_state)
    large_minus_dist = _mean_state_distance(large.minus_state, cache.free_state)
    small_pair_dist = _mean_state_distance(small.plus_state, small.minus_state)
    large_pair_dist = _mean_state_distance(large.plus_state, large.minus_state)

    assert small_plus_dist > 0.0
    assert small_minus_dist > 0.0
    assert small_plus_dist < 1.0
    assert small_minus_dist < 1.0
    assert large_plus_dist > 0.0
    assert large_minus_dist > 0.0
    assert large_pair_dist > small_pair_dist
    assert torch.allclose(block.energy.drive.current, cache.drive)
    assert torch.allclose(block.output_state(), cache.output)


def test_ep_resistive_param_grads_do_not_pick_up_extra_batch_average():
    torch.manual_seed(0)

    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=4,
        mode="asynchronous",
        non_linearity="linear",
        ff_learning_rate=1.0e-3,
        drn_learning_rate=1.0e-4,
        weight_gains=[0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))
    _configure_stable_dense_block(block)

    head = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()

    single_inputs = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
    single_targets = torch.tensor([0], dtype=torch.long)

    repeated_inputs = single_inputs.repeat(4, 1)
    repeated_targets = single_targets.repeat(4)

    single_grads = _block_ep_param_grads(
        block,
        head,
        criterion,
        single_inputs,
        single_targets,
        beta=0.1,
    )
    repeated_grads = _block_ep_param_grads(
        block,
        head,
        criterion,
        repeated_inputs,
        repeated_targets,
        beta=0.1,
    )

    single_total_norm = float(torch.cat([grad.reshape(-1) for grad in single_grads]).norm().item())
    repeated_total_norm = float(torch.cat([grad.reshape(-1) for grad in repeated_grads]).norm().item())
    assert repeated_total_norm == pytest.approx(single_total_norm, rel=5.0e-2)

    compared = 0
    for single_grad, repeated_grad in zip(single_grads, repeated_grads):
        single_norm = float(single_grad.detach().float().norm().item())
        repeated_norm = float(repeated_grad.detach().float().norm().item())
        if single_norm <= 1.0e-6:
            continue
        compared += 1
        assert repeated_norm == pytest.approx(single_norm, rel=5.0e-2)
    assert compared > 0


def test_amp_gradient_compensation_scales_weights_biases_and_drive_staircase():
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 4, 2],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        voltage_amp=4.0,
        current_amp=1.0,
        weight_gains=[0.2, 0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    ep = BlockEquilibriumProp(
        block,
        beta=0.1,
        amp_gradient_compensation=True,
        downstream_weight_count=2,
    )
    scales, drive_scale = ep._amp_compensation_scales()

    expected = [4.0**4, 4.0**4, 4.0**4, 4.0**3]
    assert scales == pytest.approx(expected)
    assert drive_scale == pytest.approx(4.0**4)


def test_amp_gradient_compensation_is_identity_when_amps_match():
    block = build_dense_drn_block(
        input_dim=3,
        layer_dims=[4, 2],
        num_iterations=2,
        mode="asynchronous",
        non_linearity="linear",
        voltage_amp=1.0,
        current_amp=1.0,
        weight_gains=[0.2],
        bias_gain=0.0,
    ).set_device(torch.device("cpu"))

    ep = BlockEquilibriumProp(
        block,
        beta=0.1,
        amp_gradient_compensation=True,
        downstream_weight_count=3,
    )
    scales, drive_scale = ep._amp_compensation_scales()
    assert scales == pytest.approx([1.0] * len(block.resistive_params()))
    assert drive_scale == pytest.approx(1.0)
