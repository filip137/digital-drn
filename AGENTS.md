# AGENTS

## Purpose
This repository contains the self-contained `digital_drn` package for digital-to-analog DRN blocks, model builders, and training experiments.

## Repo Layout
- `core/`: low-level variables, layers, interactions, parameters, and minimizers
- `energy/`: DRN energies, augmented-energy helpers, and block-local interactions
- `blocks/`: runtime wrappers that couple FF modules to DRN energies
- `models/`: model containers and reusable digital modules/heads
- `training/`: trainer, config dataclasses, experiment builders, and smoke runs
- `app/`: CLI entrypoints
- `utils/`: data transforms and misc helpers
- `docs/`: design notes, EP notes, refactor notes, and progress notes
- `hydra_conf/`: Hydra-style experiment configs
- `tests/`: unit and integration tests
- `cases/`: experiment case reports

## Defaults
- Default DRN minimizer mode should always be `"asynchronous"` unless the user explicitly asks for a different mode for a specific experiment or test.
- For CIFAR architecture work, use the `128`-sample overfit check as the first gate before larger runs. The current case report is:
  - [`cases/conv_cases/cifar10_overfit_debug/README.md`](/home/filip/digital_drn/cases/conv_cases/cifar10_overfit_debug/README.md)
- Current overfit guidance from that case:
  - good templates: digital conv + dense DRN readout, one-block mixed analog readout, two-block mixed analog readout, and the original `v0` stack with a flat readout
  - suspicious templates: conv analog block followed directly by a digital head, pooled `64 -> 10` digital output heads, and adding an extra final digital `10 -> 10` head on top of a working analog readout
- For any new top-level training config under `hydra_conf/`, always preserve the shared run `config` block so runs keep `save: true`, `output_dir: "simulation_results"`, and device/seed defaults unless the user explicitly asks otherwise.
- When launching or reviewing runs, treat [`run_metadata.json`](/home/filip/digital_drn/training/trainer.py#L459) and the resolved `checkpoint_dir` / `output_dir` fields as required provenance. If a run has no output directory configured, fix that before starting long jobs.

## Architecture Notes

- For the current FF-to-DRN coupling and hybrid backward split, use:
  - [`docs/architecture/digital_drn_coupling_and_gradients_for_attention.md`](/home/filip/digital_drn/docs/architecture/digital_drn_coupling_and_gradients_for_attention.md)
- That note explains:
  - how `ff` acts on a DRN block through an injected current
  - how ordinary BP differs from `hybrid_backward_explicit(...)`
  - how FF / drive-scale / DRN gradients are propagated today
  - what interface a future multihead-attention frontend should satisfy

## Common Diagnostics

### Free-vs-Nudged Displacement

Use this as the default beta-displacement metric unless the user explicitly asks for a different one.

Definition:

- positive displacement:
  - \(d_+ = \sqrt{\sum \|s^+ - s^0\|^2 / \sum \|s^0\|^2}\)
- negative displacement:
  - \(d_- = \sqrt{\sum \|s^- - s^0\|^2 / \sum \|s^0\|^2}\)
- reported mean:
  - \(d_{\text{mean}} = 0.5 (d_+ + d_-)\)

where:

- `s^0` is the free equilibrium state
- `s^+` is the `+beta` nudged equilibrium
- `s^-` is the `-beta` nudged equilibrium

In `digital_drn`, the preferred implementation is the pooled block-free-state metric in:

- [`training/loaded_model_beta_sweep.py::_pool_displacement`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L212)

That helper pools across:

- all DRN free-layer tensors
- all blocks
- the full batch being analyzed

Minimal recipe:

1. Run the hybrid EP pass and keep:
   - `free_states = [cache.free_state for cache in hybrid.free_cache.block_caches]`
   - `plus_states = [result.ep.plus_state for result in hybrid.blocks]`
   - `minus_states = [result.ep.minus_state for result in hybrid.blocks]`
2. Compute pooled relative RMS displacement with `_pool_displacement(...)`.
3. Report:
   - `positive_relative_disp`
   - `negative_relative_disp`
   - `mean_relative_disp`

Reference usage:

- [`training/loaded_model_beta_sweep.py::_analyze_batch`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L242)
- [`cases/conv_cases/cifar10_overfit_debug/loaded_model_beta_sweep/README.md`](/home/filip/digital_drn/cases/conv_cases/cifar10_overfit_debug/loaded_model_beta_sweep/README.md)

Copyable pattern:

```python
def _pool_displacement(block_free_states, plus_states, minus_states):
    def _accumulate(states_a, states_b):
        total_free_sq = 0.0
        total_diff_sq = 0.0
        for free_layers, perturbed_layers in zip(states_a, states_b):
            for free, perturbed in zip(free_layers, perturbed_layers):
                free = free.detach().float()
                perturbed = perturbed.detach().float()
                diff = perturbed - free
                total_free_sq += float(torch.sum(free * free).item())
                total_diff_sq += float(torch.sum(diff * diff).item())
        return total_free_sq, total_diff_sq

    pos_free_sq, pos_diff_sq = _accumulate(block_free_states, plus_states)
    neg_free_sq, neg_diff_sq = _accumulate(block_free_states, minus_states)

    def _relative(diff_sq, free_sq):
        return math.sqrt(diff_sq / free_sq) if free_sq > 1.0e-12 else float("nan")

    return {
        "positive_relative_disp": _relative(pos_diff_sq, pos_free_sq),
        "negative_relative_disp": _relative(neg_diff_sq, neg_free_sq),
        "mean_relative_disp": 0.5 * (
            _relative(pos_diff_sq, pos_free_sq)
            + _relative(neg_diff_sq, neg_free_sq)
        ),
    }
```

If a per-block view is needed, reuse the same formula block-by-block before pooling over all blocks.

### BP-vs-EP Gradient Similarity

Use this as the default similarity diagnostic unless the user asks for a different metric.

Preferred implementation:

- [`training/loaded_model_beta_sweep.py`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py)

Core helpers:

- [`_collect_bp_groups(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L82)
- [`_collect_ep_groups(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L106)
- [`_group_metrics(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L170)
- [`_overall_metrics(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L187)
- [`_cosine(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L148)
- [`_relative_error(...)`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L159)

Definitions:

- cosine similarity:
  - \(\cos(g_{\mathrm{bp}}, g_{\mathrm{ep}}) = \langle g_{\mathrm{bp}}, g_{\mathrm{ep}} \rangle / (\|g_{\mathrm{bp}}\| \|g_{\mathrm{ep}}\|)\)
- relative error:
  - \(\|g_{\mathrm{bp}} - g_{\mathrm{ep}}\| / \|g_{\mathrm{bp}}\|\)

Edge cases in the current implementation:

- if both norms are below `1e-12`, cosine returns `1.0`
- if only one side is tiny, cosine returns `NaN`
- if the BP norm is below `1e-12`, relative error returns `NaN`

Default grouping in `digital_drn`:

- `head`
- `block_i/ff`
- `block_i/drive`
- `block_i/drn`

How BP groups are collected:

- `head`: `param.grad`
- `block_i/ff`: `param.grad`
- `block_i/drive`: `block._drive_scale_raw.grad`
- `block_i/drn`: `param.state.grad`

How EP groups are collected from the hybrid result:

- `head`: `hybrid_result.head.head_param_grads`
- `block_i/ff`: `block_result.digital.ff_param_grads`
- `block_i/drive`: `block_result.digital.drive_scale_grad`
- `block_i/drn`: `block_result.ep.param_grads`

Minimal recipe:

1. Run ordinary BP on the same batch:
   - `logits = model(batch_inputs, reset=True, num_iterations=...)`
   - `loss = criterion(logits, batch_targets)`
   - `loss.backward()`
   - collect grouped BP gradients with `_collect_bp_groups(model)`
2. Clear grads and detach state:
   - `model.zero_grad(set_to_none=True)`
   - `model.zero_resistive_grad_(set_to_none=True)`
   - `model.detach_state_()`
3. Run explicit hybrid EP on the same batch:
   - `hybrid = hybrid_backward_explicit(...)`
   - collect grouped EP gradients with `_collect_ep_groups(model, hybrid)`
4. Compare:
   - per-group cosine / relative error with `_group_metrics(...)`
   - overall cosine / relative error by flattening all groups with `_overall_metrics(...)`

Reference usage:

- [`training/loaded_model_beta_sweep.py::_analyze_batch`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py#L242)
- [`cases/conv_cases/cifar10_overfit_debug/loaded_model_beta_sweep/README.md`](/home/filip/digital_drn/cases/conv_cases/cifar10_overfit_debug/loaded_model_beta_sweep/README.md)

Copyable pattern:

```python
model.zero_grad(set_to_none=True)
model.zero_resistive_grad_(set_to_none=True)

logits = model(batch_inputs, reset=True, num_iterations=num_iterations)
loss = criterion(logits, batch_targets)
loss.backward()
bp_groups = _collect_bp_groups(model)
model.detach_state_()

model.zero_grad(set_to_none=True)
model.zero_resistive_grad_(set_to_none=True)

hybrid = hybrid_backward_explicit(
    model,
    batch_inputs,
    batch_targets,
    criterion=criterion,
    beta=beta,
    reset=True,
    num_iterations=num_iterations,
)
ep_groups = _collect_ep_groups(model, hybrid)
model.detach_state_()

group_rows = _group_metrics(bp_groups, ep_groups)
overall = _overall_metrics(bp_groups, ep_groups)
```

Interpretation defaults:

- cosine near `1`: BP and EP point in the same direction
- cosine near `0`: little directional agreement
- cosine near `-1`: likely a sign mismatch or reversed finite-difference convention
- When reporting flattened group or overall cosine, also inspect individual tensor directions whenever possible. A low flattened cosine can be caused by structured per-tensor scaling differences even if each tensor has high local cosine, because cosine is invariant to one global scale factor but not to different scale factors across tensors.
- For amplified networks, especially `voltage_amp != current_amp`, report tensor-level cosine and `EP/BP = ||g_ep|| / ||g_bp||` norm ratios for DRN weights and biases separately before concluding that BP and EP point in different directions.
- good residual current alone does **not** guarantee good BP-vs-EP agreement; keep displacement and gradient similarity as separate diagnostics

## Scope
- In scope: package source, configs, tests, and docs.
- Out of scope: generated outputs under folders like `outputs/` and `simulation_results/` unless the user explicitly asks to analyze them.
