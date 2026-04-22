# EP vs BP Gradient Mismatch

## Goal

Summarize what is currently known about the mismatch between EP and BP gradients in the CIFAR widened DRN experiments.

This note focuses on three questions:

1. Is the mismatch directional or mostly a scale problem?
2. Is the mismatch coming from the FF path or the DRN path?
3. How does the mismatch change with `voltage_amp/current_amp` and with analog depth?

## Executive summary

At this point the big picture is:

1. BP and EP often point in nearly the same direction, but their norms can still be badly wrong.
2. Large residual currents show that the amplified free states are often not true equilibria of the declared energy.
3. The layer naming / indexing logic in the amplified code path is still too fragile and should be replaced by an explicit per-block depth index.

So cosine similarity by itself was too optimistic. The real failure mode is:

- direction can look acceptable
- norm scaling can still be wrong
- and the state itself can still violate the equilibrium condition `dE / ds = 0`

## Main conclusion

There are two different mismatch modes in this codebase:

1. A fixed implementation bug:
   - the EP resistive-parameter path was accidentally averaging over batch twice
   - this produced an extra `1 / batch_size` shrinkage in EP DRN gradients
2. A remaining modeling / normalization effect:
   - when `voltage_amp != current_amp`, the explicit resistive scaling factors introduce a depth-dependent shrinkage
   - this shrinkage compounds across analog depth and becomes severe for deeper analog stacks
3. A remaining equilibrium / bookkeeping concern:
   - in the amplified hard-sigmoid regime, the free-state residual current can stay large
   - the amplified code still relies on literal layer names and name-parsed depth in several places
   - this is not yet the proven root cause of the residual-current failure, but it is the wrong abstraction and should be removed

So the original mismatch was not one thing. Part of it was a real bug, and part of it is now clearly tied to the chosen amplitude normalization.

## Code locations

### Fixed batch-reduction bug

The old problem came from the interaction helper using a mean for parameter gradients:

- [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L28)
- [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L38)

In the EP block helper, the parameter-gradient path now explicitly rescales to batch-summed energy:

- [`training/ep_block.py`](/home/filip/digital_drn/training/ep_block.py#L37)

This is the current fix:

- EP parameter gradients use `grad_param_fn(...) * batch_size`
- this removes the extra `1 / B` collapse without materializing the full summed conv energy

### Remaining amplitude-dependent scaling

The resistive interactions explicitly scale by powers of `current_amp / voltage_amp`:

- dense interaction energy:
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L80)
- dense parameter view:
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L140)
- conv weight gradient:
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L292)

This is the source of the staircase pattern described below.

### Remaining naming / indexing fragility

The amplified resistive code still relies on literal layer names and on parsing the last character of the layer name:

- literal special-casing:
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L70)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L121)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L131)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L169)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L255)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L269)
- depth inferred from `name[-1]`:
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L81)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L141)
  - [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py#L293)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L127)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L173)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L214)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L245)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L283)

Current block builders partially mask the old global-counter build-order bug by reassigning names per block:

- dense block layers are renamed in [`energy/block_energy.py`](/home/filip/digital_drn/energy/block_energy.py#L250)
- conv block layers are renamed in [`energy/block_energy.py`](/home/filip/digital_drn/energy/block_energy.py#L370)

So the exact old build-order failure from `server_code` is probably not the main cause of the current widened-CIFAR issue. But the code is still relying on the wrong mechanism. Layer depth should be carried explicitly, not reconstructed from mutable names.

## Case A: original widened compare before the batch fix

Artifacts:

- [`cases/cifar10_wider_ep_vs_bp_100epochs/summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_100epochs/summary.json)
- [`cases/cifar10_wider_ep_vs_bp_100epochs/tensor_norm_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_100epochs/tensor_norm_summary.json)

Setup:

- widened CIFAR DRN config
- `beta = 1e-2`
- `128/128` CIFAR overfit gate
- `100` epochs
- one analog conv weight per block
- batch size `16`

Observed overall metrics:

- EP overall cosine: `0.4704`
- BP overall cosine: `0.4727`
- EP displacement mean: `1.06e-3`
- BP displacement mean: `1.68e-3`

Tensor-level norm ratios (`EP / BP`) for DRN tensors were nearly uniform:

- `0.0614`
- `0.0622`
- `0.0622`
- `0.0624`
- `0.0625`
- `0.0627`

Interpretation:

- This was the signature of an extra `1 / 16` factor.
- The batch size in this run was `16`.
- Multiplying the measured DRN ratios by `16` brought them back to approximately `1.0`.

This identified the old bug as a batch-reduction mismatch, not a mysterious architecture effect.

## Case B: after the batch fix, with `voltage_amp = 4`, `current_amp = 1`

Artifacts:

- [`cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs/group_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs/group_summary.json)
- [`cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs/tensor_norm_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs/tensor_norm_summary.json)

Setup:

- same widened CIFAR gate
- `beta = 1e-2`
- `voltage_amp = 4`
- `current_amp = 1`
- one analog conv weight per block

Observed overall metrics:

- EP overall cosine: `0.6045`
- BP overall cosine: `0.5558`
- EP displacement mean: `2.26e-3`
- BP displacement mean: `2.36e-3`

Group-level view:

- `head` matched exactly
- `block_0/ff` and `block_0/drive` had cosine `~1.0` but norm ratio near `1/16`
- `block_1/ff` and `block_1/drive` had cosine `~1.0` but norm ratio near `1/4`
- `block_0/drn` cosine was `0.80` to `0.83`
- `block_1/drn` cosine was `0.98`

The tensor-level DRN pattern was:

| Block | W0 | b0 | b1 |
|---|---:|---:|---:|
| 1 | `0.0624` | `0.0622` | `0.2496` |
| 2 | `0.2505` | `0.2489` | `1.0000` |

Interpretation:

- The old batch bug was gone.
- The remaining mismatch was now a depth-dependent staircase:
  - first analog stage: about `1/16`
  - second analog stage input-side terms: about `1/4`
  - last bias: about `1`
- This matches the explicit powers of `(current_amp / voltage_amp) = 1/4` in the resistive code.

So after the bug fix, the dominant remaining problem became amplitude scaling, not batch reduction.

## Residual-current evidence

Artifacts:

- [`cases/cifar10_wider_hardsigmoid_va_sweep_bp_100epochs/residual_currents_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_va_sweep_bp_100epochs/residual_currents_summary.json)
- [`cases/cifar10_wider_hardsigmoid_iteration_sweep/summary.json`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_iteration_sweep/summary.json)

Definition used here:

- true residual current is the full block-state derivative `dE / ds`
- implementation:
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L25)
  - [`core/interaction.py`](/home/filip/digital_drn/core/interaction.py#L339)

For the amplified hard-sigmoid BP sweep with `g_on = 10`, `g_off = 1e-7`, `v_off = 2`, `init_drive_scale = 2`:

- `va = 1`: overall residual RMS `0.0347`
- `va = 2`: overall residual RMS `0.603`
- `va = 4`: overall residual RMS `1.709`
- `va = 8`: overall residual RMS `5.772`

So residual current gets worse monotonically as amplification increases.

The frozen-checkpoint iteration sweep showed something even more important:

- increasing the number of solver iterations reduced free-vs-nudged displacement
- but increased the true residual current
- and degraded the BP-vs-EP cosine

This means the amplified hard-sigmoid dynamics are not simply "under-converged." In that regime, taking more minimizer steps does not move the state toward a cleaner stationary point of the declared energy.

This is the strongest current reason to distrust cosine-only diagnostics. Gradient direction can still look acceptable while the state itself is not close to `dE / ds = 0`.

## Case C: two analog conv weights per block

Artifacts:

- [`cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs_2analogconv/group_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs_2analogconv/group_summary.json)
- [`cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs_2analogconv/tensor_norm_summary.json`](/home/filip/digital_drn/cases/cifar10_wider_ep_vs_bp_va4_ca1_100epochs_2analogconv/tensor_norm_summary.json)

Setup:

- same widened CIFAR gate
- `beta = 1e-2`
- `voltage_amp = 4`
- `current_amp = 1`
- now with two analog conv weights per block

Observed overall metrics:

- EP overall cosine: `0.4015`
- BP overall cosine: `0.5228`
- EP displacement mean: `0.2621`
- BP displacement mean: `0.4796`

This was a major degradation. The displacement blow-up was concentrated in block 2:

- EP block 0 displacement: `0.0137`
- EP block 1 displacement: `0.2642`
- BP block 0 displacement: `0.0084`
- BP block 1 displacement: `0.4808`

The tensor-level DRN staircase became much steeper.

EP-trained checkpoint:

| Block | W0 | W1 | b0 | b1 | b2 |
|---|---:|---:|---:|---:|---:|
| 1 | `0.00373` | `0.00375` | `0.00401` | `0.01575` | `0.06231` |
| 2 | `0.06957` | `0.08280` | `0.06248` | `0.26558` | `1.00317` |

BP-trained checkpoint:

| Block | W0 | W1 | b0 | b1 | b2 |
|---|---:|---:|---:|---:|---:|
| 1 | `0.00391` | `0.00417` | `0.00398` | `0.01629` | `0.06442` |
| 2 | `0.07345` | `0.09155` | `0.06401` | `0.27150` | `1.01230` |

Interpretation:

- The mismatch now compounds across analog depth.
- In block 1, the earliest tensors are down near `4e-3`, which is about `1/256`.
- Later tensors then climb in powers of about `4`, ending near `1`.
- This is consistent with repeated powers of `current_amp / voltage_amp = 1/4` across the deeper analog stack.

So with deeper analog blocks, the explicit amplitude normalization is strong enough to dominate both the gradient comparison and the state displacement.

## What is actually mismatching?

The answer depends on which regime you look at.

Before the batch fix:

- the main issue was a uniform DRN scale collapse from an implementation bug

After the batch fix, with `va=4`, `ca=1`:

- the main issue is still mostly scale, not pure sign reversal
- the head remains exact
- many FF / drive groups still have cosine near `1.0`
- the DRN groups show both scale mismatch and some directional degradation, especially in the earlier analog stages

The practical message is:

- low overall cosine is not telling a single story
- part of it can come from exact or near-exact directions with wrong norm
- deeper analog stacks make the norm distortion much worse

## Current understanding

The current evidence supports this interpretation:

1. The old uniform `~1/16` DRN mismatch was a real bug and has been fixed.
2. The remaining mismatch for `voltage_amp != current_amp` is structurally tied to the resistive scaling convention.
3. That scaling compounds with analog depth.
4. Large residual currents in the amplified hard-sigmoid regime show that the state is often not near a true equilibrium of the declared energy.
5. The current name-based layer bookkeeping is still fragile and should be replaced by explicit per-block depth indexing.
6. One analog conv per block is still usable for diagnostic work.
7. Two analog convs per block with `va=4`, `ca=1` are already in a bad regime for EP-vs-BP agreement.
8. On the deeper hard-sigmoid EP checkpoint, a direct beta-by-amplification sweep shows that once `beta >= 1e-3`, the no-amplification case is already good and the dominant remaining failure mode is amplification.
   - supporting case:
     [`cases/cifar10_wider_hardsigmoid_ep_checkpoint_beta_amp_sweep/README.md`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_checkpoint_beta_amp_sweep/README.md)

## Recommendations

For future EP-vs-BP diagnostics:

1. Use `voltage_amp = 1`, `current_amp = 1` unless the goal is specifically to study amplitude scaling.
2. Treat `va != ca` as a separate experiment family, not as a small hyperparameter tweak.
3. For deeper analog stacks, always inspect tensor-level `EP / BP` norm ratios, not only overall cosine.
4. Keep the batch-reduction regression test in place:
   - [`tests/test_ep_block.py`](/home/filip/digital_drn/tests/test_ep_block.py)

For future code work:

1. Establish a real per-block `layer_index` attribute and stop using literal names like `Layer_0` / `Layer_1` as logic.
2. Remove all `name[-1]` depth parsing and replace it with the explicit depth index.
3. Decide whether the amplitude scaling in [`core/resistive.py`](/home/filip/digital_drn/core/resistive.py) is intended to be part of the optimization target or should be normalized away in EP/BP comparisons.
4. If the goal is closer EP-to-BP equivalence, the next place to intervene is the explicit `(current_amp / voltage_amp) ** layer_pre_index` scaling.
5. Keep residual current `dE / ds` as a first-class diagnostic beside cosine and displacement; any amplified regime with large residual current should be treated as suspect even if cosine looks decent.
