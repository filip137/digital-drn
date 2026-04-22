# Hard-Sigmoid Norm Mismatch Report

This note focuses only on gradient magnitudes, not direction.

Question:

- how wrong are the EP gradient norms relative to BP on the deeper widened hard-sigmoid checkpoint?

Source artifacts:

- contribution breakdown summary:
  [`summary.json`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_contribution_breakdown/results/summary.json)
- original beta x amplification sweep:
  [`summary.json`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_checkpoint_beta_amp_sweep/summary.json)

Checkpoint under test:

- [`checkpoint_best.pt`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt)

All numbers below use:

- deeper widened network
- `hard_sigmoid`
- `beta = 1e-2`

Metric:

- `EP/BP = ||g_ep|| / ||g_bp||`
- ideal value is `1`

## Executive Summary

The norm mismatch is small without amplification and severe with amplification.

- no amplification:
  - `ff_all ≈ 1.0008`
  - `drive_all ≈ 1.0002`
  - `drn_all ≈ 1.0162`
- amplified baseline (`va = 4`, compensation on):
  - `ff_all ≈ 15.89`
  - `drive_all ≈ 15.76`
  - `drn_all ≈ 14.10`
- amplified baseline (`va = 4`, compensation off):
  - `ff_all ≈ 0.0083`
  - `drive_all ≈ 0.0117`
  - `drn_all ≈ 0.0679`

So the practical picture is:

1. Without amplification, norms are basically correct.
2. With amplification and the current ad-hoc compensation enabled, EP is much too large.
3. With amplification and compensation disabled, EP is much too small.
4. The magnitude error is structured, not uniform.

## Path-Level Norm Ratios

At `beta = 1e-2`:

| Variant | `ff_all` EP/BP | `drive_all` EP/BP | `drn_all` EP/BP |
| --- | ---: | ---: | ---: |
| `va1_comp_on_layer1_on` | `1.0008` | `1.0002` | `1.0162` |
| `va4_comp_on_layer1_on` | `15.8879` | `15.7567` | `14.0986` |
| `va4_comp_off_layer1_on` | `0.0083` | `0.0117` | `0.0679` |
| `va4_comp_on_layer1_off` | `252.5629` | `254.8119` | `175.4834` |
| `va4_comp_off_layer1_off` | `0.0776` | `0.0723` | `0.2080` |

Interpretation:

- `va = 1` is clean.
- the original amplified compensated baseline overshoots by about `14x` to `16x`.
- the raw amplified un-compensated baseline undershoots by about `12x` to `120x`, depending on the path.
- disabling amplification on the current `Layer_1` while keeping the old compensation makes the overshoot catastrophic.
- disabling both compensation and `Layer_1` amplification makes the norms less wrong than the raw baseline, but still far from `1`.

## Tensor-Level DRN Norm Ratios

### No Amplification

`va1_comp_on_layer1_on`

Block 0:

- `W0 ≈ 1.0169`
- `W1 ≈ 1.0838`
- `b0 ≈ 1.0011`
- `b1 ≈ 1.0729`
- `b2 ≈ 1.0005`

Block 1:

- `W0 ≈ 1.0970`
- `W1 ≈ 0.9763`
- `b0 ≈ 1.0002`
- `b1 ≈ 1.0965`
- `b2 ≈ 1.0000`

This is the reference good regime.

### Amplified Baseline, Compensation On

`va4_comp_on_layer1_on`

Block 0:

- `W0 ≈ 16.2366`
- `W1 ≈ 17.2894`
- `b0 ≈ 16.0117`
- `b1 ≈ 17.1991`
- `b2 ≈ 16.0049`

Block 1:

- `W0 ≈ 1.2447`
- `W1 ≈ 1.0607`
- `b0 ≈ 1.0002`
- `b1 ≈ 1.1290`
- `b2 ≈ 1.0000`

This is the important amplified pattern:

- the first DRN block is wrong by about `16x`
- the second DRN block is only mildly wrong

### Amplified Baseline, Compensation Off

`va4_comp_off_layer1_on`

Block 0:

- `W0 ≈ 0.0040`
- `W1 ≈ 0.0042`
- `b0 ≈ 0.0039`
- `b1 ≈ 0.0168`
- `b2 ≈ 0.0625`

Block 1:

- `W0 ≈ 0.0778`
- `W1 ≈ 0.0663`
- `b0 ≈ 0.0625`
- `b1 ≈ 0.2823`
- `b2 ≈ 1.0000`

This is the staircase pattern:

- early tensors are tiny
- later tensors climb toward `1`
- the final bias is already correct

### Amplification Removed From Current `Layer_1`, Compensation Still On

`va4_comp_on_layer1_off`

Block 0:

- `W0 ≈ 264.1764`
- `W1 ≈ 275.9947`
- `b0 ≈ 256.4719`
- `b1 ≈ 68.8571`
- `b2 ≈ 64.0534`

Block 1:

- `W0 ≈ 4.6952`
- `W1 ≈ 4.2306`
- `b0 ≈ 4.0012`
- `b1 ≈ 1.1270`
- `b2 ≈ 1.0000`

This shows that the current ad-hoc compensation assumes the old amplified staircase. Once the first-layer amplification rule changes, the compensation badly over-corrects.

### Amplification Removed From Current `Layer_1`, Compensation Off

`va4_comp_off_layer1_off`

Block 0:

- `W0 ≈ 0.0654`
- `W1 ≈ 0.0674`
- `b0 ≈ 0.0626`
- `b1 ≈ 0.0672`
- `b2 ≈ 0.2502`

Block 1:

- `W0 ≈ 0.2935`
- `W1 ≈ 0.2644`
- `b0 ≈ 0.2501`
- `b1 ≈ 0.2818`
- `b2 ≈ 1.0000`

This is still wrong, but less extreme than the raw amplified baseline.

## Conclusion

The norm story is:

1. `va = 1` is already good.
2. The amplified raw EP gradient is too small in a depth-structured staircase.
3. The current ad-hoc compensation over-corrects that staircase, especially in the first DRN block.
4. The first block is where the norm mismatch is worst.
5. Changing the `Layer_1` amplification rule without changing the compensation exponents makes the norm mismatch much worse, not better.

So if the user wants to fix norms only, the next likely fix is:

- derive the compensation scales from the actual effective amplification rule per interaction,
- rather than hard-coding exponents from block depth alone.
