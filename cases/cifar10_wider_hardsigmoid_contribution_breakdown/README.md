# Hard-Sigmoid Contribution Breakdown

This case breaks down the BP-vs-EP mismatch on the deeper widened hard-sigmoid EP checkpoint into separate `ff`, `drive`, and `drn` contributions, and compares four amplified variants at `voltage_amp = 4`, `current_amp = 1`.

Artifacts:

- script:
  [`run_contribution_checks.py`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_contribution_breakdown/run_contribution_checks.py)
- results:
  [`summary.json`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_contribution_breakdown/results/summary.json)

Checkpoint under test:

- [`checkpoint_best.pt`](/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt)

## Main Result

At `beta = 1e-2`:

| Variant | overall cosine | ff cosine | drive cosine | drn cosine | drn relerr |
| --- | ---: | ---: | ---: | ---: | ---: |
| `va1_comp_on_layer1_on` | `0.9972` | `1.0000` | `1.0000` | `0.9963` | `0.09` |
| `va4_comp_on_layer1_on` | `0.8081` | `0.9940` | `0.9862` | `0.8834` | `13.23` |
| `va4_comp_off_layer1_on` | `0.4969` | `0.5705` | `0.4887` | `0.4558` | `0.97` |
| `va4_comp_on_layer1_off` | `0.6373` | `0.9829` | `0.9902` | `0.7323` | `174.76` |
| `va4_comp_off_layer1_off` | `0.7178` | `0.8954` | `0.9242` | `0.8393` | `0.83` |

So the picture is:

1. No amplification is clean.
2. In the amplified baseline, `ff` and `drive` keep very good direction, while `drn` is the weaker path.
3. Turning the ad-hoc EP compensation off removes the giant norm blow-up, but directional agreement collapses.
4. Turning off amplification on the current `Layer_1` improves the raw amplified direction if compensation is also off.
5. But turning off `Layer_1` amplification while keeping the old compensation on makes the norm mismatch catastrophically worse.

## Tensor-Level Pattern

For the original amplified baseline (`va4_comp_on_layer1_on`, `beta = 1e-2`), the block-0 DRN tensors are all about `16x` too large:

- `block_0/W0`: cosine `0.9993`, `EP/BP ≈ 16.24`
- `block_0/W1`: cosine `0.9853`, `EP/BP ≈ 17.29`
- `block_0/b0`: cosine `1.0000`, `EP/BP ≈ 16.01`
- `block_0/b1`: cosine `0.9983`, `EP/BP ≈ 17.20`
- `block_0/b2`: cosine `0.9998`, `EP/BP ≈ 16.00`

while the later block is much closer:

- `block_1/W0`: cosine `0.8814`, `EP/BP ≈ 1.24`
- `block_1/W1`: cosine `0.9935`, `EP/BP ≈ 1.06`
- `block_1/b0`: cosine `0.9997`, `EP/BP ≈ 1.00`
- `block_1/b1`: cosine `0.9981`, `EP/BP ≈ 1.13`
- `block_1/b2`: cosine `1.0000`, `EP/BP ≈ 1.00`

With compensation off, the same tensors are mostly too small instead:

- in `block_0`, most DRN tensors are around `EP/BP ≈ 1/256` to `1/16`
- in `block_1`, most DRN tensors are around `EP/BP ≈ 1/16` to `1/4`
- the final bias `b2` is already near `1`

This is the same staircase pattern seen in the earlier amplified cases.

## Interpretation

The new evidence points to two distinct issues:

1. The raw amplified EP gradient is still directionally wrong.
   - This is why `va4_comp_off_layer1_on` has poor cosine even though its relative error is near `1`.
2. The ad-hoc EP compensation is hard-coded to the old amplification staircase.
   - This is why `va4_comp_on_layer1_off` explodes so badly.
   - Once the effective first-layer amplification rule changes, the compensation scales no longer match the actual energy scaling.

The most likely next fix is:

1. stop hard-coding the EP amplitude compensation from block depth alone
2. derive the compensation exponents from the actual resistive interactions in use
3. include the `amplify_first_free_layer` rule in that derivation

Until then, the safest diagnostic regime remains:

- `voltage_amp = 1`
- `current_amp = 1`
- `beta >= 1e-3`
