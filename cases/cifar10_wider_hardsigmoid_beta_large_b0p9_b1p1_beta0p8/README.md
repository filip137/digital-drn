# Hard-Sigmoid A/B Amplification Scaling Grid

Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.

- checkpoint: `/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- beta: `0.8`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Grid Summary

| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.9 | 4.444 | 0.4144 | 0.06023 | 0.9952 | 0.02551 | 0.2289 |
| 4 | 1.1 | 3.636 | 0.4881 | 0.07782 | 0.9853 | 0.052 | 0.2471 |

## Fitted Scaling

Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.

| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drn_all` | 0.4047 | 1.277 | 1.000 | nan | nan | nan |
| `block_0/drn/W0` | 2.349 | 4.602 | 1.000 | nan | nan | nan |
| `block_0/drn/W1` | 2.819 | 4.687 | 1.000 | nan | nan | nan |
| `block_0/drn/b0` | 1.001 | 4.001 | 1.000 | nan | nan | nan |
| `block_0/drn/b1` | 0.969 | 2.926 | 1.000 | nan | nan | nan |
| `block_0/drn/b2` | 1 | 2.000 | 1.000 | nan | nan | nan |
| `block_1/drn/W0` | 11.6 | 3.656 | 1.000 | nan | nan | nan |
| `block_1/drn/W1` | 6.495 | 3.299 | 1.000 | nan | nan | nan |
| `block_1/drn/b0` | 1 | 2.000 | 1.000 | nan | nan | nan |
| `block_1/drn/b1` | 0.8804 | 0.825 | 1.000 | nan | nan | nan |
| `block_1/drn/b2` | 1 | 0.000 | 1.000 | nan | nan | nan |

## Interpretation

- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.
- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.
- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.
