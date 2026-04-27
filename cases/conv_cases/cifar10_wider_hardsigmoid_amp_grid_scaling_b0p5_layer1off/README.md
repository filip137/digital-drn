# Hard-Sigmoid A/B Amplification Scaling Grid

Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- beta: `0.01`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `False`

## Grid Summary

| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5 | 4 | 0.2087 | 0.2281 | 0.6975 | 0.2307 | 0.2647 |
| 3 | 0.5 | 6 | 0.1312 | 0.2421 | 0.6697 | 0.3015 | 0.229 |
| 4 | 0.5 | 8 | 0.1242 | 0.1974 | 0.6320 | 0.2319 | 0.2109 |

## Fitted Scaling

Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.

| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drn_all` | 0.3067 | 0.185 | 0.377 | -0.185 | 0.613 | 0.377 |
| `block_0/drn/W0` | 0.5797 | 1.505 | 0.863 | -1.505 | 0.744 | 0.863 |
| `block_0/drn/W1` | 1.112 | 2.295 | 0.981 | -2.295 | 0.695 | 0.981 |
| `block_0/drn/b0` | 0.2316 | 1.992 | 1.000 | -1.992 | 1.331 | 1.000 |
| `block_0/drn/b1` | 0.5548 | 2.029 | 1.000 | -2.029 | 0.934 | 1.000 |
| `block_0/drn/b2` | 0.4794 | 0.967 | 1.000 | -0.967 | 0.658 | 1.000 |
| `block_1/drn/W0` | 0.3679 | -0.378 | 0.309 | 0.378 | 0.345 | 0.309 |
| `block_1/drn/W1` | 0.3716 | 0.304 | 0.392 | -0.304 | 0.562 | 0.392 |
| `block_1/drn/b0` | 0.453 | 0.919 | 1.000 | -0.919 | 0.669 | 1.000 |
| `block_1/drn/b1` | 1.208 | 1.033 | 1.000 | -1.033 | 0.247 | 1.000 |
| `block_1/drn/b2` | 1.037 | 0.026 | 0.966 | -0.026 | -0.009 | 0.966 |

## Interpretation

- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.
- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.
- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.
