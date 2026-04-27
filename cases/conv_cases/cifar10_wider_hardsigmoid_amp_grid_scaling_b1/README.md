# Hard-Sigmoid A/B Amplification Scaling Grid

Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- beta: `0.01`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Grid Summary

| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 2 | 0.7773 | 0.1745 | 0.9926 | 0.17 | 0.3751 |
| 3 | 1 | 3 | 0.5863 | 0.09219 | 0.9880 | 0.06881 | 0.2747 |
| 4 | 1 | 4 | 0.4555 | 0.06796 | 0.9820 | 0.03766 | 0.238 |

## Fitted Scaling

Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.

| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drn_all` | 0.4416 | 1.374 | 0.989 | -1.374 | 0.000 | 0.989 |
| `block_0/drn/W0` | 0.974 | 3.952 | 1.000 | -3.952 | 0.000 | 1.000 |
| `block_0/drn/W1` | 1.12 | 4.025 | 1.000 | -4.025 | 0.000 | 1.000 |
| `block_0/drn/b0` | 0.9985 | 3.999 | 1.000 | -3.999 | 0.000 | 1.000 |
| `block_0/drn/b1` | 1.062 | 2.992 | 1.000 | -2.992 | 0.000 | 1.000 |
| `block_0/drn/b2` | 0.9996 | 2.000 | 1.000 | -2.000 | 0.000 | 1.000 |
| `block_1/drn/W0` | 1.134 | 1.953 | 1.000 | -1.953 | 0.000 | 1.000 |
| `block_1/drn/W1` | 0.9771 | 1.936 | 1.000 | -1.936 | 0.000 | 1.000 |
| `block_1/drn/b0` | 1 | 2.000 | 1.000 | -2.000 | 0.000 | 1.000 |
| `block_1/drn/b1` | 1.09 | 0.975 | 1.000 | -0.975 | 0.000 | 1.000 |
| `block_1/drn/b2` | 1 | 0.000 | 0.920 | -0.000 | 0.000 | 0.920 |

## Interpretation

- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.
- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.
- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.
