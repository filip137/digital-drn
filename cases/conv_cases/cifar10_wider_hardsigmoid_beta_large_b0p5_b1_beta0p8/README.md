# Hard-Sigmoid A/B Amplification Scaling Grid

Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- beta: `0.8`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Grid Summary

| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5 | 8 | 0.3258 | 0.01506 | 0.8647 | 0.009973 | 0.1888 |
| 4 | 1 | 4 | 0.4574 | 0.06773 | 0.9924 | 0.03653 | 0.238 |

## Fitted Scaling

Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.

| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drn_all` | 1.37 | 2.169 | 1.000 | nan | nan | nan |
| `block_0/drn/W0` | 1.385 | 4.225 | 1.000 | nan | nan | nan |
| `block_0/drn/W1` | 3.103 | 4.761 | 1.000 | nan | nan | nan |
| `block_0/drn/b0` | 1.09 | 4.062 | 1.000 | nan | nan | nan |
| `block_0/drn/b1` | 1.14 | 3.043 | 1.000 | nan | nan | nan |
| `block_0/drn/b2` | 1.078 | 2.054 | 1.000 | nan | nan | nan |
| `block_1/drn/W0` | 0.3649 | 1.174 | 1.000 | nan | nan | nan |
| `block_1/drn/W1` | 4.859 | 3.099 | 1.000 | nan | nan | nan |
| `block_1/drn/b0` | 1.065 | 2.045 | 1.000 | nan | nan | nan |
| `block_1/drn/b1` | 1.161 | 1.020 | 1.000 | nan | nan | nan |
| `block_1/drn/b2` | 1.083 | 0.057 | 1.000 | nan | nan | nan |

## Interpretation

- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.
- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.
- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.
