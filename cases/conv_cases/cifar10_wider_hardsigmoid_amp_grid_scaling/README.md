# Hard-Sigmoid A/B Amplification Scaling Grid

Uses the existing trained small checkpoint shape and varies only `voltage_amp=A` and `current_amp=B`.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- beta: `0.01`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Grid Summary

| A | B | A/B | flat DRN cos | DRN EP/BP | local DRN cos | weight EP/BP mean | bias EP/BP mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5 | 4 | 0.2234 | 0.06402 | 0.7597 | 0.05071 | 0.2381 |
| 2 | 0.75 | 2.667 | 0.6469 | 0.1037 | 0.9543 | 0.07329 | 0.2985 |
| 2 | 2 | 1 | 0.7686 | 2.239 | 0.9373 | 4.224 | 1.037 |
| 3 | 0.5 | 6 | 0.1091 | 0.04935 | 0.7080 | 0.04673 | 0.2024 |
| 3 | 0.75 | 4 | 0.4455 | 0.05634 | 0.9342 | 0.03746 | 0.2389 |
| 3 | 2 | 1.5 | 0.7739 | 0.7922 | 0.9359 | 1.244 | 0.5269 |
| 4 | 0.5 | 8 | 0.1390 | 0.03605 | 0.7111 | 0.04827 | 0.1894 |
| 4 | 0.75 | 5.333 | 0.3252 | 0.04173 | 0.8760 | 0.0281 | 0.2157 |
| 4 | 2 | 2 | 0.6977 | 0.4786 | 0.9349 | 0.5727 | 0.377 |

## Fitted Scaling

Fit form: `EP/BP ~= C * (A/B)^(-k)`. Suggested multiplier to align norms is therefore `C^-1 * (A/B)^k` for that target.

| target | C | k | R2 | two-factor A exp | two-factor B exp | two-factor R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drn_all` | 1.62 | 2.107 | 0.925 | -1.463 | 2.260 | 0.945 |
| `block_0/drn/W0` | 1.456 | 3.782 | 0.945 | -3.183 | 3.926 | 0.951 |
| `block_0/drn/W1` | 8.603 | 5.403 | 0.982 | -4.117 | 5.710 | 0.996 |
| `block_0/drn/b0` | 0.9934 | 3.990 | 1.000 | -3.982 | 3.992 | 1.000 |
| `block_0/drn/b1` | 1.107 | 3.018 | 1.000 | -3.020 | 3.017 | 1.000 |
| `block_0/drn/b2` | 0.9866 | 1.977 | 1.000 | -1.976 | 1.977 | 1.000 |
| `block_1/drn/W0` | 1.67 | 1.555 | 0.745 | -0.932 | 1.704 | 0.774 |
| `block_1/drn/W1` | 4.609 | 2.986 | 0.948 | -1.839 | 3.260 | 0.981 |
| `block_1/drn/b0` | 0.9919 | 1.987 | 1.000 | -1.984 | 1.988 | 1.000 |
| `block_1/drn/b1` | 1.132 | 0.998 | 1.000 | -1.015 | 0.994 | 1.000 |
| `block_1/drn/b2` | 1.009 | 0.016 | 0.471 | -0.017 | 0.016 | 0.471 |


## Practical Scaling Recommendation

For one global DRN norm correction on this checkpoint/grid, the fitted aggregate law is:

```text
EP/BP ~= 1.62 * (A/B)^(-2.11)
```

so the corresponding multiplier for EP DRN gradients is approximately:

```text
scale_global_drn ~= 0.62 * (A/B)^2.11
```

This is only an aggregate correction. The tensor-level fits show a depth staircase, especially for biases:

```text
block_0: b0 -> (A/B)^4, b1 -> (A/B)^3, b2 -> (A/B)^2
block_1: b0 -> (A/B)^2, b1 -> (A/B)^1, b2 -> (A/B)^0
```

For weights, the fitted exponents are less clean, but the useful empirical values are:

```text
block_0/W0: k ~= 3.8
block_0/W1: k ~= 5.4
block_1/W0: k ~= 1.6  # weaker fit
block_1/W1: k ~= 3.0
```

So if the goal is to align tensor-local norms, a single global multiplier is not enough. Use tensor- or layer-depth-specific powers of `A/B`; if the goal is just to align the flattened DRN norm, use the global `0.62 * (A/B)^2.11` rule as a first approximation.

## Interpretation

- If the two-factor fit has `A exponent ~= -B exponent`, the scaling mostly depends on `A/B`.
- Different tensors have different exponents, so a single global correction can align one aggregate but cannot align all tensor norms simultaneously.
- Use tensor-specific or parameter-class-specific powers if the goal is to preserve tensor-local directions while fixing norms.
