# Tensor Direction Summary

Derived from `summary.json`; no BP/EP run was repeated.

This separates local tensor direction from flattened cross-tensor scaling. A flattened cosine can be low when each tensor has high local cosine but the EP/BP norm ratio differs strongly across tensors.

## Variant Summary

| variant | beta | flattened overall | flattened DRN | DRN local mean | DRN weight mean | DRN bias mean | weight EP/BP range | bias EP/BP range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `va1_comp_on_layer1_on` | 0.001 | 0.9933 | 0.9911 | 0.9841 | 0.9615 | 0.9991 | 0.9806-1.185 | 1-1.097 |
| `va1_comp_on_layer1_on` | 0.01 | 0.9972 | 0.9963 | 0.9921 | 0.9812 | 0.9994 | 0.9763-1.097 | 1-1.097 |
| `va4_comp_on_layer1_on` | 0.001 | 0.7986 | 0.8730 | 0.9158 | 0.7992 | 0.9935 | 1.131-17.6 | 1-17.04 |
| `va4_comp_on_layer1_on` | 0.01 | 0.8081 | 0.8834 | 0.9855 | 0.9649 | 0.9993 | 1.061-17.29 | 1-17.2 |
| `va4_comp_off_layer1_on` | 0.001 | 0.4953 | 0.4020 | 0.8543 | 0.6567 | 0.9861 | 0.004348-0.2202 | 0.004026-1 |
| `va4_comp_off_layer1_on` | 0.01 | 0.4969 | 0.4558 | 0.9827 | 0.9580 | 0.9992 | 0.003998-0.0778 | 0.00391-1 |
| `va4_comp_on_layer1_off` | 0.001 | 0.6371 | 0.7320 | 0.9432 | 0.8653 | 0.9952 | 4.489-275.4 | 1-257.2 |
| `va4_comp_on_layer1_off` | 0.01 | 0.6373 | 0.7323 | 0.9918 | 0.9806 | 0.9993 | 4.231-276 | 1-256.5 |
| `va4_comp_off_layer1_off` | 0.001 | 0.7050 | 0.7306 | 0.9219 | 0.8127 | 0.9947 | 0.06742-0.5563 | 0.06297-1 |
| `va4_comp_off_layer1_off` | 0.01 | 0.7178 | 0.8393 | 0.9915 | 0.9797 | 0.9993 | 0.06542-0.2935 | 0.06262-1 |

## Focus: `va4_comp_off_layer1_on`, beta = 1e-2

- flattened overall cosine: `0.4969`
- flattened DRN cosine: `0.4558`
- local DRN tensor mean cosine: `0.9827`
- local DRN weight mean cosine: `0.9580`
- local DRN bias mean cosine: `0.9992`
- weight EP/BP norm-ratio range: `0.003998` to `0.0778`
- bias EP/BP norm-ratio range: `0.00391` to `1`

Interpretation: the low flattened cosine is mostly a structured scaling/staircase effect across tensors, not uniformly poor local tensor direction.
