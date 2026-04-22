# Hard-Sigmoid Size Sweep

This is a random-initialization diagnostic. No checkpoint was loaded, because wider networks do not match the existing checkpoint shapes.

- voltage_amp: `4.0`
- current_amp: `1.0`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`
- sample_indices: `[0, 1]`

| scale | beta | DRN params | flattened DRN | local DRN mean | finite tensors | weight mean | bias mean | weight EP/BP range | bias EP/BP range |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001 | 1828544 | 0.9997 | 0.6195 | 8/10 | 0.6498 | 0.6014 | 0.06251-0.4381 | 0.04348-1 |
| 1 | 0.01 | 1828544 | 0.9997 | 0.6361 | 8/10 | 0.6522 | 0.6264 | 0.06251-0.7863 | 0.0625-1 |
| 2 | 0.001 | 6636928 | 0.9999 | 0.5865 | 8/10 | 0.5663 | 0.5987 | 0.06251-0.4995 | 0.0625-1 |
| 2 | 0.01 | 6636928 | 0.9999 | 0.6114 | 8/10 | 0.5678 | 0.6375 | 0.06251-0.9386 | 0.0625-1 |
| 4 | 0.001 | 25211648 | 1.0000 | 0.6153 | 8/10 | 0.6346 | 0.6037 | 0.06251-0.392 | 0.06155-1 |
| 4 | 0.01 | 25211648 | 1.0000 | 0.5507 | 9/10 | 0.4739 | 0.6121 | 0.06251-1534 | 0.0625-2.076 |

Interpretation should focus on the local tensor means and EP/BP norm-ratio ranges, not only the flattened cosine.
