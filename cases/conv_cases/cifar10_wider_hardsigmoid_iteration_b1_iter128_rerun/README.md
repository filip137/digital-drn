# Hard-Sigmoid Iteration/B Cosine Sweep

Uses the trained wider hard-sigmoid checkpoint and varies only solver iterations and current_amp=B.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- voltage_amp A: `4.0`
- beta: `0.01`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Summary

| iters | B | A/B | overall cos | ff cos | drive cos | flat DRN cos | local DRN cos | weight mean cos | bias mean cos | DRN EP/BP | disp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 4 | 0.7911 | 0.8606 | 0.8235 | 0.1156 | 0.4913 | 0.2900 | 0.6255 | 0.166 | 0.001599 |

CSV outputs:

- `results/summary.csv`
- `results/tensor_details.csv`
