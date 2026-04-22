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
| 128 | 0.25 | 16 | 0.6852 | -0.0028 | 0.1016 | -0.0027 | 0.1235 | -0.0529 | 0.2411 | 0.1025 | 0.001208 |
| 128 | 0.5 | 8 | 0.5783 | 0.3066 | 0.7619 | 0.0147 | 0.3868 | 0.1219 | 0.5634 | 0.1318 | 0.001371 |
| 128 | 1 | 4 | 0.7911 | 0.8606 | 0.8235 | 0.1156 | 0.4913 | 0.2900 | 0.6255 | 0.166 | 0.001599 |
| 128 | 2 | 2 | 0.7561 | 0.9888 | -0.1999 | 0.3217 | 0.4971 | 0.3875 | 0.5701 | 0.3624 | 0.004538 |

CSV outputs:

- `results/summary.csv`
- `results/tensor_details.csv`
