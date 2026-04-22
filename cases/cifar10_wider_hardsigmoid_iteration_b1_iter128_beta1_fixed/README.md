# Hard-Sigmoid Iteration/B Cosine Sweep

Uses the trained wider hard-sigmoid checkpoint and varies only solver iterations and current_amp=B.

- checkpoint: `/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- voltage_amp A: `4.0`
- beta: `1.0`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Summary

| iters | B | A/B | overall cos | ff cos | drive cos | flat DRN cos | local DRN cos | weight mean cos | bias mean cos | DRN EP/BP | disp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 4 | 0.8105 | 0.8885 | 0.8429 | 0.8291 | 0.9984 | 0.9960 | 1.0000 | 0.05821 | 0.01936 |

CSV outputs:

- `results/summary.csv`
- `results/tensor_details.csv`
