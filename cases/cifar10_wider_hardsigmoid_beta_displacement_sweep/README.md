# Hard-Sigmoid Beta Displacement Sweep

This case checks beta sensitivity after fixing the iteration override in hybrid EP.

- checkpoint: `/home/filip/digital_drn/cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- voltage_amp A: `4.0`
- current_amp B: `1.0`
- A/B: `4.0`
- num_iterations: `128`
- amp_gradient_compensation: `False`
- amplify_first_free_layer: `True`

## Why This Case Exists

An earlier 128-iteration diagnostic was invalid because `num_iterations=128` was applied to ordinary BP and the EP free phase, but the `+beta` and `-beta` nudged phases still used the block-local training minimizer default of 6 iterations.
That has been fixed by passing the override into `BlockEquilibriumProp` and temporarily applying it to the nudged training minimizer. The fixed 128-step run is slower, as expected, because both nudged phases now also run 128 relaxation iterations.

## Beta Sweep

| beta | disp mean | disp + | disp - | overall cos | flat DRN cos | local DRN cos | weight mean cos | bias mean cos | DRN EP/BP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-06 | 0.0193602 | 0.0193602 | 0.0193602 | 0.4240 | 0.0109 | 0.0313 | 0.0275 | 0.0344 | 2.061 |
| 3e-06 | 0.0193602 | 0.0193602 | 0.0193602 | 0.1911 | 0.0010 | -0.0010 | 0.0172 | -0.0131 | 5.181 |
| 1e-05 | 0.0193602 | 0.0193602 | 0.0193602 | 0.1605 | 0.0059 | 0.0096 | 0.0284 | -0.0030 | 6.354 |
| 3e-05 | 0.0193602 | 0.0193602 | 0.0193602 | 0.2273 | -0.0021 | 0.0488 | 0.0076 | 0.0763 | 4.262 |
| 0.0001 | 0.0193602 | 0.0193602 | 0.0193602 | 0.2811 | 0.0122 | 0.1653 | -0.0188 | 0.2880 | 3.482 |
| 0.0003 | 0.0193602 | 0.0193602 | 0.0193602 | 0.3905 | 0.0524 | 0.4107 | 0.0453 | 0.6542 | 2.501 |
| 0.001 | 0.0193602 | 0.0193602 | 0.0193602 | 0.5788 | 0.0451 | 0.5556 | 0.1230 | 0.8440 | 1.282 |
| 0.003 | 0.0193602 | 0.0193601 | 0.0193602 | 0.7409 | 0.0824 | 0.6497 | 0.2260 | 0.9321 | 0.5748 |
| 0.01 | 0.0193602 | 0.01936 | 0.0193603 | 0.8010 | 0.2293 | 0.8015 | 0.5236 | 0.9867 | 0.2161 |
| 0.03 | 0.0193602 | 0.0193598 | 0.0193606 | 0.8090 | 0.4750 | 0.8819 | 0.7075 | 0.9982 | 0.1012 |
| 0.1 | 0.0193602 | 0.0193589 | 0.0193615 | 0.8104 | 0.7387 | 0.9579 | 0.8951 | 0.9998 | 0.06514 |
| 0.3 | 0.0193603 | 0.0193563 | 0.0193643 | 0.8105 | 0.8157 | 0.9910 | 0.9776 | 1.0000 | 0.05915 |
| 1 | 0.0193614 | 0.019348 | 0.0193749 | 0.8105 | 0.8291 | 0.9984 | 0.9960 | 1.0000 | 0.05821 |
| 3 | 0.0193714 | 0.0193311 | 0.0194117 | 0.8105 | 0.8303 | 0.9992 | 0.9980 | 1.0000 | 0.05807 |
| 10 | 0.0194842 | 0.019351 | 0.0196175 | 0.8105 | 0.8304 | 0.9993 | 0.9983 | 1.0000 | 0.05805 |
| 30 | 0.0204448 | 0.020068 | 0.0208216 | 0.8105 | 0.8305 | 0.9993 | 0.9983 | 1.0000 | 0.05805 |
| 100 | 0.0290605 | 0.0282157 | 0.0299053 | 0.8105 | 0.8305 | 0.9993 | 0.9983 | 1.0000 | 0.05807 |
| 300 | 0.0670165 | 0.066093 | 0.0679399 | 0.8104 | 0.8301 | 0.9992 | 0.9981 | 0.9999 | 0.0577 |

## Displacement Change Point

Mean relative displacement first changed by more than 1% at `beta=30` relative to `beta=1e-06`.
Mean relative displacement first changed by more than 10% at `beta=100` relative to `beta=1e-06`.

## Interpretation

- For the amplified `A=4, B=1` 128-step diagnostic, increasing beta mainly improves the finite-difference gradient direction rather than changing the pooled free-vs-nudged displacement over the tested range.
- The direction improvement is concentrated in DRN weights. Bias directions are already near perfect once the iteration mismatch is fixed.
- The remaining EP/BP norm mismatch follows the amplification staircase and should be handled separately from direction cosine.

## Related Fixed Single-Point Cases

- `cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_fixed`: `A=4, B=1, beta=1e-2`; local DRN cosine improved after the iteration fix but deeper block weights were still weak.
- `cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta0p1_fixed`: `A=4, B=1, beta=0.1`; DRN directions became much cleaner.
- `cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta1_fixed`: `A=4, B=1, beta=1`; DRN tensor directions were nearly perfect, while norm scaling remained amplified.
- `cases/cifar10_wider_hardsigmoid_iteration_a1_b1_iter128_fixed`: `A=1, B=1, beta=1e-2`; unamplified 128-step directions were mostly healthy, with weakness mainly in block-1 weights.

CSV outputs:

- `results/summary.csv`
- `results/tensor_details.csv`
