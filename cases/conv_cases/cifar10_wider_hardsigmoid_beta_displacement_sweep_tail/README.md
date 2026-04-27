# Hard-Sigmoid Beta Displacement Sweep

This case checks beta sensitivity after fixing the iteration override in hybrid EP.

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
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
| 30 | 0.0204448 | 0.020068 | 0.0208216 | 0.8105 | 0.8305 | 0.9993 | 0.9983 | 1.0000 | 0.05805 |
| 100 | 0.0290605 | 0.0282157 | 0.0299053 | 0.8105 | 0.8305 | 0.9993 | 0.9983 | 1.0000 | 0.05807 |
| 300 | 0.0670165 | 0.066093 | 0.0679399 | 0.8104 | 0.8301 | 0.9992 | 0.9981 | 0.9999 | 0.0577 |

## Displacement Change Point

Mean relative displacement first changed by more than 1% at `beta=100` relative to `beta=30`.
Mean relative displacement first changed by more than 10% at `beta=100` relative to `beta=30`.

## Interpretation

- For the amplified `A=4, B=1` 128-step diagnostic, increasing beta mainly improves the finite-difference gradient direction rather than changing the pooled free-vs-nudged displacement over the tested range.
- The direction improvement is concentrated in DRN weights. Bias directions are already near perfect once the iteration mismatch is fixed.
- The remaining EP/BP norm mismatch follows the amplification staircase and should be handled separately from direction cosine.

## Related Fixed Single-Point Cases

- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_fixed`: `A=4, B=1, beta=1e-2`; local DRN cosine improved after the iteration fix but deeper block weights were still weak.
- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta0p1_fixed`: `A=4, B=1, beta=0.1`; DRN directions became much cleaner.
- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_b1_iter128_beta1_fixed`: `A=4, B=1, beta=1`; DRN tensor directions were nearly perfect, while norm scaling remained amplified.
- `cases/conv_cases/cifar10_wider_hardsigmoid_iteration_a1_b1_iter128_fixed`: `A=1, B=1, beta=1e-2`; unamplified 128-step directions were mostly healthy, with weakness mainly in block-1 weights.

CSV outputs:

- `results/summary.csv`
- `results/tensor_details.csv`
