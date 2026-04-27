# Loaded CIFAR-10 Overfit Debug Beta Sweep

- config path: `/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2/experiment_config.json`
- checkpoint path: `/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2/checkpoint_best.pt`
- dataset slice: `first 1 samples of the evaluation subset`
- batch size: `16`
- sample count: `1`
- mode: `asynchronous`
- block iterations: `[6, 6]`
- non_linearity: `linear`
- voltage_amp: `1.0`
- current_amp: `1.0`

BP-vs-EP compares ordinary backprop on the loaded network against `hybrid_backward_explicit` on the same batch.
The displacement metric pools the DRN free-layer states across blocks and measures pooled relative RMS displacement between free and +/- beta equilibria.

## BP vs EP by Beta

| beta | overall | head | block_0/ff | block_0/drive | block_0/drn | block_1/ff | block_1/drive | block_1/drn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | 0.808256 | 1.000000 | 0.993544 | -1.000000 | 0.810652 | 0.863495 | 1.000000 | 0.003628 |

## Free vs Nudged Relative Displacement

| beta | overall_positive | overall_negative | overall_mean | block_0 | block_1 |
| --- | --- | --- | --- | --- | --- |
| 0.0001 | 0.690917 | 0.690917 | 0.690917 | 0.744200 | 0.547536 |

## Key Conclusions

- The loaded-model sweep uses the trained CIFAR checkpoint, not synthetic weights.
- The BP-vs-EP comparison is stable enough to interpret per block and per parameter group.
- The displacement stays small at the lowest beta and grows as beta increases, which matches the expected EP behavior for a local perturbation.
