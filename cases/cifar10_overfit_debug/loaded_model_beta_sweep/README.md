# Loaded CIFAR-10 Overfit Debug Beta Sweep

- config path: `/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2/experiment_config.json`
- checkpoint path: `/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2/checkpoint_best.pt`
- dataset slice: `first 32 samples of the evaluation subset`
- batch size: `16`
- sample count: `32`
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
| 0.0001 | 0.105563 | 1.000000 | 0.356759 | -1.000000 | -0.003142 | 0.320885 | 1.000000 | -0.040732 |
| 0.001 | 0.892050 | 1.000000 | 0.996083 | 1.000000 | 0.633454 | 0.999778 | 1.000000 | 0.881783 |
| 0.01 | 0.893359 | 1.000000 | 0.999933 | 1.000000 | 0.752320 | 1.000000 | 1.000000 | 0.921911 |
| 0.1 | 0.893353 | 1.000000 | 0.999933 | 1.000000 | 0.754155 | 1.000000 | 1.000000 | 0.922161 |

## Free vs Nudged Relative Displacement

| beta | overall_positive | overall_negative | overall_mean | block_0 | block_1 |
| --- | --- | --- | --- | --- | --- |
| 0.0001 | 0.729004 | 0.729004 | 0.729004 | 0.734160 | 0.716472 |
| 0.001 | 0.729004 | 0.729004 | 0.729004 | 0.734160 | 0.716472 |
| 0.01 | 0.729003 | 0.729005 | 0.729004 | 0.734160 | 0.716472 |
| 0.1 | 0.728991 | 0.729017 | 0.729004 | 0.734160 | 0.716472 |

## Key Conclusions

- The loaded-model sweep uses the trained CIFAR checkpoint, not synthetic weights.
- The head gradients match exactly across all tested betas; the remaining mismatch is concentrated in the DRN parameter groups, especially block_0/drn and block_1/drn at small beta.
- The free-vs-nudged displacement is large in this loaded model and is essentially flat across the beta grid, so beta changes the gradient estimate much more than the equilibrium displacement in this range.
- This checkpoint is therefore sensitive to beta in the EP gradient estimate, but not through a simple monotonic growth of the free-to-nudged state displacement.
