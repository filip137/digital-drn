# Loaded CIFAR-10 Overfit Debug Beta Sweep

- config path: `/home/filiposana/digital_drn/cases/conv_cases/cifar10_wider_loaded_ep_bp_va4_ca1_akib/config_va4_ca1.json`
- checkpoint path: `/home/filiposana/digital_drn/simulation_results/cifar10_drn_only_signed_norm_readout/20260408-170358-integnano-akib/checkpoint_best.pt`
- dataset slice: `first 32 samples of the evaluation subset`
- batch size: `16`
- sample count: `32`
- mode: `asynchronous`
- block iterations: `[6, 6]`
- non_linearity: `perfect_diode`
- voltage_amp: `4.0`
- current_amp: `1.0`

BP-vs-EP compares ordinary backprop on the loaded network against `hybrid_backward_explicit` on the same batch.
The displacement metric pools the DRN free-layer states across blocks and measures pooled relative RMS displacement between free and +/- beta equilibria.

## BP vs EP by Beta

| beta | overall | head | block_0/ff | block_0/drive | block_0/drn | block_1/ff | block_1/drive | block_1/drn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.240380 | 1.000261 | 0.932634 | 1.000000 | 0.360916 | 0.851132 | 1.000000 | 0.413093 |

## Free vs Nudged Relative Displacement

| beta | overall_positive | overall_negative | overall_mean | block_0 | block_1 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 0.019781 | 0.019781 | 0.019781 | 0.000080 | 0.021440 |

## Key Conclusions

- The loaded-model sweep uses the trained CIFAR checkpoint, not synthetic weights.
- The head gradients match exactly across all tested betas; the remaining mismatch is concentrated in the DRN parameter groups, especially block_0/drn and block_1/drn at small beta.
- The free-vs-nudged displacement is large in this loaded model and is essentially flat across the beta grid, so beta changes the gradient estimate much more than the equilibrium displacement in this range.
- This checkpoint is therefore sensitive to beta in the EP gradient estimate, but not through a simple monotonic growth of the free-to-nudged state displacement.
