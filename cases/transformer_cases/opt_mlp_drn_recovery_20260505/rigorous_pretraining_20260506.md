# Rigorous Single-Block DRN Pretraining

Date: 2026-05-06

Branch: `opt-partial-fine-tuning`

This note records the first implementation and cached-data comparison for a
more rigorous OPT MLP-to-DRN pretraining objective.

## Implemented Changes

The single-block trainer now supports:

- `rigorous_pretrain` objective:
  - local MLP delta MSE
  - cosine-direction loss
  - output norm-ratio loss
  - optional post-residual MSE weight
  - next-layernorm MSE
  - optional final-readout logit KL for the final decoder layer
- `--objective_schedule`, for staged objectives such as:

```text
post_residual:750,rigorous_pretrain:250
```

- `checkpoint_best.pt`, selected by validation loss, in addition to
  `checkpoint_last.pt`.
- A package bootstrap in `opt_mlp_drn` so direct worktree runs import the
  sibling `digital_drn` source tree instead of an older external checkout.

The final-readout KL term is only valid for the final OPT decoder layer. It
compares:

```text
LMHead(final_LN(a_l + r_l^T))
```

against:

```text
LMHead(final_LN(a_l + DRN_l(z_l^T)))
```

with the final layer norm and LM head frozen.

## Cached Data

Runs used the existing normalized cached activation shards:

- layer 10: `simulation_results/opt_mlp_drn_cache_layer10_norm_16b_20260505`
- layer 11: `simulation_results/opt_mlp_drn_cache_layer11_norm_16b_20260505`

Common settings:

```text
steps = 1000
eval_interval = 250
drn_iter = 4
drive_architecture = signed_input_free
learn_amplification = true
lr = 3e-4
drn_amp_lr = 1e-3
weight_decay = 0
batch_size = 8
device = cuda
```

Rigorous objective weights:

```text
alpha_cosine = 0.3
alpha_norm = 0.1
alpha_next_ln = 0.3
alpha_post_residual = 0.0
alpha_logit_kl = 0.01 for layer 11, 0.0 for layer 10
logit_temperature = 2.0
```

## Results

For each current-code run, `best eval` is the best validation point found in
`metrics.jsonl`; `final` is `checkpoint_last.pt`.

| Run | Layer | Objective | Best step | Best relMSE | Best cosine | Best norm ratio | Best post-res relMSE | Final relMSE | Final cosine |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical post-residual | 10 | post_residual | 1000 | 0.2516 | 0.8652 | 0.8798 | 0.00175 | 0.2516 | 0.8652 |
| current post-residual | 10 | post_residual | 250 | 0.1552 | 0.9274 | 0.8040 | 0.00108 | 0.3243 | 0.8278 |
| current rigorous | 10 | rigorous_pretrain | 500 | 0.2453 | 0.8718 | 0.7983 | 0.00171 | 0.2887 | 0.8455 |
| historical post-residual | 11 | post_residual | 1000 | 0.0655 | 0.9923 | 0.7683 | 0.3808 | 0.0655 | 0.9923 |
| current post-residual | 11 | post_residual | 1000 | 0.9665 | 0.1961 | 0.1259 | 5.6359 | 0.9665 | 0.1961 |
| current rigorous | 11 | rigorous_pretrain | 1000 | 0.9999 | 0.9997 | 0.00003 | 5.8311 | 0.9999 | 0.9997 |
| current staged | 11 | post_residual -> rigorous | 750 | 0.9723 | 0.1750 | 0.1208 | 5.6700 | 0.9907 | 0.9986 |

## Interpretation

The first rigorous weighting did not improve the cached single-block fits.

Layer 10:

- The current post-residual baseline had the best validation point at step 250,
  not step 1000.
- This makes `checkpoint_best.pt` necessary; otherwise pretraining can keep an
  overtrained final checkpoint.
- The rigorous objective was worse than the current post-residual best point.

Layer 11:

- The current direct worktree import path does not reproduce the historical
  layer-11 post-residual result.
- The rigorous one-shot objective learned a very high cosine but nearly zero
  output norm, so it matched direction while failing magnitude.
- The staged run did not fix this because the post-residual warmup itself was
  already weak under the current code path.

This means the immediate bottleneck is not the final-readout KL term. The first
issue to solve is reliable reproduction of layer-11 local/post-residual
pretraining under the current `opt-partial-fine-tuning` source tree.

## Next Actions

1. Use `checkpoint_best.pt` for single-block initialization going forward.
2. Reproduce the historical layer-11 post-residual run under the current import
   path before trusting any final-logit auxiliary comparison.
3. For layer 10, prefer the current post-residual best checkpoint at step 250
   over the last checkpoint.
4. Re-test rigorous losses only after layer 11 can again reach high cosine and
   low relMSE under plain post-residual training.

