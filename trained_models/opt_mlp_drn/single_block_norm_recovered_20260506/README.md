# OPT MLP DRN Single-Block Norm-Recovered Checkpoints

Date: 2026-05-06

This directory is a curated local registry for reusable OPT-125M single-block DRN MLP replacement checkpoints. The selected checkpoints replace MLP layers `9`, `10`, and `11` and were trained on cached teacher activations with explicit cosine and norm-ratio penalties.

The large `checkpoint_best.pt` files are intentionally ignored by git. Keep them locally or move them to an external artifact store; commit only this README, `.gitignore`, and `manifest.json`.

## Files

```text
layer_9/checkpoint_best.pt
layer_10/checkpoint_best.pt
layer_11/checkpoint_best.pt
manifest.json
```

## Metrics

| Layer | Best step | Cosine | Norm ratio | relMSE | output gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 750 | 0.994966 | 0.997970 | 0.010051 | 57.462 |
| 10 | 750 | 0.995365 | 0.998330 | 0.009257 | 51.226 |
| 11 | 1000 | 0.999743 | 0.991902 | 0.000575 | 535.822 |

## Training Settings

```text
model_name = facebook/opt-125m
objective = local_mlp_cosine
mlp_input_mode = normalized
drn_drive_architecture = signed_input_free
drn_iter = 4
drn_learn_amplification = true
drn_amp_lr = 0.01
output_gain_lr = 1.0
lr = 0.0003
weight_decay = 0.0
grad_clip = 1.0
```

Layer-specific objective weights:

```text
layers 9/10: alpha_cosine = 100,   alpha_norm = 100
layer 11:    alpha_cosine = 10000, alpha_norm = 100
```

See `manifest.json` for source run paths, activation caches, checksums, and full metadata.
