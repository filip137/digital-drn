# Noise-Robust Single-Block DRN Pretraining, 1M Synthetic Split

Date: 2026-05-06

## Implementation

Added support for the planned noise/drift robust single-block experiments:

- `opt_mlp_drn.cache_activations` now accepts explicit `--train_data`, `--val_data`, and optional `--test_data` split files.
- The cache CLI also supports `--sequence_stride` so long token files can be cached as roughly non-overlapping contexts instead of every sliding window.
- `opt_mlp_drn.single_block_train` now supports `--input_noise_std`, `--residual_drift_std`, `--input_noise_mode gaussian`, `--noise_train_only/--no-noise_train_only`, and `--init_checkpoint`.
- Added training-time augmentation:
  - `z_aug = z + input_noise_std * std(z) * eps`
  - `a_aug = a + residual_drift_std * std(a) * eps`
- Added objective `drift_compensated_residual_cosine`:
  - target delta is `h_next - a_aug`
  - loss is post-residual MSE plus cosine and norm-ratio penalties against that compensated target.
- Added standalone single-block checkpoint loading that restores module state, resistive parameters, `input_scale`, `output_scale`, and `output_gain`.

Tests run:

```text
python -m pytest opt_mlp_drn/tests/test_single_block.py opt_mlp_drn/tests/test_opt_mlp_drn.py opt_mlp_drn/tests/test_phase_trainers.py -q
31 passed

python -m py_compile opt_mlp_drn/single_block.py opt_mlp_drn/single_block_train.py opt_mlp_drn/cache_activations.py opt_mlp_drn/checkpoints.py
passed
```

## Data And Cache

Generated independent teacher-synthetic splits:

| Split | Requested tokens | File |
| --- | ---: | --- |
| train | 1,000,000 | `simulation_results/opt_mlp_drn_synthetic_splits_1m_20260506/train_1m.txt` |
| val | 100,000 | `simulation_results/opt_mlp_drn_synthetic_splits_1m_20260506/val_100k.txt` |
| test | 100,000 | `simulation_results/opt_mlp_drn_synthetic_splits_1m_20260506/test_100k.txt` |

Cached layers `9,10,11` from `facebook/opt-125m` with:

```text
block_size = 256
sequence_stride = 256
batch_size = 16
dtype = float16
mlp_input_mode = normalized
```

Cache location:

```text
simulation_results/opt_mlp_drn_cache_1m_layers9_11_norm_20260506/
```

Cache consistency check passed. For cached `z`, the frozen digital OPT MLP recomputes cached `r` with cosine essentially `1.0` for layers `9`, `10`, and `11`, so the low DRN scores below are not caused by a bad cache.

## Planned Signed-Input-Free Run

The requested shared settings were used first:

```text
drn_drive_architecture = signed_input_free
drn_iter = 4
drn_learn_amplification = true
lr = 0.0003
drn_amp_lr = 0.01
output_gain_lr = 1.0
alpha_cosine = 100 for layer 9
alpha_norm = 100
init_checkpoint = trained_models/opt_mlp_drn/single_block_norm_recovered_20260506/layer_9/checkpoint_best.pt
```

Raw output:

```text
simulation_results/opt_mlp_drn_noise_robust_single_block_1m_20260506/clean_layer9/opt_mlp_drn_single_block_20260506-162212/
```

Result:

| Architecture | Layer | Steps | Cosine | Norm ratio | Target relMSE | Best step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| signed_input_free | 9 | 1000 | 0.5683 | 1.0235 | 0.8852 | 1000 | Failed acceptance |

A same-hyperparameter continuation from this best checkpoint reached only cosine `0.6561` after another 1000 steps. A sharper probe with `alpha_cosine=1000`, `lr=0.001`, `drn_amp_lr=0.001`, and `output_gain_lr=0.1` was worse, reaching only `0.5316` by step 750.

Interpretation: for the broader 1M-token synthetic activation distribution, the `signed_input_free` single-block DRN is not recovering layer 9 under the requested settings. The old layer-9 checkpoint still scores `0.995` cosine on its original small cache, so the failure is a distribution/generalization/optimization issue rather than checkpoint corruption.

## Projected-Hidden Diagnostic

To isolate whether the problem was the cache/objective or the signed-input-free architecture, one diagnostic used the older projected-hidden current frontend initialized from teacher `fc1`:

```text
drn_drive_architecture = projected_hidden
init_mode = teacher_frontend_scale
drn_iter = 4
lr = 0.0003
drn_amp_lr = 0.01
output_gain_lr = 1.0
alpha_cosine = 100
alpha_norm = 100
```

Raw output:

```text
simulation_results/opt_mlp_drn_noise_robust_single_block_1m_20260506/diagnostic_layer9_projected_teacher_frontend/opt_mlp_drn_single_block_20260506-163752/
```

Result:

| Architecture | Layer | Steps | Cosine | Norm ratio | Target relMSE | Best step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| projected_hidden + teacher_frontend | 9 | 1000 | 0.9916 | 0.9987 | 0.0168 | 1000 | Passed acceptance |

This confirms the objective and cache are usable. The current blocker is specifically the `signed_input_free` architecture/initialization on broader activations.

## Decision

Do not curate the interrupted `signed_input_free` layer-9 run into `trained_models/opt_mlp_drn/single_block_noise_robust_1m_20260506/` as a successful checkpoint: it does not meet cosine `> 0.95`.

Recommended next step: either

1. treat `projected_hidden + teacher_frontend_scale` as the near-term recoverable architecture for noise/drift robustness, then run the full clean/noise/drift matrix for layers `9,10,11`; or
2. add a real teacher-weight/sign-split initialization for `signed_input_free` before rerunning the requested matrix.
