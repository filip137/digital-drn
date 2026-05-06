# OPT MLP DRN Output-Gain Cosine Check, Layers 9-11

Date: 2026-05-06

## Code change

Added a learnable scalar output gain after each DRN MLP replacement:

```text
student_delta = output_gain * output_scale * DRN(z / input_scale)
```

The gain is present in both the standalone single-block pretraining module and the full OPT wrapper layer. Single-block checkpoints now save and reload this scalar, and the full-wrapper checkpoint loader copies it into the corresponding replacement layer.

Optimizer behavior:

- standalone single-block training puts `output_gain` in its own optimizer group
- when `--drn_amp_lr` is provided, `output_gain` uses that amplifier learning rate
- joint OPT training uses `--drn_amp_lr` for replaced-layer output gains when provided

## Test setup

Shared settings:

```text
model_name = facebook/opt-125m
objective = local_mlp_cosine
mlp_input_mode = normalized
eval_logit_kl_batches = 0
steps = 1000
eval_interval = 250
eval_iters = 20
batch_size = 8
drn_iter = 4
drn_drive_architecture = signed_input_free
drn_learn_amplification = true
drn_amp_lr = 0.001
lr = 0.0003
weight_decay = 0.0
grad_clip = 1.0
alpha_next_ln = 1.0
device = cuda
```

Activation caches:

```text
layer 9:  simulation_results/opt_mlp_drn_cache_layer9_norm_16b_20260505
layer 10: simulation_results/opt_mlp_drn_cache_layer10_norm_16b_20260505
layer 11: simulation_results/opt_mlp_drn_cache_layer11_norm_16b_20260505
```

Result root:

```text
simulation_results/opt_mlp_drn_output_gain_cosine_20260506/
```

## Results

| Layer | Cosine | relMSE | norm ratio | output gain | Best step | Peak CUDA memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 0.997296 | 0.999313 | 0.0003445 | 1.25945 | 1000 | 1414 MB |
| 10 | 0.997784 | 0.999192 | 0.0004051 | 1.28038 | 1000 | 1414 MB |
| 11 | 0.999715 | 0.999940 | 0.0000300 | 1.48881 | 1000 | 1414 MB |

Best checkpoints:

```text
layer 9:  simulation_results/opt_mlp_drn_output_gain_cosine_20260506/layer9_amp_lr/opt_mlp_drn_single_block_20260506-115147/layer_9/checkpoint_best.pt
layer 10: simulation_results/opt_mlp_drn_output_gain_cosine_20260506/layer10_amp_lr/opt_mlp_drn_single_block_20260506-115238/layer_10/checkpoint_best.pt
layer 11: simulation_results/opt_mlp_drn_output_gain_cosine_20260506/layer11_amp_lr/opt_mlp_drn_single_block_20260506-115331/layer_11/checkpoint_best.pt
```

## Interpretation

The cosine threshold is cleared for layers 9, 10, and 11. These runs therefore satisfy the direction-only criterion of cosine greater than 0.95.

This is not yet a useful MLP replacement checkpoint. The DRN output norm remains much too small: the predicted MLP residual norm is only about `3e-5` to `4e-4` of the teacher residual norm, and relMSE stays near `1.0`. In other words, the model has learned a high-cosine but near-zero-amplitude direction. The output gain helps only modestly over 1000 steps because the required scale correction is orders of magnitude larger than the learned scalar reached here.

Next useful tests should make the magnitude requirement explicit, for example:

- initialize or calibrate `output_gain` from the initial teacher/DRN norm ratio
- optimize a log-space output gain so large scale corrections are reachable
- use a magnitude-aware objective such as `rigorous_pretrain` with a stronger norm term
- report cosine and norm ratio together as the pass/fail criterion

## Verification

```text
/home/filip/miniconda3/envs/py312/bin/python -m pytest opt_mlp_drn/tests/test_single_block.py opt_mlp_drn/tests/test_opt_mlp_drn.py opt_mlp_drn/tests/test_phase_trainers.py -q
22 passed

/home/filip/miniconda3/envs/py312/bin/python -m py_compile opt_mlp_drn/single_block.py opt_mlp_drn/model.py opt_mlp_drn/checkpoints.py opt_mlp_drn/single_block_train.py opt_mlp_drn/progressive_train.py opt_mlp_drn/joint_train.py
passed
```
