# OPT MLP DRN Output-Gain Cosine Check, Layers 9-11

Date: 2026-05-06

## Brief summary

Implemented a learnable scalar output gain after each OPT MLP-replacement DRN, wired it into standalone pretraining and full OPT wrapper layers, preserved it in checkpoints, and trained layers 9, 10, and 11 on cached activations. All three layers passed the cosine target above `0.95`, but the learned outputs still have near-zero magnitude, so these checkpoints should be treated as direction-only fits rather than successful MLP replacements.

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


## Follow-up: explicit norm penalty and higher gain LR

After the first output-gain runs, cosine was high but output magnitude was nearly zero. The training code was updated so the `local_mlp_cosine` objective uses separate weights:

```text
loss = local_mse
     + alpha_cosine * E[||r_teacher||^2] * cosine_loss
     + alpha_norm   * E[||r_teacher||^2] * norm_ratio_loss
```

A separate optimizer knob was also added:

```text
--output_gain_lr
```

This lets the post-DRN output gain use a much larger learning rate than the main DRN conductance/frontend parameters. In these runs, amplification also used a larger learning rate:

```text
--drn_amp_lr 0.01
--output_gain_lr 1.0
```

The useful settings were:

```text
layers 9/10: alpha_cosine = 100,   alpha_norm = 100
layer 11:    alpha_cosine = 10000, alpha_norm = 100
```

| Layer | Best step | Cosine | Norm ratio | relMSE | output gain | current amp | voltage amp | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 750 | 0.994966 | 0.997970 | 0.010051 | 57.462 | 0.347 | 1.691 | 0.002391 |
| 10 | 750 | 0.995365 | 0.998330 | 0.009257 | 51.226 | 0.388 | 1.647 | 0.002827 |
| 11 | 1000 | 0.999743 | 0.991902 | 0.000575 | 535.822 | 0.250 | 3.718 | 1.886319 |

Best checkpoints:

```text
layer 9:  simulation_results/opt_mlp_drn_output_gain_norm_penalty_20260506/layer9_cos100_norm100/opt_mlp_drn_single_block_20260506-151704/layer_9/checkpoint_best.pt
layer 10: simulation_results/opt_mlp_drn_output_gain_norm_penalty_20260506/layer10_cos100_norm100/opt_mlp_drn_single_block_20260506-151801/layer_10/checkpoint_best.pt
layer 11: simulation_results/opt_mlp_drn_output_gain_norm_penalty_20260506/layer11_cos10000_norm100/opt_mlp_drn_single_block_20260506-151949/layer_11/checkpoint_best.pt
```

Interpretation: the previous failure was not that the DRN could not represent the teacher MLP direction. It was an optimization/objective issue: without a strong norm term and a fast output-gain parameter, cosine training could settle on a tiny-amplitude direction. With explicit norm pressure and high output-gain LR, all three layers recover both direction and magnitude on the cached validation batches.

## Verification

```text
/home/filip/miniconda3/envs/py312/bin/python -m pytest opt_mlp_drn/tests/test_single_block.py opt_mlp_drn/tests/test_opt_mlp_drn.py opt_mlp_drn/tests/test_phase_trainers.py -q
22 passed

/home/filip/miniconda3/envs/py312/bin/python -m py_compile opt_mlp_drn/single_block.py opt_mlp_drn/model.py opt_mlp_drn/checkpoints.py opt_mlp_drn/single_block_train.py opt_mlp_drn/progressive_train.py opt_mlp_drn/joint_train.py
passed
```
