# OPT MLP DRN

Experiments for replacing selected OPT decoder MLP residual updates with
tokenwise DRNs. This package is intentionally separate from `gpt2_ladder_drn`:
it is not ladder side-tuning, and the base model is Hugging Face OPT.

Default first run:

```bash
python -m opt_mlp_drn.train \
  --model_name facebook/opt-125m \
  --data data/tiny_shakespeare.txt \
  --replace_mlp_layers last:1 \
  --objective distill_then_ce \
  --distill_steps 2000 \
  --ce_steps 2000 \
  --block_size 256 \
  --batch_size 8 \
  --device cuda \
  --drn_iter 4
```

The stage-one distillation loss matches the frozen OPT teacher state after the
selected MLP residual update and final layer norm:

```text
u = h + Attention(LN_1(h))
teacher_delta = fc2(ReLU(fc1(LN_2(u))))
student_delta = DRN(LN_2(u))
loss = mean((final_LN(u + student_delta) - stopgrad(final_LN(u + teacher_delta)))^2)
```

Only DRN parameters are trainable. All OPT embeddings, attention modules, MLP
teacher modules, layer norms, and LM head remain frozen.

## Single-Block MLP Distillation

The block-local experiments train one standalone DRN against frozen teacher
activations. For OPT-125M the default probe layers are `0,6,11`.

For layer `l`, the teacher-forced tensors are:

```text
z_l      = LN_2(a_l)
r_l      = MLP_l^T(z_l)
h_{l+1}  = a_l + r_l
```

The implemented losses are:

```text
local_mlp:     mean((DRN_l(z_l) - r_l)^2)
post_residual: mean((a_l + DRN_l(z_l) - h_{l+1})^2)
next_ln_aux:   post_residual + alpha * mean((LN_1,l+1(h_{l+1}^S) - LN_1,l+1(h_{l+1}^T))^2)
```

Calibration collects scalar `mean`, `std`, and approximate `q0.999(abs(.))`
statistics for `z_l` and `r_l`, then uses them as DRN input/output scales:

```bash
python -m opt_mlp_drn.calibration \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers default \
  --max_batches 512 \
  --output runs/opt125m_calibration.json \
  --device cuda
```

Run local MLP MSE on the representative layers:

```bash
python -m opt_mlp_drn.single_block_train \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers 0,6,11 \
  --objective local_mlp \
  --calibration runs/opt125m_calibration.json \
  --steps 2000 \
  --eval_logit_kl_batches 8 \
  --drn_iter 4 \
  --device cuda
```

For repeated layerwise experiments, first materialize a teacher activation
cache. This runs the frozen teacher once and stores sharded tensors for every
selected layer:

```text
z_l       = MLP input
r_l       = teacher MLP output
a_l       = pre-MLP residual state
h_{l+1}   = a_l + r_l
next_ln_l = LN_1,l+1(h_{l+1}) or decoder final LN for the last layer
```

```bash
python -m opt_mlp_drn.cache_activations \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers all \
  --mlp_input_mode normalized \
  --block_size 256 \
  --batch_size 8 \
  --dtype float16 \
  --output_dir runs/opt125m_activation_cache \
  --device cuda
```

Then train a layer from the cached tensors without running the full transformer
inside the training loop:

```bash
python -m opt_mlp_drn.single_block_train \
  --model_name facebook/opt-125m \
  --activation_cache runs/opt125m_activation_cache \
  --layers 0 \
  --objective local_mlp \
  --mlp_input_mode normalized \
  --eval_logit_kl_batches 0 \
  --steps 2000 \
  --device cuda
```

Use `--mlp_input_mode raw` for the no-MLP-LayerNorm ablation, where `z_l = a_l`
and the target is `MLP_l^T(a_l)`.

For stronger KD-style pretraining, use `--objective rigorous_pretrain`. This
adds optional delta cosine, delta norm-ratio, post-residual, next-LN, and
final-layer logit-KL terms. See
[`docs/progress/opt_mlp_drn_progressive_kd.md`](../docs/progress/opt_mlp_drn_progressive_kd.md)
for the exact loss and recommended flags.

Teacher-front-end initialization copies OPT `fc1` into the DRN current
frontend. It currently requires `--no-drn_signed_drive`; full conductance-level
`fc2`/sign-split initialization is a later ablation.

```bash
python -m opt_mlp_drn.single_block_train \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers 0,6,11 \
  --objective local_mlp \
  --init_mode teacher_frontend_scale \
  --no-drn_signed_drive \
  --calibration runs/opt125m_calibration.json \
  --steps 2000 \
  --device cuda
```

Corpus helpers cover the three intended data sources:

```bash
python -m opt_mlp_drn.corpus --mode real_subset --real_path raw.txt --output real_1m.txt --max_tokens 1000000
python -m opt_mlp_drn.corpus --mode generate_synthetic --output synthetic_1m.txt --max_tokens 1000000 --device cuda
python -m opt_mlp_drn.corpus --mode mix --real_path real_1m.txt --synthetic_path synthetic_1m.txt --output mixed_1m.txt
```

## Phase 5-8 Workflows

All-layer independent distillation runs the same teacher-forced block-local
objective for every OPT layer and writes per-layer checkpoints plus a summary
table:

```bash
python -m opt_mlp_drn.single_block_train \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers all \
  --objective local_mlp \
  --steps 2000 \
  --device cuda
```

Progressive replacement trains layer `k` while layers `0..k-1` are already
replaced by previously saved DRNs:

```bash
python -m opt_mlp_drn.progressive_train \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --layers all \
  --checkpoint_dir runs/opt_mlp_drn_single_block_* \
  --steps_per_layer 2000 \
  --device cuda
```

For full-model KD after local pretraining, `joint_train` supports progressive
teacher/DRN module replacement:

```bash
python -m opt_mlp_drn.joint_train \
  --model_name facebook/opt-125m \
  --replace_mlp_layers last:3 \
  --trainable_scope drn_attn_full_ln \
  --train_final_ln \
  --distill_objective progressive_hidden_kl \
  --replacement_schedule 0.1,0.3,0.6,1.0 \
  --hidden_layers replaced_outputs \
  --hidden_loss_type normed_mse \
  --kl_temperature 2.0 \
  --post_residual_weight 0.3 \
  --next_ln_weight 0.3 \
  --delta_cosine_weight 0.1 \
  --delta_norm_weight 0.1 \
  --device cuda
```

Full-transformer joint distillation replaces all selected MLPs and optimizes
hidden-state MSE plus teacher-student logit KL, with optional CE:

```bash
python -m opt_mlp_drn.joint_train \
  --model_name facebook/opt-125m \
  --data data/calibration.txt \
  --replace_mlp_layers all \
  --checkpoint_dir runs/opt_mlp_drn_progressive_* \
  --hidden_weight 1.0 \
  --kl_weight 0.1 \
  --kl_temperature 1.0 \
  --steps 2000 \
  --device cuda
```

The experiment matrix CLI records the planned ablations for dataset source,
token budget, loss, initialization, solver iterations, DRN capacity, and
trainable-parameter scope:

```bash
python -m opt_mlp_drn.experiment_matrix --list --write_plan runs/opt_mlp_drn_matrix.jsonl
```
