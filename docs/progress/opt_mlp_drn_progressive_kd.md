# OPT MLP DRN Progressive KD

This note records the KD-oriented training additions for the
`opt-progressive-kd` branch. The goal is to make OPT MLP-to-DRN replacement
less dependent on random joint recovery after local MLP MSE pretraining.

## Stronger Single-Block Pretraining

`opt_mlp_drn.single_block_train` now supports:

```text
--objective rigorous_pretrain
```

The loss can combine:

```text
MSE(DRN_l(z_l), r_l^T)
+ alpha_cosine * target_energy * (1 - cosine(DRN_l(z_l), r_l^T))
+ alpha_norm * target_energy * log(||DRN_l(z_l)|| / ||r_l^T||)^2
+ alpha_post_residual * MSE(a_l^T + DRN_l(z_l), h_{l+1}^T)
+ alpha_next_ln * MSE(next_LN(a_l^T + DRN_l(z_l)), next_LN(h_{l+1}^T))
+ alpha_logit_kl * KL(logits_T || logits_S)  # final decoder layer only
```

For the last OPT layer, the logit auxiliary uses the frozen final layer norm
and frozen LM head:

```text
logits_T = LMHead(final_LN(a_11^T + r_11^T))
logits_S = LMHead(final_LN(a_11^T + DRN_11(z_11^T)))
```

Example:

```bash
python -m opt_mlp_drn.single_block_train \
  --model_name facebook/opt-125m \
  --activation_cache runs/opt125m_activation_cache \
  --layers 9,10,11 \
  --objective rigorous_pretrain \
  --alpha_cosine 0.1 \
  --alpha_norm 0.1 \
  --alpha_post_residual 0.3 \
  --alpha_next_ln 0.3 \
  --alpha_logit_kl 0.1 \
  --logit_temperature 2.0 \
  --drn_drive_architecture signed_input_free \
  --drn_learn_amplification \
  --drn_amp_lr 1e-3 \
  --drn_iter 4 \
  --device cuda
```

## Progressive Full-Model KD

`opt_mlp_drn.joint_train` now supports:

```text
--distill_objective progressive_hidden_kl
--replacement_schedule 0.1,0.3,0.6,1.0
```

During training each replaced MLP layer samples:

```text
teacher MLP path with probability 1 - p
DRN MLP path     with probability p
```

Validation and test force `p = 1.0`, so reported metrics are always for the
fully replaced student.

The progressive objective can combine:

```text
kl_weight * shifted_next_token_KL
+ hidden_weight * selected_hidden_loss
+ post_residual_weight * post_residual_MSE
+ next_ln_weight * next_LN_MSE
+ delta_cosine_weight * delta_cosine_loss
+ delta_norm_weight * delta_norm_loss
+ ce_weight * shifted_CE
```

Hidden targets are configurable:

```text
--hidden_layers all
--hidden_layers replaced_outputs
--hidden_layers 10-12
```

For last-three replacement, `replaced_outputs` means hidden depths
`10,11,12`.

Example fixed-LM-head last-three run:

```bash
python -m opt_mlp_drn.joint_train \
  --model_name facebook/opt-125m \
  --train_data simulation_results/opt_mlp_drn_synthetic_splits_20260505/train_100k.txt \
  --val_data simulation_results/opt_mlp_drn_synthetic_splits_20260505/val_20k.txt \
  --test_data simulation_results/opt_mlp_drn_synthetic_splits_20260505/test_20k.txt \
  --replace_mlp_layers last:3 \
  --checkpoint_path 9=/path/to/layer_9/checkpoint_last.pt \
  --checkpoint_path 10=/path/to/layer_10/checkpoint_last.pt \
  --checkpoint_path 11=/path/to/layer_11/checkpoint_last.pt \
  --trainable_scope drn_attn_full_ln \
  --train_final_ln \
  --no-train_lm_head \
  --distill_objective progressive_hidden_kl \
  --replacement_schedule 0.1,0.3,0.6,1.0 \
  --hidden_layers replaced_outputs \
  --hidden_loss_type normed_mse \
  --hidden_weight 0.3 \
  --kl_weight 1.0 \
  --kl_temperature 2.0 \
  --post_residual_weight 0.3 \
  --next_ln_weight 0.3 \
  --delta_cosine_weight 0.1 \
  --delta_norm_weight 0.1 \
  --ce_weight 0.0 \
  --early_stopping_metric logit_kl \
  --patience 5 \
  --min_delta 1e-4 \
  --block_size 256 \
  --batch_size 4 \
  --steps 1000 \
  --eval_interval 100 \
  --eval_iters 32 \
  --drn_drive_architecture signed_input_free \
  --drn_learn_amplification \
  --drn_amp_lr 1e-4 \
  --drn_iter 4 \
  --device cuda
```

## Intended Comparison

Compare three runs on the same splits/checkpoints:

```text
pure KL, p = 1 from start
hidden+KL, p = 1 from start
progressive hidden+KL, p = 0.1 -> 1.0
```

The main target is to improve fixed-LM-head last-three replacement below the
current `~0.327` local-recovery test KL, with `0.20` as the first useful
threshold and `0.10` as a stronger recovery threshold.

