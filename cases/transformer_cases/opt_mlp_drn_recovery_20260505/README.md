# OPT MLP DRN Recovery and GPT-2 LST Context

Date: 2026-05-05

Branch: `partial_fine_tuning`

This note summarizes the transformer simulations run so far in this thread, with emphasis on the OPT-125M MLP-to-DRN replacement work. The purpose is to make the next experimental decision explicit rather than continuing to add training runs without a clear selection criterion.

Latest layer-count pure-KL sweep: [`layer_count_kl_sweep.md`](layer_count_kl_sweep.md).

## Correction: Single-Block Checkpoint Loading

After this note was first written, we checked whether the joint runs really loaded the learned single-block DRN conductances. They did not. The single-block checkpoints contained a separate `resistive_parameters` payload, but the joint checkpoint loader only restored the ordinary PyTorch module state. That means the earlier last-three joint runs loaded ordinary state such as DRN scales/amplifiers, but not the learned resistive/conductance tensors.

Code fix:

- `opt_mlp_drn/checkpoints.py` now loads `resistive_parameters` into the target `drn_mlp`.
- A regression test was added in `opt_mlp_drn/tests/test_opt_mlp_drn.py`.

Immediate eval-only comparison with the fixed-LM-head setup:

| Setup | Train steps | Val KL | Test KL | Val PPL | Test PPL | Comment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no single-block checkpoint | 0/effectively 0 | 0.6534 | 0.6780 | 18.87 | 14.54 | random DRN-native conductances |
| corrected single-block checkpoint load | 0/effectively 0 | 0.6746 | 0.6379 | 19.10 | 14.16 | learned conductances restored |

Interpretation:

- The old joint runs should not be described as true learned-conductance initialization runs.
- They are better described as random-conductance joint recovery runs with some loaded non-resistive DRN state.
- The corrected checkpoint initialization is not obviously better at step 0; it improves test KL relative to no-checkpoint eval but worsens validation KL.
- The right next comparison is a real training run with corrected checkpoint loading versus a fully no-checkpoint random-conductance training run.

## Executive Summary

The strongest result so far is still not teacher-equivalent. For OPT-125M last-three MLP replacement, the best logit-distillation runs are in the `0.34-0.41` KL range on the synthetic test split. A KL of `0.35` nats/token is roughly `0.50` bits/token, or about a `1.42x` perplexity penalty, so this is meaningful recovery but not close recovery. Because of the checkpoint-loading correction above, these runs should be interpreted as joint recovery without true learned-conductance initialization.

The cleanest fixed-backbone policy is:

- replace MLPs in OPT layers `9-11` with DRNs
- train DRNs plus attention q/k/v/out in layers `9-11`
- train layer norms in layers `9-11`
- train final decoder layer norm
- keep token embeddings and tied LM head fixed
- keep blocks `0-8` fixed

That clean fixed-LM-head run produced:

- best validation KL: `0.3877`
- test KL: `0.4049`
- test PPL: `11.94`
- teacher test PPL: `7.56`

The same local recovery with an untied trainable LM head did better:

- best validation KL: `0.3614`
- test KL: `0.3452`
- test PPL: `10.86`

But that changes the output classifier and is less clean if the goal is to preserve the pretrained embedding/LM-head geometry.

The most useful next move is probably not simply "more steps" on the fixed-head run. The fixed-head run early-stopped, and the hidden+KL runs suggest that stronger hidden-state recovery can help. The next high-value experiment is a fixed-LM-head objective that combines pure logit KL with a hidden-state loss on layers `10-12`, selected by validation KL.

## GPT-2 Ladder Side-Tuning Context

Before switching to OPT MLP replacement, we tested GPT-2 ladder-side variants. These results matter because they showed that digital LST is a stronger baseline than the DRN-LST variant so far.

### Upstream-Pruned Digital Runs

These runs used sparse upstream-style side layer counts with magnitude-pruned digital side initialization.

| Variant | Output mode | Side layers | Trainable params | Best val loss | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| upstream-pruned digital | residual_logits | 3 | 705,509 | 2.4915 | 2960 MB |
| upstream-pruned digital | residual_logits | 6 | 1,262,504 | 2.5203 | 2970 MB |
| upstream-pruned digital | residual_logits | 9 | 1,819,499 | 2.5462 | 2982 MB |
| upstream-pruned digital | side_only | 3 | 705,508 | 2.7287 | 2567 MB |
| upstream-pruned digital | side_only | 6 | 1,262,503 | 2.7601 | 2578 MB |
| upstream-pruned digital | side_only | 9 | 1,819,498 | 2.7990 | 2589 MB |

Interpretation:

- `residual_logits` clearly beat `side_only`.
- Increasing side-layer count did not help in these short runs.
- Best observed digital LST result here was the 3-layer upstream-pruned residual-logit run at `2.4915`.

### Digital Gated-Logit Reduction Sweep

Common setup:

- mode: `lst`
- side block type: digital transformer block
- output mode: `gated_logits`
- logit gate alpha init: `0.0`
- tap preset: `upstream_t5_base_3`
- initial side state: `full_tap`
- structural init: `magnitude_pruned`
- reduction factors: `4, 8, 16, 32`

| Ladder injection | r | d_s | Trainable params | Best val loss | Peak memory | Final rho | Mean abs lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-block mix | 4 | 192 | 2,073,797 | 2.5554 | 3375 MB | 0.5818 | n/a |
| pre-block mix | 8 | 96 | 705,509 | 2.5733 | 3352 MB | 0.4069 | n/a |
| pre-block mix | 16 | 48 | 270,197 | 2.6045 | 3344 MB | 0.2999 | n/a |
| pre-block mix | 32 | 24 | 114,749 | 2.6768 | 3341 MB | 0.2470 | n/a |
| post-block residual | 4 | 192 | 2,075,333 | 2.5680 | 3375 MB | 0.5720 | 1.0361 |
| post-block residual | 8 | 96 | 706,277 | 2.5624 | 3352 MB | 0.4128 | 1.0417 |
| post-block residual | 16 | 48 | 270,581 | 2.6196 | 3344 MB | 0.3060 | 1.0493 |
| post-block residual | 32 | 24 | 114,941 | 2.6780 | 3341 MB | 0.2522 | 1.0482 |

Interpretation:

- Pre-block mix and post-block residual are close for digital side blocks.
- The best gated-logit digital result was `2.5554` at `r=4`.
- The best signed-drive DRN hmult=8 result seen in the earlier sweep was `2.5713`, still behind the strongest digital LST result.
- These results motivated moving away from GPT-2 LST as the main DRN validation target and toward a more direct MLP replacement/distillation setup.

## OPT-125M DRN Replacement Setup

Teacher:

- frozen `facebook/opt-125m`
- evaluation mode
- no gradients into embeddings, transformer blocks, final LN, or tied LM head unless explicitly ablated

Student:

- same OPT architecture, with selected MLPs replaced by DRNs
- first focus: final MLP only, then layers `9-11`
- DRNs initialized from single-block distillation checkpoints for the last-three joint runs

Primary DRN architecture used for the later successful runs:

```text
digital input z:        768
signed input-free:      [z, -z] -> 1536
DRN hidden layer:       3072 * 2 = 6144
DRN output layer:       768
```

Important details:

- The signed drive is injected into the input free layer.
- There is no trainable digital `768 -> 3072` frontend in the signed-input-free architecture.
- The current injection coefficient / drive scale is learned.
- Amplification was made learnable in the stronger single-block runs.
- Default nonlinearity for the strong runs was `perfect_diode`.
- For hard-sigmoid ablations, `g_off` should be `1e-7`.

Synthetic split used for the later joint runs:

- train: `simulation_results/opt_mlp_drn_synthetic_splits_20260505/train_100k.txt`
- val: `simulation_results/opt_mlp_drn_synthetic_splits_20260505/val_20k.txt`
- test: `simulation_results/opt_mlp_drn_synthetic_splits_20260505/test_20k.txt`
- generated independently from the frozen teacher with different prompts/seeds
- train-only optimization, validation selection, final test once from the selected checkpoint

## Single-Block Distillation Results

The most useful single-block recipe was normalized post-residual distillation with signed-input-free DRNs, cached teacher states, learned amplification, and 4 solver iterations.

| Layer | Setup | Best step | relMSE | Cosine | Post-residual relMSE | Notes |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | raw input-free, learned amp | 750 | 0.0459 | 0.9905 | 0.0451 | good local fit |
| 0 | normalized post-residual input-free, learned amp | 750 | 0.0515 | 0.9816 | 0.0495 | good local fit |
| 5 | signed input-free earlier run | 1500 | 0.8185 | 0.4548 | 0.0010 | local output cosine poor, residual effect small |
| 9 | normalized post-residual input-free, learned amp | 1000 | 0.0570 | 0.9711 | 0.0003 | good final-three initializer |
| 10 | normalized post-residual input-free, learned amp | 1000 | 0.2516 | 0.8652 | 0.0018 | weakest of last-three local MLP fits |
| 11 | normalized post-residual input-free, learned amp | 1000 | 0.0655 | 0.9923 | 0.3808 | local delta fit good, final post-residual state remains sensitive |

Checkpoint sources:

- layer 9: `simulation_results/opt_mlp_drn_phase5_layer9_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/opt_mlp_drn_single_block_20260505-173625/layer_9/checkpoint_last.pt`
- layer 10: `simulation_results/opt_mlp_drn_phase5_layer10_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/opt_mlp_drn_single_block_20260505-173344/layer_10/checkpoint_last.pt`
- layer 11: `simulation_results/opt_mlp_drn_phase5_layer11_norm_postres_inputfree_cached_amp_lr1e3_base3e4_steps1000_20260505/opt_mlp_drn_single_block_20260505-173054/layer_11/checkpoint_last.pt`

Interpretation:

- Single-block local distillation can produce high cosine for some layers.
- Layer 10 remains a weak point in the last-three stack.
- Layer 11 has high MLP-delta cosine but large post-residual relative error, likely because the final hidden state/logits are sensitive to small structured errors at the final depth.
- Local MLP matching alone is not enough to recover model behavior after multiple replacements.

## Last-Three Joint Recovery: Hidden+KL Objective

These runs used independent synthetic train/val/test splits and initialized layers `9-11` from the single-block checkpoints above. They used the earlier hidden+KL style objective rather than pure analog-style logit KL.

Digital teacher baseline on this split:

- validation PPL: about `9.58`
- test PPL: about `7.55`

| Variant | Trainable scope | Trainable params | Best val KL | Test KL | Val PPL | Test PPL | Test final hidden rel RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean DRN replacement | DRNs only | 42,490,377 | 0.3549 | 0.2926 | 14.62 | 10.07 | 0.449 |
| attention adapter recovery | DRNs + attn out in 9-11 | 44,262,153 | 0.3522 | 0.3156 | 13.67 | 10.22 | 0.458 |
| full local recovery | DRNs + q/k/v/out + LNs in 9-11 | 49,586,697 | 0.3443 | 0.2869 | 14.31 | 10.20 | 0.446 |

Plot:

- `simulation_results/opt_mlp_drn_joint_last3_independent_splits_20260505/last3_trainable_scope_validation_ppl.png`

Interpretation:

- Training more than DRNs helped validation KL somewhat.
- The full local recovery scope had the best test KL in this group (`0.2869`), although PPL was still far from the teacher.
- These runs suggest a hidden-state auxiliary loss is useful and should be revisited in the fixed-LM-head setup.

## Pure Logit KL Distillation Runs

These runs were added to mimic the analog-foundation-model style more closely:

```text
loss = KL(teacher next-token distribution || student next-token distribution)
temperature = 1
beta = 1
no CE term
```

All runs used shifted causal logits: `logits[:, :-1]` predict `input_ids[:, 1:]`.

| Variant | Trainable scope | LM head policy | Trainable params | Best val KL | Test KL | Test PPL | Teacher test PPL | Test final hidden rel RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all-student | all student params | tied LM/embedding trainable | 167,729,673 | 0.3400 | 0.3582 | 11.01 | 7.56 | 0.517 |
| local + final LM, tied-head bug run | DRNs + q/k/v/out + local LNs + final LN + LM head | LM head tied to embeddings, so embeddings also changed | 88,197,129 | 0.3760 | 0.3401 | 10.81 | 7.56 | 0.529 |
| local + untied final LM | DRNs + q/k/v/out + local LNs + final LN + independent LM head | LM head copied/untied; embeddings frozen | 88,197,129 | 0.3614 | 0.3452 | 10.86 | 7.56 | 0.537 |
| local + fixed tied LM head | DRNs + q/k/v/out + local LNs + final LN | tied LM/embedding fixed | 49,588,233 | 0.3877 | 0.4049 | 11.94 | 7.56 | 0.564 |

Run sources:

- all-student: `simulation_results/opt_mlp_drn_logit_distill_independent_splits_20260505/all_student_lm_head_beta1/opt_mlp_drn_joint_20260505-215301`
- local tied-head bug run: `simulation_results/opt_mlp_drn_logit_distill_independent_splits_20260505/local_attn_ln_final_lm_beta1/opt_mlp_drn_joint_20260505-220309`
- local untied LM head: `simulation_results/opt_mlp_drn_logit_distill_independent_splits_20260505/local_attn_ln_final_lm_frozen_embed_beta1/opt_mlp_drn_joint_20260505-220819`
- local fixed LM head: `simulation_results/opt_mlp_drn_logit_distill_independent_splits_20260505/local_attn_ln_final_ln_fixed_lm_beta1/opt_mlp_drn_joint_20260505-221906`

Interpretation:

- If the LM head is trainable, OPT weight tying matters. Training `lm_head.weight` also trains token embeddings unless the head is explicitly untied.
- The untied-LM-head run is a useful ablation, but it changes the output classifier.
- The fixed-LM-head run is the cleanest analog of replacing internal modules while preserving the pretrained token geometry.
- Fixed LM head is significantly harder: test KL worsened from `0.3452` to `0.4049`.
- The fixed-LM-head run early-stopped, so simply increasing steps is not the first thing to try.

## What Worked

- Independent train/val/test synthetic splits made the results more interpretable than a single contiguous synthetic file.
- Single-block cached distillation is much faster and clearer than running the full transformer for every block-local experiment.
- Signed-input-free DRNs with learned amplification worked much better than the earlier naive current frontend attempts.
- Freezing embeddings and earlier blocks can be verified directly: hidden drift through layers `1-9` was exactly zero in the fixed-prefix runs.
- Untying the LM head gives a clean way to train the output classifier without accidentally training token embeddings.

## What Did Not Work Yet

- GPT-2 DRN-LST did not beat digital LST baselines.
- Side-only LST was much worse than residual-logit LST in the GPT-2 experiments.
- Pure local MLP MSE is not sufficient for full-model recovery after multiple MLP replacements.
- Training the fixed-LM-head last-three student with pure KL alone did not recover teacher perplexity.
- The fixed-LM-head run got worse on test (`KL 0.4049`) than the untied-LM-head run (`KL 0.3452`), which means part of the apparent recovery can come from output classifier adaptation rather than internal DRN recovery.

## Recommended Next Experiments

### 1. Fixed LM Head With Hidden+KL Loss

Run the clean fixed-LM-head trainable scope again, but use a mixed objective:

```text
loss = lambda_h * hidden_loss(layers 10, 11, 12) + lambda_kl * shifted_logit_KL
```

Initial grid:

| lambda_h | lambda_kl | selection metric |
| ---: | ---: | --- |
| 0.1 | 1.0 | validation KL |
| 0.3 | 1.0 | validation KL |
| 1.0 | 1.0 | validation KL |

Rationale:

- The earlier hidden+KL full local recovery had better test KL (`0.2869`) than pure KL fixed-head (`0.4049`).
- Pure KL alone may let hidden states drift in ways that are hard to correct with a fixed LM head.

### 2. Improve Layer 10 Single-Block Distillation

Layer 10 is the weakest last-three local MLP fit:

- relMSE: `0.2516`
- cosine: `0.8652`

Try:

- more cached data
- longer single-block training
- smaller drive scale
- more solver iterations
- hidden multiplier/capacity ablation
- layer-specific amplification learning rate

Rationale:

- Joint recovery may be bottlenecked by the weakest local DRN.

### 3. Revisit Layer 11 Post-Residual Matching

Layer 11 has high local MLP cosine but poor post-residual relative error:

- cosine: `0.9923`
- post-residual relMSE: `0.3808`

Try training layer 11 with a final-LN/logit-aware target rather than only normalized local delta MSE.

Rationale:

- The last layer directly controls the final hidden state consumed by the fixed LM head.

### 4. Scale Data Only After Objective Is Stable

Do not jump straight to `50M+` tokens yet. First use the current independent splits to settle the objective. Then scale:

- train: `1M` tokens
- validation: `100k` tokens
- test: `100k` tokens

Rationale:

- The current fixed-head run early-stopped, so data scale is not clearly the current bottleneck.
- Larger data should come after the objective and trainable scope are chosen.

### 5. Keep Reporting Both KL and PPL

For decision-making, always report:

- validation KL
- test KL
- teacher PPL
- student PPL
- final hidden rel RMS
- frozen-prefix hidden drift
- trainable parameter count
- LM-head/embedding policy

Rationale:

- KL alone tells us teacher-distribution matching.
- PPL tells us how the student behaves on the generated sequences.
- Hidden drift tells us whether internal recovery is failing before the logits.

## Current Decision

Use the fixed tied LM head as the primary clean experiment, even though it is currently worse. Keep the untied-LM-head run as a diagnostic ablation, not the main result.

The next concrete run should be:

```text
replace_mlp_layers = 9-11
trainable_scope = drn_attn_full_ln
train_final_ln = true
train_lm_head = false
distill objective = hidden + shifted logit KL
select checkpoint by validation KL
```

Success threshold for calling the last-three fixed-head replacement "working":

- test KL below `0.20` nats/token as an intermediate target
- test KL below `0.10` nats/token for a convincing recovery claim
- no hidden drift in frozen prefix
- no NaNs in DRN states
- no gradients on embeddings, LM head, or blocks `0-8`
