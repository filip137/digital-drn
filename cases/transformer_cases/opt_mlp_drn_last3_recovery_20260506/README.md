# OPT MLP DRN Last-Three Recovery Case Family

Date: 2026-05-06

Branch: `opt-partial-fine-tuning`

## Goal

This case family tracks one narrow target:

> Minimize held-out teacher-student divergence when OPT-125M MLP layers
> `9, 10, 11` are replaced by DRNs.

This should be the main decision table for the last-three replacement problem.
The broader historical notes remain in:

- [`../opt_mlp_drn_recovery_20260505/README.md`](../opt_mlp_drn_recovery_20260505/README.md)

## Fixed Problem Definition

Teacher:

- frozen `facebook/opt-125m`
- `eval()` mode
- no gradients

Student:

- same OPT architecture
- MLPs in layers `9, 10, 11` replaced by DRNs
- validation and test always force the fully replaced student:

```text
active layers = [9, 10, 11]
replacement probability = 1.0
```

Primary train/test data:

```text
train = simulation_results/opt_mlp_drn_synthetic_splits_20260505/train_100k.txt
val   = simulation_results/opt_mlp_drn_synthetic_splits_20260505/val_20k.txt
test  = simulation_results/opt_mlp_drn_synthetic_splits_20260505/test_20k.txt
```

Primary model constraint:

```text
embeddings frozen
LM head frozen
layers 0-8 frozen
```

unless the row is explicitly marked as an upper-bound or diagnostic ablation.

## Primary Metric

Use validation KL for model selection and test KL for the final comparison.

```text
primary selection metric = validation shifted next-token KL(teacher || student)
primary final metric     = test shifted next-token KL(teacher || student)
```

Secondary metrics:

```text
student test PPL
teacher test PPL
student PPL / teacher PPL
final hidden relative RMS
trainable parameter count
peak CUDA memory
```

A run only counts as a new best result if:

1. It improves best validation KL relative to the chosen baseline.
2. It also improves test KL, or the test result is statistically close and the
   run improves a clearly stated secondary metric.
3. The changed variable is isolated.

## Two Axes

All runs in this case family should be described by two independent choices.

### Axis A: DRN Pretraining

This answers:

> What DRN checkpoints are inserted before joint fine-tuning?

Tracked options:

| Pretrain ID | Description | Current status |
| --- | --- | --- |
| `random` | random DRN-native initialization | diagnostic lower bound |
| `phase5_postres_last` | historical normalized post-residual checkpoints, `checkpoint_last.pt` | current main baseline |
| `current_postres_best` | current-code post-residual pretraining, `checkpoint_best.pt` | partly useful; layer 11 reproduction issue |
| `rigorous_pretrain` | MSE + cosine + norm + next-LN/logit terms | implemented, not yet better |
| `output_gain_cosine` | high-cosine output-gain checkpoints | not a valid replacement checkpoint; norm ratio near zero |
| `output_gain_norm_penalty` | high cosine/norm-penalty checkpoints | promising for layers 9/10, poor for layer 11 |
| `calibrated_output_gain` | initialize output gain from teacher/DRN norm ratio | planned |
| `channel_output_gain` | per-channel output gain calibration | planned |

Pretraining pass criteria per layer:

```text
cosine >= 0.95
norm_ratio in [0.8, 1.25]
relMSE <= 0.10 preferred
post_residual_relMSE small
no NaNs
```

Layer `11` needs an extra readout-aware check because it directly feeds final
LN and the tied LM head:

```text
KL(
  LMHead(final_LN(a_11 + teacher_delta)),
  LMHead(final_LN(a_11 + DRN_delta))
)
```

### Axis B: Joint Fine-Tuning

This answers:

> After inserting the pretrained DRNs, how do we recover the full student?

Tracked options:

| Fine-tune ID | Description | Current status |
| --- | --- | --- |
| `pure_kl_no_curriculum` | shifted next-token KL only, fully replaced from step 0 | current local baseline |
| `hidden_kl_no_curriculum` | hidden-state MSE + shifted KL, fully replaced from step 0 | should be tested cleanly |
| `active_layer_curriculum` | `[11] -> [10,11] -> [9,10,11]`, stochastic DRN path | implemented; trains but not best yet |
| `forced_active_curriculum` | same layer schedule, but force at least one active DRN path per batch | planned |
| `deterministic_stage_curriculum` | stage-wise deterministic teacher-to-DRN transition | planned |
| `progressive_layer_training` | train layer 11, then 10+11, then 9+10+11 with carryover | planned/partly implemented |

Primary trainable scope:

```text
drn_attn_full_ln + train_final_ln
```

This means:

```text
train DRNs in layers 9-11
train attention q/k/v/out in layers 9-11
train layer norms in layers 9-11
train final decoder LN
freeze embeddings
freeze tied LM head
freeze layers 0-8
```

Upper-bound scope:

```text
all_student
```

This is useful for measuring recoverability but should not be mixed with the
fixed-backbone primary result.

## Anchor Results

| ID | Pretraining | Fine-tuning | Trainable scope | Val KL | Test KL | Student test PPL | Teacher test PPL | Final hidden RMS | Conclusion | Run path |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `L3-B0` | `phase5_postres_last` | `pure_kl_no_curriculum` | `drn_attn_full_ln + final LN` | 0.3559 | 0.3271 | 10.6973 | 7.6527 | 0.5257 | Main fixed-head local baseline. | `simulation_results/opt_mlp_drn_layer_count_kl_20260505/local_last3/...` |
| `L3-B1` | `phase5_postres_last` | `pure_kl_no_curriculum` | `all_student` | 0.2936 | 0.2327 | 9.6468 | 7.6527 | 0.4789 | Whole-student upper bound, not the primary fixed-backbone result. | `simulation_results/opt_mlp_drn_layer_count_kl_20260505/whole_last3/...` |
| `L3-C0` | `phase5_postres_last` | `active_layer_curriculum + hidden_kl` | `drn_attn_full_ln + final LN` | 0.3826 | 0.3424 | 11.0436 | 7.6527 | 0.5100 | Curriculum trains and improves from initialization, but does not beat `L3-B0`. | `simulation_results/opt_mlp_drn_active_curriculum_phase5_pretrained_20260506/opt_mlp_drn_joint_20260506-152557` |

## Current Best

Primary fixed-backbone best:

```text
L3-B0
test KL = 0.3271
```

Upper-bound best:

```text
L3-B1
test KL = 0.2327
```

The current active-layer curriculum result is useful evidence but not the
leader:

```text
L3-C0 test KL = 0.3424
```

## Open Questions

1. Can better DRN pretraining beat `phase5_postres_last` without changing the
   fine-tuning recipe?
2. Can curriculum beat no-curriculum when loss, checkpoints, and trainable scope
   are held fixed?
3. Is layer `11` the main bottleneck because final hidden errors are amplified
   by final LN + LM head?
4. Does output-gain calibration solve norm mismatch without hiding a weak DRN
   function approximation?
5. How close can the fixed-backbone scope get to the whole-student upper bound?

## Next Controlled Runs

Run only one changed variable at a time.

| Planned ID | Baseline | Change | Purpose |
| --- | --- | --- | --- |
| `L3-H0` | `L3-B0` | use `hidden_kl_no_curriculum` instead of pure KL | isolate whether hidden loss helps without curriculum |
| `L3-C1` | `L3-C0` | force at least one active DRN replacement per training batch | test whether stochastic zero-DRN batches weaken curriculum |
| `L3-P1` | `L3-B0` | replace `phase5_postres_last` with best calibrated-output-gain pretraining | isolate pretraining quality |
| `L3-T2` | `L3-B0` | train KL with temperature `2.0` | test Hinton-style softened KD |
| `L3-W0` | `L3-B1` | repeat whole-student run with same improved pretraining as primary candidate | update upper bound |

## Retired Or Diagnostic Runs

These runs should not be used as main evidence unless explicitly marked.

| Run family | Reason |
| --- | --- |
| `output_gain_cosine` | high cosine but near-zero norm ratio; relMSE stayed near `1.0` |
| first `rigorous_pretrain` runs | did not improve cached fits and exposed current-code layer-11 reproduction issue |
| untied/trainable LM-head runs | useful diagnostic, but changes output classifier geometry |
| whole-student runs | useful upper bound, but not the fixed-backbone primary target |

## Reporting Template

Every future run should add one row to the anchor table or to a follow-up table
with this sentence:

```text
This run changes <one variable> relative to <baseline ID>.
It changes validation KL from <old> to <new> and test KL from <old> to <new>.
Conclusion: <keep / reject / diagnostic only>.
```

If a run changes both pretraining and fine-tuning, split it into two runs unless
there is a specific reason not to.

