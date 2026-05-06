# Partial Fine-Tuning Reading List

This is the curated reading subset for the `opt-partial-fine-tuning` branch.
Use it when working on OPT MLP replacement, DRN student recovery, and
teacher-student distillation. The full literature packet contains broader DRN
and transformer background; this file points only to the papers that should
usually matter for this branch.

## Core PEFT

- [LoRA](core_finetuning/lora.md)
- [Houlsby adapters](core_finetuning/houlsby_adapters.md)
- [BitFit](core_finetuning/bitfit.md)
- [FISH Mask](core_finetuning/fish_mask_fixed_sparse_masks.md)
- [Ladder Side-Tuning](core_finetuning/ladder_side_tuning.md)

Use these papers to frame parameter-efficient recovery policies:

- DRN-only recovery.
- DRNs plus selected attention projections.
- DRNs plus local layer norms.
- Full-student fine-tuning as the high-capacity upper bound.

## Knowledge Distillation And Module Replacement

- [Hinton knowledge distillation](core_knowledge_distillation/hinton_distilling_knowledge.md)
- [DistilBERT](core_knowledge_distillation/distilbert.md)
- [TinyBERT](core_knowledge_distillation/tinybert.md)
- [BERT-of-Theseus](core_knowledge_distillation/bert_of_theseus_progressive_module_replacement.md)

Use these papers for the branch's main training logic:

- Teacher logits are produced by the frozen digital OPT model.
- Student logits are produced by the OPT model with selected MLPs replaced by
  DRNs.
- The primary full-model objective is shifted next-token
  `KL(teacher || student)`.
- Single-block pretraining uses teacher-forced MLP inputs and matches either
  MLP deltas or post-residual hidden states.
- Progressive replacement is the natural response to hidden-state distribution
  shift after earlier MLPs are replaced.

## Analog-Aware Transformer Baselines

- [AHWA-LoRA](core_analog_transformers/ahwa_lora_analog_transformer_adaptation.md)
- [Analog Foundation Models](core_analog_transformers/buchel_analog_foundation_models.md)
- [3D analog in-memory LLMs / MoE](core_analog_transformers/buchel_moe_3d_analog_in_memory_llms.md)
- [Analog in-memory attention for LLMs](core_analog_transformers/leroux_analog_in_memory_attention_llms.md)

Use these papers only for comparison framing. The current branch deliberately
does not simulate analog noise, quantization, clipping, or learned hardware
ranges. Its immediate question is narrower:

> Can DRN MLP replacements recover the frozen digital teacher under ordinary
> digital teacher-student distillation?

## DRN And Equilibrium Context

- [Equilibrium propagation](core_drn_eqprop/scellier_bengio_equilibrium_propagation.md)
- [Fast nonlinear resistive network solver](core_drn_eqprop/scellier_fast_nonlinear_resistive_network_solver.md)
- [Analog neural networks with equilibrium propagation](core_drn_eqprop/kendall_analog_neural_networks_eqprop.md)

Use these papers when changing DRN internals, solver iterations, conductance
constraints, drive injection, or trainable amplification. They are less central
when only changing OPT distillation loops or trainable parameter policies.

## Transformer Reference

- [Attention Is All You Need](core_transformers/vaswani_attention_is_all_you_need.md)
- [GPT-2](core_transformers/gpt2_unsupervised_multitask_learners.md)
- [GLU variants](core_transformers/shazeer_glu_variants_transformer.md)

These are background references. For this branch, the concrete transformer is
OPT-125M, whose MLP is ReLU-based and whose decoder block applies the MLP delta
inside the normal residual stream.

## Branch-Specific Implementation Notes

Relevant local package:

- [`opt_mlp_drn`](../opt_mlp_drn/README.md)

Relevant case notes:

- [OPT MLP DRN recovery case](../cases/transformer_cases/opt_mlp_drn_recovery_20260505/README.md)
- [Layer-count KL sweep](../cases/transformer_cases/opt_mlp_drn_recovery_20260505/layer_count_kl_sweep.md)

When deciding what to read for an implementation task:

1. If the task is about trainable parameter scope, start with Core PEFT.
2. If the task is about KL, hidden-state matching, or progressive replacement,
   start with Knowledge Distillation And Module Replacement.
3. If the task compares against analog foundation-model procedures, start with
   Analog-Aware Transformer Baselines.
4. If the task modifies DRN physics, conductances, nonlinearities, drive
   injection, amplification, or solver behavior, start with DRN And Equilibrium
   Context.

