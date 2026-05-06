# Shared Literature Packet

This packet provides shared background for agents working on DRNs, Equilibrium
Propagation, Transformer MLP replacement, knowledge distillation, partial
fine-tuning, and analog foundation models.

Markdown summaries are tracked in git. Downloaded PDFs are kept in the ignored
local cache at `literature/_pdf_cache/`.

## Download PDFs

Dry run:

```bash
python tools/download_literature_pdfs.py --dry-run
```

Download available direct PDFs:

```bash
python tools/download_literature_pdfs.py
```

The downloader reads `literature/papers.json`, writes PDFs under
`literature/_pdf_cache/<category>/`, and records checksums in the ignored file
`literature/_pdf_cache/manifest.sha256`.

## Folders

- `core_transformers/`: Transformer and GPT-style block references.
- `core_knowledge_distillation/`: teacher-student and module-replacement KD.
- `core_drn_eqprop/`: EqProp, nonlinear resistive networks, and DRN solvers.
- `core_finetuning/`: adapter, LoRA, sparse, and memory-efficient PEFT baselines.
- `core_analog_transformers/`: analog Transformer and analog foundation-model references.

## P0 Reading Order

1. `core_transformers/vaswani_attention_is_all_you_need.md`
2. `core_transformers/gpt2_unsupervised_multitask_learners.md`
3. `core_transformers/shazeer_glu_variants_transformer.md`
4. `core_knowledge_distillation/hinton_distilling_knowledge.md`
5. `core_knowledge_distillation/distilbert.md`
6. `core_knowledge_distillation/tinybert.md`
7. `core_knowledge_distillation/bert_of_theseus_progressive_module_replacement.md`
8. `core_drn_eqprop/scellier_bengio_equilibrium_propagation.md`
9. `core_drn_eqprop/kendall_analog_neural_networks_eqprop.md`
10. `core_drn_eqprop/scellier_fast_nonlinear_resistive_network_solver.md`
11. `core_finetuning/houlsby_adapters.md`
12. `core_finetuning/lora.md`
13. `core_finetuning/bitfit.md`
14. `core_finetuning/fish_mask_fixed_sparse_masks.md`
15. `core_finetuning/ladder_side_tuning.md`
16. `core_analog_transformers/leroux_analog_in_memory_attention_llms.md`
17. `core_analog_transformers/buchel_analog_foundation_models.md`
18. `core_analog_transformers/buchel_moe_3d_analog_in_memory_llms.md`
19. `core_analog_transformers/ahwa_lora_analog_transformer_adaptation.md`

These summaries are working research context for implementation agents, not
exhaustive reviews. Use the raw links in each summary for source material.
