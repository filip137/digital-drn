# Li et al. - Efficient Transformer Adaptation for Analog In-Memory Computing via Low-Rank Adapters

## Links

- arXiv: https://arxiv.org/abs/2411.17367
- PDF: https://arxiv.org/pdf/2411.17367
- IBM page: https://research.ibm.com/publications/efficient-transformer-adaptation-for-analog-in-memory-computing-using-low-rank-adapters
- Local PDF cache: `literature/_pdf_cache/core_analog_transformers/ahwa_lora_analog_transformer_adaptation.pdf`

## Summary

This paper proposes analog hardware-aware LoRA for Transformer adaptation on
AIMC hardware. It is highly relevant to partial fine-tuning because it keeps
analog weights fixed and adapts using low-rank trainable components.

## Use For

- Analog hardware-aware LoRA.
- Low-rank adaptation under AIMC constraints.
- Fixed analog weights plus trainable correction.
- Direct baseline for DRN partial fine-tuning.

## Notes For This Repo

Use AHWA-LoRA as a direct comparator for DRN replacement or DRN adapters when
the experiment claims analog-hardware-aware adaptation benefits.
