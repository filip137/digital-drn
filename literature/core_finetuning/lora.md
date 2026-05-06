# Hu et al. - LoRA

## Links

- arXiv: https://arxiv.org/abs/2106.09685
- PDF: https://arxiv.org/pdf/2106.09685
- OpenReview: https://openreview.net/forum?id=nZeVKeeFYf9
- Local PDF cache: `literature/_pdf_cache/core_finetuning/lora.pdf`

## Summary

LoRA freezes pretrained weights and injects trainable low-rank updates into
Transformer layers. It is the strongest default PEFT baseline for DRN
experiments because it is simple, widely used, and directly targets Transformer
projections.

## Use For

- Main PEFT baseline.
- Low-rank adaptation.
- Comparing DRN replacement against standard parameter-efficient methods.

## Notes For This Repo

Compare DRN replacement or DRN adapters against LoRA on MLP projections,
attention projections, and broader Transformer targets before claiming a PEFT
advantage.
