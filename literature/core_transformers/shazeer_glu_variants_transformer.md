# Shazeer - GLU Variants Improve Transformer

## Links

- arXiv: https://arxiv.org/abs/2002.05202
- PDF: https://arxiv.org/pdf/2002.05202
- Local PDF cache: `literature/_pdf_cache/core_transformers/shazeer_glu_variants_transformer.pdf`

## Summary

This paper studies gated FFN variants such as ReGLU, GEGLU, and SwiGLU inside
Transformer feed-forward layers. It is directly relevant to MLP replacement
because modern Transformer MLPs are not always simple ReLU or GELU MLPs.

## Use For

- Modern Transformer MLP variants.
- Comparing ReLU, GELU, and gated FFNs.
- Identifying architectural mismatch between a DRN student and a teacher MLP.

## Notes For This Repo

Before designing a DRN replacement, inspect the teacher block. GPT-2-style GELU
MLPs, LLaMA-style SwiGLU MLPs, and simpler ReLU MLPs may need different
distillation targets or DRN capacities.
