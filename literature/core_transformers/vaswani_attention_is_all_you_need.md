# Vaswani et al. - Attention Is All You Need

## Links

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762
- Local PDF cache: `literature/_pdf_cache/core_transformers/vaswani_attention_is_all_you_need.pdf`

## Summary

This is the original Transformer paper. It defines the canonical Transformer
block with attention, residual connections, layer normalization, and the
position-wise feed-forward network. For DRN-in-Transformer work, treat it as
the reference for what the MLP/FFN sublayer is and where it sits inside the
block.

## Use For

- Transformer block structure.
- Position-wise FFN/MLP definition.
- Defining the module a DRN block replaces.

## Notes For This Repo

Use this as the baseline architectural reference when mapping `digital_drn`
blocks into Transformer-style residual streams. The DRN replacement target is
the position-wise FFN behavior, not the attention mechanism.
