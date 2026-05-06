# Sanh et al. - DistilBERT

## Links

- arXiv: https://arxiv.org/abs/1910.01108
- PDF: https://arxiv.org/pdf/1910.01108
- Local PDF cache: `literature/_pdf_cache/core_knowledge_distillation/distilbert.pdf`

## Summary

DistilBERT is important because it combines language-modeling loss,
distillation loss, and cosine-distance loss. For DRN distillation, it suggests
that raw MSE may not be the only useful objective; cosine or normalized
hidden-state matching may matter when magnitude mismatch is less important than
direction.

## Use For

- Cosine loss.
- Multi-term distillation objectives.
- Comparing raw-output MSE against normalized hidden-state matching.

## Notes For This Repo

When DRN outputs have the right direction but wrong scale, evaluate cosine and
norm-ratio diagnostics separately instead of relying only on MSE.
