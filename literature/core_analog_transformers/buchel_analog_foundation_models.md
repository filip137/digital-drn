# Buchel et al. - Analog Foundation Models

## Links

- arXiv: https://arxiv.org/abs/2505.09663
- arXiv HTML: https://arxiv.org/html/2505.09663
- PDF: https://arxiv.org/pdf/2505.09663
- OpenReview: https://openreview.net/forum?id=zo4zYTR8vn
- IBM page: https://research.ibm.com/publications/analog-foundation-models
- Local PDF cache: `literature/_pdf_cache/core_analog_transformers/buchel_analog_foundation_models.pdf`

## Summary

This paper studies foundation models trained to be robust to analog
in-memory-computing hardware noise and constraints. It is one of the most
important references for adapting large pretrained models to analog hardware.

## Use For

- Analog-aware foundation-model training.
- Robustness to analog hardware noise.
- Comparing DRN-based analog modules against AIMC baselines.
- Deciding retraining scope: full model, projections, adapters, or hardware-aware components.

## Notes For This Repo

Use this paper when deciding whether DRN replacement should be trained locally,
with partial fine-tuning, or with broader analog-aware retraining.
