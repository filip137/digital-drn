# Jiao et al. - TinyBERT

## Links

- arXiv: https://arxiv.org/abs/1909.10351
- PDF: https://arxiv.org/pdf/1909.10351
- ACL: https://aclanthology.org/2020.findings-emnlp.372/
- ACL PDF: https://aclanthology.org/2020.findings-emnlp.372.pdf
- Local PDF cache: `literature/_pdf_cache/core_knowledge_distillation/tinybert.pdf`

## Summary

TinyBERT is a Transformer-specific KD paper. It distills Transformer-layer
knowledge during both general pretraining-style distillation and task-specific
distillation. It is useful for deciding which internal signals a DRN should
mimic: MLP outputs, residual outputs, hidden states, logits, or combinations.

## Use For

- Transformer-specific KD.
- Layer-wise matching.
- Two-stage distillation.
- Choosing internal imitation signals for DRN replacement.

## Notes For This Repo

Use TinyBERT as a template for structured DRN experiments: start with local
block matching, then test whether that match survives in full-model loss and
task behavior.
