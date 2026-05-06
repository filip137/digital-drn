# Ben Zaken et al. - BitFit

## Links

- arXiv: https://arxiv.org/abs/2106.10199
- PDF: https://arxiv.org/pdf/2106.10199
- ACL: https://aclanthology.org/2022.acl-short.1/
- Local PDF cache: `literature/_pdf_cache/core_finetuning/bitfit.pdf`

## Summary

BitFit updates only bias terms or a subset of bias terms. It is a minimal
partial fine-tuning baseline. If DRN replacement requires many trainable
parameters or heavy relaxation, compare it against BitFit to check whether the
DRN is actually useful.

## Use For

- Minimal sparse fine-tuning baseline.
- Bias-only updates.
- Sanity checking against very cheap adaptation.

## Notes For This Repo

Use BitFit as a low-cost baseline for partial fine-tuning after DRN
substitution. It is especially useful when the task is small or adaptation is
shallow.
