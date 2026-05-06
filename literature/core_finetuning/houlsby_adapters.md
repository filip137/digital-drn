# Houlsby et al. - Parameter-Efficient Transfer Learning for NLP

## Links

- arXiv: https://arxiv.org/abs/1902.00751
- PDF: https://arxiv.org/pdf/1902.00751
- Local PDF cache: `literature/_pdf_cache/core_finetuning/houlsby_adapters.pdf`

## Summary

This is the canonical adapter paper. It freezes the pretrained backbone and
inserts small trainable adapter modules. It is an important baseline for any
DRN module inserted into, beside, or after Transformer blocks.

## Use For

- Adapter baseline.
- Frozen backbone plus trainable module.
- Comparing DRN adapters or replacements against standard PEFT.

## Notes For This Repo

If a DRN is used as a side module rather than a strict MLP replacement, compare
it against adapter-style training rather than only full fine-tuning.
