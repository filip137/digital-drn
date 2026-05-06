# Sung et al. - Training Neural Networks with Fixed Sparse Masks

## Links

- arXiv: https://arxiv.org/abs/2111.09839
- PDF: https://arxiv.org/pdf/2111.09839
- OpenReview: https://openreview.net/forum?id=Uwh-v1HSw-x
- Code: https://github.com/varunnair18/FISH
- Local PDF cache: `literature/_pdf_cache/core_finetuning/fish_mask_fixed_sparse_masks.pdf`

## Summary

FISH Mask precomputes a fixed sparse mask that selects which parameters are
updated during training, using Fisher information as an importance proxy. It is
relevant when choosing principled partial fine-tuning policies instead of
arbitrary parameter subsets.

## Use For

- Fisher-based parameter selection.
- Fixed sparse trainable masks.
- DRN conductance importance estimation.

## Notes For This Repo

For DRNs, the analogous question is whether conductance importance can be
estimated from gradients, voltage-drop sensitivity, or Fisher-like scores.
