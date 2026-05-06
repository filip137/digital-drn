# Sung, Cho, and Bansal - Ladder Side-Tuning

## Links

- arXiv: https://arxiv.org/abs/2206.06522
- PDF: https://arxiv.org/pdf/2206.06522
- Code: https://github.com/ylsung/ladder-side-tuning
- Local PDF cache: `literature/_pdf_cache/core_finetuning/ladder_side_tuning.pdf`

## Summary

Ladder Side-Tuning trains a side network connected to intermediate backbone
representations through ladder connections. It separates parameter efficiency
from training-memory efficiency: many PEFT methods update few parameters but
still need backpropagation through the frozen backbone.

## Use For

- Side-network adaptation.
- Training-memory-efficient PEFT.
- Comparing DRN side networks or DRN adapters against memory-aware baselines.

## Notes For This Repo

Use LST when making memory-efficiency arguments. A DRN method that still
backpropagates through the full Transformer backbone may not have the same
training-memory profile as a side-tuning method.
