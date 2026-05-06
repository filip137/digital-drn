# Hinton, Vinyals, and Dean - Distilling the Knowledge in a Neural Network

## Links

- arXiv: https://arxiv.org/abs/1503.02531
- PDF: https://arxiv.org/pdf/1503.02531
- Local PDF cache: `literature/_pdf_cache/core_knowledge_distillation/hinton_distilling_knowledge.pdf`

## Summary

This is the foundational teacher-student distillation paper. It motivates
training a student model to mimic a teacher using soft outputs. In this project,
the pretrained Transformer MLP, block, or full model can act as teacher while a
DRN block acts as student.

## Use For

- Teacher-student framing.
- Soft-target distillation.
- Explaining why DRN initialization can be done with backpropagation before
  later EqProp-style training.

## Notes For This Repo

Use this paper for the general KD framing. For DRN MLP replacement, the first
objective can be local teacher-output matching before trying full-model EP or
partial fine-tuning.
