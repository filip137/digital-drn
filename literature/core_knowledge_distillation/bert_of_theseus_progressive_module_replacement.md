# Xu et al. - BERT-of-Theseus

## Links

- arXiv: https://arxiv.org/abs/2002.02925
- PDF: https://arxiv.org/pdf/2002.02925
- ACL: https://aclanthology.org/2020.emnlp-main.633/
- Local PDF cache: `literature/_pdf_cache/core_knowledge_distillation/bert_of_theseus_progressive_module_replacement.pdf`

## Summary

This is one of the most important references for module replacement. It divides
BERT into modules, builds compact substitutes, and progressively increases the
replacement probability during training. For DRN work, it motivates schedules
that replace Transformer MLPs gradually rather than swapping all MLPs at once.

## Use For

- Progressive replacement.
- Probabilistic teacher-module versus student-module substitution.
- Staged DRN insertion into pretrained Transformers.

## Notes For This Repo

Use this when designing DRN MLP replacement schedules. A practical experiment
can replace one block first, then increase the number or probability of DRN
substitutions after local KD succeeds.
