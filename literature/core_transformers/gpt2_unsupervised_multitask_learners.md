# Radford et al. - Language Models are Unsupervised Multitask Learners

## Links

- Official PDF: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Official page: https://openai.com/index/better-language-models/
- Local PDF cache: `literature/_pdf_cache/core_transformers/gpt2_unsupervised_multitask_learners.pdf`

## Summary

This is the key reference when experiments use GPT-2-like decoder-only
Transformers. It frames a large autoregressive Transformer as a general-purpose
language model. For this project, GPT-2 is the practical teacher architecture
when reasoning about cached hidden states, residual streams, MLP blocks, and
language-modeling loss after DRN substitution.

## Use For

- Decoder-only Transformer architecture.
- GPT-style residual stream and block-local replacement.
- Language-modeling evaluation after local MLP substitution.

## Notes For This Repo

When a DRN block locally matches a GPT-2 MLP output, still evaluate model-level
language-modeling loss. Local imitation error and global autoregressive behavior
are related but not interchangeable.
