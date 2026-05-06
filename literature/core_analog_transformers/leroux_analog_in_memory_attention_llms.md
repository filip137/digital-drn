# Leroux et al. - Analog In-Memory Computing Attention Mechanism for Fast and Energy-Efficient Large Language Models

## Links

- arXiv: https://arxiv.org/abs/2409.19315
- PDF: https://arxiv.org/pdf/2409.19315
- Nature Computational Science: https://www.nature.com/articles/s43588-025-00854-1
- Local PDF cache: `literature/_pdf_cache/core_analog_transformers/leroux_analog_in_memory_attention_llms.pdf`

## Summary

This paper proposes an analog in-memory self-attention architecture based on
gain-cell memories for generative Transformers. It shows that analog LLM
acceleration is not only about MLP layers; attention and KV-cache movement are
also major bottlenecks.

## Use For

- Analog attention.
- Gain-cell memory.
- KV-cache and self-attention bottlenecks.
- Comparing analog-attention approaches with DRN MLP replacement.

## Notes For This Repo

Use this paper as a counterweight to MLP-only replacement. Even if DRNs target
MLPs first, attention and memory movement remain important for full LLM
efficiency.
