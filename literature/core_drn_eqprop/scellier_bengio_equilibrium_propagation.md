# Scellier and Bengio - Equilibrium Propagation

## Links

- arXiv: https://arxiv.org/abs/1602.05179
- PDF: https://arxiv.org/pdf/1602.05179
- Frontiers: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full
- Local PDF cache: `literature/_pdf_cache/core_drn_eqprop/scellier_bengio_equilibrium_propagation.pdf`

## Summary

This is the theoretical foundation of Equilibrium Propagation. It defines
learning through a free phase and a nudged phase in energy-based systems. In
the current Transformer branch, initial DRN replacement may be trained by BP/KD,
but EqProp remains the long-term motivation for local energy-based training.

## Use For

- EqProp theory.
- Free and nudged phases.
- Local learning in energy-based systems.
- Distinguishing initial KD from later EqProp-style training.

## Notes For This Repo

Use this paper when explaining why `digital_drn` keeps free-state and nudged
state diagnostics separate from ordinary backpropagation diagnostics.
