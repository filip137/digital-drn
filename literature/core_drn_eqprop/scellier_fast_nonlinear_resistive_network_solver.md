# Scellier - A Fast Algorithm to Simulate Nonlinear Resistive Networks

## Links

- arXiv: https://arxiv.org/abs/2402.11674
- PDF: https://arxiv.org/pdf/2402.11674
- Local PDF cache: `literature/_pdf_cache/core_drn_eqprop/scellier_fast_nonlinear_resistive_network_solver.pdf`

## Summary

This paper is critical for scalable DRN simulation. It formulates nonlinear
resistive network simulation as an optimization problem and solves it
efficiently using coordinate descent. It should guide experiment design when
solver speed, convergence, CD sweeps, SPICE comparison, or Transformer-loop
scaling are relevant.

## Use For

- Fast DRN simulation.
- Coordinate descent solver design.
- Comparison against SPICE.
- Deciding whether DRN-in-Transformer experiments are computationally feasible.

## Notes For This Repo

Use this paper when reasoning about minimizer iterations, voltage tolerance,
runtime, and whether a proposed Transformer replacement experiment is feasible.
