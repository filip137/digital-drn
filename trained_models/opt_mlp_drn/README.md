# OPT MLP DRN Trained Models

This directory stores curated reusable trained artifacts for OPT MLP-to-DRN replacement experiments.

Raw experiment outputs remain under `simulation_results/`. This folder is for selected checkpoints that should be reused by later training/evaluation jobs. Large `.pt` binaries should stay local or be uploaded to an artifact store; git should track only metadata and documentation.

Current entries:

- `single_block_norm_recovered_20260506/`: OPT-125M single-block DRN checkpoints for layers 9, 10, and 11 with recovered cosine and output norm.
