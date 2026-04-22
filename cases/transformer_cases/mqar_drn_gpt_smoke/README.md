# MQAR DRN GPT Smoke

Initial transformer case for testing a GPT-style model whose MLP sublayers are
tokenwise dense DRN blocks.

## Task

Synthetic multi-query associative recall:

- each sample starts with unique key/value pairs
- the tail contains key queries
- only query positions are supervised
- non-query targets use `ignore_index = -100`

The first gate is a CPU one-step training smoke:

```bash
digital-drn-train --config-name mqar_drn_gpt_smoke --device cpu --epochs 1 --max-steps 1 --no-save
```

The longer local/remote experiment should keep the same config provenance and
write under `simulation_results/mqar_small_drn_gpt/`.
