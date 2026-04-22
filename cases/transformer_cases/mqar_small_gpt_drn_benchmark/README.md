# MQAR Small-GPT DRN Benchmark

This case records the small-gpt-compatible MQAR benchmark for `SmallDRNGPT`,
where the transformer MLP sublayers are replaced by tokenwise DRN MLP blocks.

The dataset and training protocol mirror `/home/filip/small_gpt`:

- `vocab_size: 512`
- `seq_len: 64`
- `num_kv_pairs: 4`
- `train_examples: 50000`
- `val_examples: 5000`
- `batch_size: 128`
- `lr: 3e-4`
- `warmup_steps: 200`
- `max_steps: 3000`
- validation every 200 steps

DRN settings:

- `drn_non_linearity: perfect_diode`
- `drn_mode: asynchronous`
- `drn_signed_drive: true`
- `train_num_iterations: 6`
- `eval_num_iterations: 6`

## Run

```bash
cd /home/filip/digital_drn
/home/filip/miniconda3/envs/py312/bin/python -m digital_drn.app.mqar_benchmark_cli --device cuda
```

Run directory:

```text
/home/filip/digital_drn/simulation_results/mqar_small_gpt_drn/20260422-160520-nom-cool-2
```

The run directory contains:

- `config.json`
- `run_metadata.json`
- `history.json`
- `best.pt`

## Result

Final and best validation metrics at step 3000:

| step | train loss | val loss | val query acc | val exact acc |
| ---: | ---: | ---: | ---: | ---: |
| 3000 | 0.0387 | 0.0054 | 0.9983 | 0.9934 |

This recovers the small-gpt target performance to within the expected range.
The earlier MQAR results in `mqar_drn_gpt_smoke` should be interpreted as
compact overfit/smoke checks, not as the full small-gpt benchmark.
