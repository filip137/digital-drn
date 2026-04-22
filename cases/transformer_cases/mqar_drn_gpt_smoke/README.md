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

## 128-Sample Overfit Gate

Run date: 2026-04-22

Command:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m digital_drn.app.cli \
  --config-name mqar_drn_gpt_smoke \
  --device cpu \
  --epochs 50 \
  --train-subset 128 \
  --test-subset 128 \
  --batch-size 32 \
  --lr 0.001 \
  --log-every 4 \
  --save-last
```

Checkpoint directory:

```text
/home/filip/digital_drn/simulation_results/mqar_small_drn_gpt/20260422-153455-nom-cool-2
```

Provenance files were written:

- `run_metadata.json`
- `experiment_config.json`
- `trainer_config.json`
- `history.json`

Result:

- final train loss: `1.2045`
- final train accuracy: `0.9922`
- best train accuracy: `0.9941` at epoch 49
- final held-out eval loss: `3.4705`
- final held-out eval accuracy: `0.1289`
- best held-out eval accuracy: `0.1309`

Interpretation: the DRN-MLP transformer passes the first MQAR overfit gate. It
can memorize the fixed 128-sample associative-recall set with the tokenwise DRN
MLP path. The held-out split remains low, which is expected for this gate and
should be treated separately from the memorization check.

## 512-Sample Overfit Gate

Run date: 2026-04-22

Command:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m digital_drn.app.cli \
  --config-name mqar_drn_gpt_smoke \
  --device cpu \
  --epochs 100 \
  --train-subset 512 \
  --test-subset 512 \
  --batch-size 64 \
  --lr 0.001 \
  --log-every 8 \
  --save-last
```

Checkpoint directory:

```text
/home/filip/digital_drn/simulation_results/mqar_small_drn_gpt/20260422-154026-nom-cool-2
```

Provenance files were written:

- `run_metadata.json`
- `experiment_config.json`
- `trainer_config.json`
- `history.json`

Result:

- final train loss: `0.3174`
- final train accuracy: `0.9790`
- best train accuracy: `0.9790` at epoch 100
- final held-out eval loss: `4.1205`
- final held-out eval accuracy: `0.1958`
- best held-out eval accuracy: `0.2075` at epoch 31

Interpretation: this larger memorization gate did not fully pass under the
current smoke hyperparameters. Train accuracy improved steadily through the
last epoch but stayed below the `>0.99` memorization level reached by the
128-sample run. Held-out accuracy peaked early and then drifted down, so this
run is mainly an optimization/capacity signal rather than evidence of stronger
generalization.

Suggested next probes:

- increase optimizer aggressiveness first, e.g. `--lr 0.003`
- if that plateaus, increase DRN solve depth with
  `--train-num-iterations 4 --eval-num-iterations 4`
- if both still plateau below `0.99` train accuracy, increase transformer
  capacity before changing the MQAR task: `n_layers: 2` or `d_model: 64`
