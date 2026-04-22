# digital_drn

`digital_drn` is a self-contained prototype package for digital feedforward frontends coupled to dense resistive equilibrium blocks.

Current scope:

- dense resistive blocks
- fixed-resolution conv resistive blocks
- feedforward current injection into the first free layer
- equilibrium solved with a quadratic minimizer
- backpropagation through unrolled equilibrium updates
- MNIST smoke-training utilities
- CIFAR-10 digital/analog proof-of-concept model wiring
- GPT-style transformer blocks with tokenwise DRN MLPs
- synthetic MQAR smoke data for associative-recall transformer checks

The package no longer depends on `/home/filip/server_code` at runtime. The minimal DRN core used by the prototype is vendored under [`core/`](/home/filip/digital_drn/core).

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e .[dev]
```

## CLI

Dry-run the default config:

```bash
digital-drn-train --dry-run
```

Run the conv MNIST config for 50 epochs:

```bash
digital-drn-train --model conv_mnist_1block --epochs 50
```

Run the widened CIFAR-10 signed-readout config through its dedicated wrapper:

```bash
digital-drn-cifar10-wider-train -- --epochs 100 --batch-size 64 --download
```

Use a smaller subset for a quick check:

```bash
digital-drn-train --model conv_mnist_1block --epochs 2 --train-subset 2048 --test-subset 512
```

Run the initial MQAR transformer/DRN smoke config:

```bash
digital-drn-train --config-name mqar_drn_gpt_smoke --device cpu --epochs 1 --max-steps 1 --no-save
```

## Outputs

By default, runs write into [`simulation_results/`](/home/filip/digital_drn/simulation_results) using:

- `simulation_results/<model-config-name>/<timestamp>/`

The trainer currently stores:

- `run_metadata.json`
- `experiment_config.json`
- `trainer_config.json`
- `history.json`
- `checkpoint_best.pt` by default
- `checkpoint_epoch_XXXX.pt` only if `save_every > 0`
- `checkpoint_last.pt` only if `save_last: true`
- TensorBoard event files `events.out.tfevents.*`

## Main objects

- `DigitalDRNBlock`
- `build_dense_drn_block(...)`
- `build_conv_drn_block(...)`
- `build_conv_dense_drn_block(...)`
- `DenseDRNBlockEnergy`
- `ConvDRNBlockEnergy`
- `ConvDenseDRNBlockEnergy`
- `DigitalDRNNet`
- `DigitalAnalogNet`
- `SequentialDigitalDRNNet`
- `SmallDRNGPT`
- `TokenwiseDRNMLP`
- `MQARDataset`
- `build_default_dense_ff(...)`
- `FFCurrentInteraction`
- `QuadraticMinimizer`

`DigitalDRNBlock` is now the shared runtime wrapper. The `build_*_drn_block(...)` helpers choose the energy geometry and optionally build a default FF frontend.

## Frontends

FF construction is separated from the runtime block:

- pass a custom `ff: nn.Module` directly into `DigitalDRNBlock(...)` or any `build_*_drn_block(...)`
- use [`frontends.py`](/home/filip/digital_drn/frontends.py) for the default dense/conv frontend builder

That makes it straightforward to swap in richer FF modules such as attention or custom hybrid frontends without changing the DRN runtime class.

## CIFAR-10 v0

The repository now includes a proof-of-concept alternating digital/analog CIFAR-10 architecture spec:

- [cifar10_digital_analog_v0.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_digital_analog_v0.yaml)
- [cifar10.yaml](/home/filip/digital_drn/hydra_conf/data/cifar10.yaml)

This path is intentionally minimal:

- two digital-to-analog handoffs
- fixed-resolution conv DRN blocks
- no residuals
- no pooling or normalization inside DRN blocks

It is meant as a coupling/debugging baseline, not a final high-accuracy architecture.

## CIFAR Overfit Debug

The main architecture debugging record lives here:

- [CIFAR-10 Overfit Debug](/home/filip/digital_drn/cases/conv_cases/cifar10_overfit_debug/README.md)

The practical rule from those runs is:

- before spending time on full CIFAR training, first check whether the model can memorize `128` training samples

Current takeaways from that overfit suite:

- worked:
  - digital conv + dense DRN readout
  - one-block mixed analog readout
  - two-block mixed analog readout
  - the original `v0` stack once its pooled output head was replaced with a flat readout
- failed or underperformed:
  - the original pooled-head `v0` setup
  - early conv-analog-block + digital-head formulations
  - adding a final digital `10 -> 10` head on top of a working mixed-analog readout

So for new CIFAR experiments, prefer architectures with an analog dense readout and treat extra digital output heads as suspicious until they pass the tiny-subset overfit test.

## Example

```python
from digital_drn import SequentialDigitalDRNNet

net = SequentialDigitalDRNNet(
    input_dim=28 * 28,
    block_configs=[{"layer_dims": [128, 10], "weight_gains": [0.05], "bias_gain": 0.0}],
    num_iterations=4,
    mode="asynchronous",
    ff_activation="tanh",
    non_linearity="linear",
    weight_min=0.0,
    weight_max=1.0,
)
```
