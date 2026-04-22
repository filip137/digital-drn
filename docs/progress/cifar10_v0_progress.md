# CIFAR-10 v0 Progress

This note tracks the current state of the alternating digital/analog CIFAR-10 proof of concept.

## Implemented

- fixed-resolution convolutional DRN energy blocks via:
  - `ConvWeight`
  - `ConvResistive`
  - `ConvDRNBlockEnergy`
- conv digital-to-analog block wrapper:
  - `ConvDigitalDRNBlock`
- alternating top-level model:
  - `DigitalAnalogNet`
- digital head support:
  - `DigitalClassifierHead`
- YAML model spec:
  - `hydra_conf/model/cifar10_digital_analog_v0.yaml`
- CIFAR-10 data config:
  - `hydra_conf/data/cifar10.yaml`
- builder support in `experiment.py`
- CIFAR-10 dataset branch in `build_dataloaders_from_config()`

## Current Architecture

The current v0 path is:

```text
x
-> ConvDigitalDRNBlock(F1 + E1)
-> ConvDigitalDRNBlock(F2 + E2)
-> DigitalClassifierHead(F3)
-> logits
```

Where:

- the digital submodule produces the static drive tensor
- the analog conv DRN block relaxes to equilibrium
- the final analog state is passed to the next stage

## What Is Still Intentionally Limited

- no residual connections
- no attention
- no normalization inside DRN blocks
- no pooling inside DRN blocks
- no stride inside DRN blocks
- no mixed dense/conv generic block builder yet
- no EP implementation for the conv architecture

## What Still Needs Work

1. CIFAR-10 training smoke run through the CLI.
2. Tiny-subset overfit test for the v0 architecture.
3. Better diagnostics for conv analog equilibria:
   - state norms
   - coefficient magnitudes
   - repeatability under reset
4. Optional support for transition Option B in experiments.
5. Later generalization toward shared base block classes.

## Why This Matters

The main blocker before was missing conv analog physics. That part is now present.

The remaining work is mostly:

- architecture stabilization
- training/debugging
- eventual cleanup/generalization

This is the right point to start proof-of-concept CIFAR-10 experiments.
