# Base Class Refactor TODO

This note records the current implementation strategy and the follow-up refactor needed to make the architecture more general.

## Current Decision

Use the least disruptive path first:

- keep the current `DigitalDRNBlock` as the dense implementation
- add a separate `ConvDigitalDRNBlock` for fixed-resolution conv analog states
- add a top-level `DigitalAnalogNet` that chains blocks and an optional digital head
- add a CIFAR-10 builder for the v0 alternating digital/analog architecture

This avoids refactoring the whole block hierarchy before the first conv digital-analog proof of concept runs.

## Immediate Implementation Path

1. Keep `DigitalDRNBlock` unchanged as the dense block.
2. Add `ConvDigitalDRNBlock`:
   - digital conv frontend / transition frontend
   - `ConvDRNBlockEnergy`
   - `QuadraticMinimizer`
   - same public methods as the dense block where possible
3. Add `DigitalClassifierHead`:
   - global average pool
   - flatten
   - linear classifier
4. Add `DigitalAnalogNet`:
   - chain `block1 -> block2 -> head`
   - preserve `set_device`, optimizer-group, reset, and detach helpers
5. Extend the YAML builder for:
   - `cifar10_digital_analog_v0`
   - CIFAR-10 dataloading

## Future Generalization Goal

After the first conv proof of concept works, refactor toward explicit base classes.

### Target Hierarchy

1. `BaseDigitalDRNBlock(nn.Module)`
- owns shared mechanics only
- responsibilities:
  - `drive_scale`
  - device synchronization
  - state reset
  - equilibrium helper
  - optimizer-group helper
  - resistive parameter exposure

2. `DenseDigitalDRNBlock(BaseDigitalDRNBlock)`
- current dense implementation
- dense feedforward drive
- dense DRN energy

3. `ConvDigitalDRNBlock(BaseDigitalDRNBlock)`
- conv digital frontend
- conv DRN energy
- fixed-resolution analog refinement

4. `BaseDigitalHead(nn.Module)`
- interface for optional output heads

5. `DigitalClassifierHead(BaseDigitalHead)`
- global average pooling + linear classifier

6. `DigitalAnalogNet(nn.Module)`
- generic block chain plus optional head
- should not assume dense-only or conv-only blocks

## Shared Block Interface To Standardize

Every block subclass should expose the same public surface:

- `forward(x, reset=False, num_iterations=None)`
- `set_device(device)`
- `reset_state(batch_size, device)`
- `detach_state_()`
- `clamp_resistive_params_()`
- `enable_resistive_grad_(enabled=True)`
- `resistive_params()`
- `resistive_param_states()`
- `optimizer_param_groups()`
- `output_state()`

This lets the top-level network and trainer stay agnostic to dense vs conv blocks.

## Refactor Steps

### Phase 1

Extract shared logic from the current dense block into helper methods without changing behavior:

- drive-scale initialization
- device canonicalization
- energy-device sync
- minimizer iteration override
- optimizer-group assembly

### Phase 2

Create `BaseDigitalDRNBlock` and move the shared logic there.

The subclass boundary should be:

- subclass defines `ff`, `energy`, and drive/output tensor conventions
- base class handles training/runtime plumbing

### Phase 3

Rename the current block explicitly:

- current `DigitalDRNBlock` becomes `DenseDigitalDRNBlock`
- optionally keep `DigitalDRNBlock` as a compatibility alias temporarily

### Phase 4

Generalize the builders:

- model YAML should choose block class from config
- builder should support mixed block types in one network
- head construction should be independent of block construction

## Config Cleanup To Do Later

Once the conv path is stable, clean up the YAML schema:

- keep `ff` and `drn` nested config structure
- allow a block-level `type`
- separate:
  - digital frontend config
  - analog energy config
  - output head config

Possible future shape:

```yaml
blocks_config:
  - type: "conv_digital_drn"
    ff: ...
    drn: ...

  - type: "dense_digital_drn"
    ff: ...
    drn: ...

head:
  type: "classifier"
  ...
```

## Testing To Add After Refactor

1. Interface parity tests:
- dense and conv blocks support the same public methods

2. Optimizer-group tests:
- FF and DRN learning rates remain distinct

3. Device/state tests:
- mixed dense/conv networks move cleanly across CPU/GPU

4. Builder tests:
- YAML can instantiate dense-only, conv-only, and mixed networks

5. Checkpoint tests:
- checkpoint/load works across the refactored block hierarchy

## Non-Goals For The First Refactor

Do not add these during the base-class extraction:

- residual connections
- attention
- stride inside DRN energy blocks
- pooling inside DRN energy blocks
- normalization inside DRN energy blocks
- EP support

Those should stay separate from the structural cleanup.
