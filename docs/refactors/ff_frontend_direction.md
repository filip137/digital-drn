# FF Frontend Direction

## Current Structure

The DRN runtime block should stay generic:

- `DigitalDRNBlock(ff, energy, ...)`

It should not decide whether the FF frontend is:

- dense
- conv
- attention
- hybrid

That choice belongs at construction time, not inside the runtime behavior.

## Current Builders

The code now separates these roles:

- runtime wrapper:
  - [DigitalDRNBlock](/home/filip/digital_drn/blocks/base.py)
- dense builder:
  - [build_dense_drn_block](/home/filip/digital_drn/blocks/block.py)
- conv builder:
  - [build_conv_drn_block](/home/filip/digital_drn/blocks/conv_block.py)
- conv+dense builder:
  - [build_conv_dense_drn_block](/home/filip/digital_drn/blocks/conv_block.py)
- default FF builder:
  - [build_default_dense_ff](/home/filip/digital_drn/frontends.py)

So the runtime block is now independent of the default FF-construction logic.

## Recommended Usage

For custom FF frontends, prefer:

```python
block = DigitalDRNBlock(
    ff=my_frontend_module,
    energy=my_energy,
    num_iterations=6,
    mode="asynchronous",
)
```

That lets the frontend be any `nn.Module`, including attention-based modules.

The convenience builders should remain optional wrappers:

- `build_dense_drn_block(...)`
- `build_conv_drn_block(...)`
- `build_conv_dense_drn_block(...)`

## Optimizer Groups

If a custom FF module exposes:

```python
optimizer_param_groups()
```

the runtime block should respect it. This is now supported in:

- [DigitalDRNBlock.optimizer_param_groups](/home/filip/digital_drn/blocks/base.py)

This allows custom frontends to define their own parameter grouping while the block still adds:

- `drive_scale`
- DRN resistive parameter groups

## Next Step

If we want to push this further, the next structural cleanup should be:

1. keep `DigitalDRNBlock` as the only runtime block class
2. keep `build_*_drn_block(...)` as convenience factories only
3. add a small frontend-builder module for any reusable default FFs
4. let `training/experiment.py` own most of the construction branching

That keeps the runtime block simple and makes new FF types easy to add.
