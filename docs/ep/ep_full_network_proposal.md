# Proposal: Full-Network Hybrid BP-EP In `digital_drn`

This note turns the algorithm plan in [ep_full_network_plan.md](/home/filip/digital_drn/ep_full_network_plan.md) into a concrete code-structure proposal.

The goal is to implement training for the full alternating network:

```text
digital block F_1
DRN block E_1
digital block F_2
DRN block E_2
...
digital readout F_N
```

with:

- DRN blocks trained by centered EP with current-force nudging
- digital modules trained by explicit VJPs
- no differentiation through the DRN solver

## 1. Design Principles

The implementation should follow these rules.

### Keep the old DRN logic where it belongs

The old DRN EP stack had a good separation:

- wrapped layers and parameters with `.state`
- scalar energy object
- augmented energy
- inference minimizer
- training minimizer
- EP estimator
- trainer writes `.grad`
- optimizer updates `.state`

That should remain true for each DRN block.

### Keep digital coupling explicit

The digital side should not be hidden inside the DRN solver.

The trainer should:

- get `delta_h_K` from the final head by standard autograd
- ask each DRN block for `delta_drive_k`
- run a local VJP through the preceding digital block

This keeps the analog and digital pieces modular.

### Keep block-local EP separate from the trainer

The trainer should orchestrate.
It should not manually:

- restore free states
- flip nudging signs
- call `grad_param_fn` itself

That logic should live in one reusable block-local EP helper.

### Reuse current `BPTrainer` infrastructure

We should not rewrite:

- checkpointing
- history
- TensorBoard summaries
- optimizer/scheduler creation
- evaluation loop

We only need a new training step and a few new cache/helper classes.

## 2. Proposed File Structure

I would add the EP code in a small cluster of new files instead of spreading it across `trainer.py`, `block.py`, and `experiment.py`.

Suggested layout:

```text
digital_drn/
  augmented.py                  # already exists
  block.py
  conv_block.py
  network.py
  network_digital_analog.py
  trainer.py                    # keep BPTrainer here
  experiment.py

  ep_types.py                   # NEW: dataclasses and protocols
  ep_block.py                   # NEW: block-local EP helper
  ep_digital.py                 # NEW: local digital VJPs
  ep_forward.py                 # NEW: free forward with caches
  ep_trainer.py                 # NEW: HybridEPTrainer

  tests/
    test_augmented.py
    test_explicit_gradient_coupling.py
    test_two_block_partial_ep_overfit.py
    test_ep_block.py            # NEW
    test_ep_forward.py          # NEW
    test_ep_trainer.py          # NEW
```

This keeps the EP implementation readable and lets BP and EP coexist cleanly.

## 3. Responsibilities By File

### `augmented.py`

Keep this focused on:

- `Nudging`
- `AugmentedFunction`

Semantics:

- `Nudging` is a fixed current-force term
- no cost-function object
- no quadratic cost contribution

I would keep the default mode:

- `mode="current"`

and not add a second EP variant until the first one works end to end.

### `block.py` and `conv_block.py`

Blocks should own runtime mechanics, not trainer logic.

They should expose:

- `set_drive(h_prev) -> drive`
- `output_state()`
- `output_layer()`
- `resistive_params()`
- `resistive_param_states()`
- `reset_state(...)`
- `detach_state_()`
- `clamp_resistive_params_()`

New methods to add:

```python
def capture_free_cache(self, h_prev) -> BlockFreeCache:
    ...

def restore_free_cache(self, cache: BlockFreeCache) -> None:
    ...

def first_free_layer(self):
    ...

def free_layers(self):
    ...
```

And one small but important addition:

- store `self.augmented_minimizer`

so each block owns:

- inference minimizer on `self.energy`
- training minimizer on `self.augmented_energy`

That matches the old DRN structure much better than recreating a minimizer in tests or in the trainer.

### `ep_types.py`

This file should only hold small dataclasses.

Suggested contents:

```python
@dataclass
class BlockFreeCache:
    h_prev: torch.Tensor
    drive: torch.Tensor
    free_state: list[torch.Tensor]
    output: torch.Tensor

@dataclass
class BlockEPResult:
    param_grads: list[torch.Tensor]
    delta_drive: torch.Tensor
    plus_output: torch.Tensor
    minus_output: torch.Tensor

@dataclass
class NetworkFreeCache:
    block_caches: list[BlockFreeCache]
    final_hidden: torch.Tensor
    logits: torch.Tensor | None
```

No methods here. Only transport objects.

### `ep_block.py`

This should be the analog core.

Suggested main object:

```python
class BlockEquilibriumProp:
    def __init__(self, block, beta: float):
        ...

    def compute_gradients(
        self,
        *,
        free_cache: BlockFreeCache,
        output_cotangent: torch.Tensor,
    ) -> BlockEPResult:
        ...
```

This object should do:

1. prepare nudging from `output_cotangent`
2. restore free state
3. run `+beta` equilibrium with `block.augmented_minimizer`
4. restore free state
5. run `-beta` equilibrium with `block.augmented_minimizer`
6. compute centered parameter gradients with `block.energy.grad_param_fn`
7. compute `delta_drive`

For v1, `delta_drive` should use the closed form:

```text
delta_drive = (z1_minus - z1_plus) / (2 * beta)
```

because the drive coupling is:

```text
E_drive = - <z1, drive>
```

So this file becomes the direct analogue of the old DRN `EquilibriumProp`, but at block scope.

### `ep_digital.py`

This file should hold only local digital backward helpers.

Suggested function:

```python
def vjp_ff_block(block, h_prev, delta_drive):
    ...
    return ff_param_grads, delta_h_prev
```

Implementation idea:

```python
h_prev_leaf = h_prev.detach().clone().requires_grad_(True)
drive = block.drive_scale * block.ff(h_prev_leaf)
grads = torch.autograd.grad(
    outputs=drive,
    inputs=(h_prev_leaf, *block.ff_parameters()),
    grad_outputs=delta_drive,
)
```

This gives:

- `delta_h_prev`
- FF parameter grads
- drive-scale grad if enabled

Nothing in this file should know about DRN states.

### `ep_forward.py`

This file should gather free-phase caches from the network.

Suggested helper:

```python
def forward_free_with_cache(model, inputs, *, reset, num_iterations) -> NetworkFreeCache:
    ...
```

It should:

1. run each block once in free mode
2. capture per-block:
   - `h_prev`
   - `drive`
   - free analog states
   - output
3. run the head if present
4. return a structured cache

This keeps free-pass bookkeeping out of the trainer.

### `ep_trainer.py`

This should implement:

```python
class HybridEPTrainer(BPTrainer):
    ...
```

Reuse everything from `BPTrainer` except the training step.

The key overridden method should be:

```python
def train_epoch(...):
    ...
```

and inside it, each batch should do:

1. free forward with cache
2. head BP
3. reverse sweep through blocks:
   - `BlockEquilibriumProp`
   - `vjp_ff_block`
4. write grads into `.grad`
5. `optimizer.step()`
6. clamp DRN params
7. detach states

This keeps all EP orchestration in one place.

## 4. How To Wire It Into Existing Code

### `network.py` and `network_digital_analog.py`

Add only minimal cache-friendly helpers.

I would not put the full EP loop into the network class.

Good additions:

- `iter_blocks()`
- `has_head()`
- maybe `head_parameters()`

Bad additions:

- trainer-like loops
- explicit gradient writing
- optimizer logic

The network should remain a model, not a trainer.

### `experiment.py`

Current behavior:

- `algorithm != "bp"` raises `NotImplementedError`

Target behavior:

```python
if algorithm_name == "bp":
    return BPTrainer(...)
if algorithm_name == "ep":
    return HybridEPTrainer(...)
raise ValueError(...)
```

Add an EP config section with:

- `beta`
- `variant: centered`
- `freeze_bn_stats: true`

Nothing more initially.

### `trainer.py`

I would leave `BPTrainer` where it is.

Two options:

1. keep `HybridEPTrainer` in `ep_trainer.py`
2. import it from there in `__init__.py`

That is cleaner than turning `trainer.py` into a mixed BP/EP file.

## 5. Recommended APIs

These are the interfaces I would standardize first.

### Block API

```python
class SomeDigitalDRNBlock(nn.Module):
    def set_drive(self, h_prev) -> torch.Tensor: ...
    def output_state(self) -> torch.Tensor: ...
    def output_layer(self): ...
    def first_free_layer(self): ...
    def free_layers(self) -> list: ...
    def capture_free_cache(self, h_prev) -> BlockFreeCache: ...
    def restore_free_cache(self, cache: BlockFreeCache) -> None: ...
    def resistive_params(self) -> list: ...
    def ff_parameters(self) -> list[nn.Parameter]: ...
```

### Block EP API

```python
class BlockEquilibriumProp:
    def compute_gradients(
        self,
        *,
        free_cache: BlockFreeCache,
        output_cotangent: torch.Tensor,
    ) -> BlockEPResult: ...
```

### Digital backward API

```python
def vjp_ff_block(block, h_prev, delta_drive) -> tuple[list[torch.Tensor], torch.Tensor]:
    ...
```

### Free forward API

```python
def forward_free_with_cache(model, x, *, reset, num_iterations) -> NetworkFreeCache:
    ...
```

These are enough to build the full trainer without overcomplicating the first version.

## 6. Suggested Rollout

### Phase 1. Clean block-local EP

Implement:

- `ep_types.py`
- `ep_block.py`
- `capture_free_cache()` and `restore_free_cache()` on blocks
- `augmented_minimizer` on blocks

Success condition:

- current single-block and two-block tests still pass

### Phase 2. Last-block training through normal trainer entrypoint

Implement `HybridEPTrainer` that trains only:

- final head
- last DRN block

with earlier blocks frozen

Success condition:

- replicate the current test behavior through the real trainer

### Phase 3. One-step digital coupling

Implement `vjp_ff_block` and train:

- final head
- last DRN block
- last FF block before it

Success condition:

- 2-block toy model overfits through the real training loop

### Phase 4. Full reverse sweep

Generalize the trainer over all blocks.

Success condition:

- tiny dense synthetic problem overfits
- then tiny MNIST subset

### Phase 5. Conv path

Extend the exact same interfaces to:

- `ConvDigitalDRNBlock`
- `ConvDenseDigitalDRNBlock`

Success condition:

- small conv/MNIST or CIFAR subset can run through EP trainer

## 7. What To Avoid

To keep the design clean:

- do not put full EP logic into `block.py`
- do not put full EP logic into `network.py`
- do not make the trainer directly manipulate raw layer lists long-term
- do not reintroduce `cost_fn`-style augmented energy
- do not differentiate through the DRN solver
- do not support multiple EP variants at first

The first correct version is more valuable than a flexible but muddy implementation.

## 8. Bottom Line

The code should be structured as:

```text
block runtime
  -> owns energy, augmented_energy, minimizers, state

block EP helper
  -> computes centered EP grads and delta_drive

digital VJP helper
  -> computes FF grads and previous cotangent

free forward helper
  -> builds caches for one batch

HybridEPTrainer
  -> orchestrates the whole reverse sweep
```

This keeps the structure close to the original DRN code where it matters, while making multi-block digital coupling explicit and testable.
