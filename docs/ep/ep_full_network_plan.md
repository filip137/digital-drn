# Full-Network EP Plan For `digital_drn`

This note proposes how to implement explicit hybrid BP-EP training for the full alternating network in `digital_drn`.

The target model is an alternating chain:

```text
digital block F_1
DRN block E_1
digital block F_2
DRN block E_2
...
digital readout F_N
```

The key design choice is:

- keep the DRN update rule close to the original DRN EP code
- but make nudging a fixed injected current coming from the digital loss gradient, not `beta * cost_fn`

So the block-local training object is still:

```text
augmented_block_energy = block_energy + nudging_interaction
```

but the nudging interaction is now linear in the exposed output state and is driven by a fixed free-phase cotangent from the digital side.

## 1. Desired Algorithm

For the final digital readout:

```text
delta_h_K = ∇_{h_K} loss(logits, y)
g_{ω_N}   = ∇_{ω_N} loss(logits, y)
```

Then for each DRN block in reverse order:

```text
s_k^{+} = argmin_s [ E_k(s, θ_k, x_k*) + β <output_k(s), delta_h_k> ]
s_k^{-} = argmin_s [ E_k(s, θ_k, x_k*) - β <output_k(s), delta_h_k> ]

g_{θ_k}   = [ ∂E_k/∂θ_k (s_k^{+}, x_k*) - ∂E_k/∂θ_k (s_k^{-}, x_k*) ] / (2β)
delta_x_k = [ ∂E_k/∂x_k (s_k^{+}, x_k*) - ∂E_k/∂x_k (s_k^{-}, x_k*) ] / (2β)

g_{ω_k}, delta_h_{k-1} = VJP(F_k, inputs=(h_{k-1}, ω_k), cotangent=delta_x_k)
```

Where:

- `h_{k-1}` is the exposed output from the previous block
- `x_k*` is the fixed free-phase drive injected into DRN block `k`
- `output_k(s)` is the DRN state slice exposed to the next digital block or head

For the current block designs, `x_k*` is the feedforward current injected into the first analog layer, and `output_k(s)` is usually the final analog state returned by `block.output_state()`.

## 2. What To Preserve From The Original DRN Code

The old DRN EP structure was good and should be preserved at block scope:

```text
Layer / Parameter wrappers
  |
  +--> .state tensors
  |
  +--> used by
       energy_fn
          |
          +--> grad_layer_fn(layer)
          +--> grad_param_fn(param)
          +--> second_fn(param)
                |
                v
           Minimizer / EP estimator

AugmentedFunction
  = energy_fn + Nudging(...)

energy_minimizer_inference
  minimizes energy_fn

energy_minimizer_training
  minimizes augmented_fn

EquilibriumProp
  uses energy_minimizer_training
  and ParamUpdater(param, energy_fn)

Trainer
  gets grads from EquilibriumProp
  writes them into param.state.grad

Optimizer(torch.optim.SGD/Adam)
  updates param.state
```

For `digital_drn`, the main change is:

- `Nudging` no longer wraps a `cost_fn`
- `Nudging` instead wraps a fixed output force from the digital side

So the new block-local structure should be:

```text
Layer / Parameter wrappers
  |
  +--> .state tensors
  |
  +--> used by
       block_energy
          |
          +--> grad_layer_fn(layer)
          +--> grad_param_fn(param)

NudgingInteraction
  = linear current-force term on block output

BlockAugmentedFunction
  = block_energy + NudgingInteraction

block_minimizer_inference
  minimizes block_energy

block_minimizer_training
  minimizes block_augmented_energy

BlockEP
  runs centered nudged phases
  returns g_theta_k and delta_x_k

HybridEPTrainer
  computes digital cotangents
  calls BlockEP block-by-block
  writes grads into param.state.grad and nn.Parameter.grad
```

## 3. Recommended Architecture

The cleanest structure is to split responsibilities into four layers.

### A. Block Runtime Layer

This is the current `DigitalDRNBlock` / `ConvDigitalDRNBlock` layer.

It should own:

- `ff`
- `energy`
- `augmented_energy`
- `minimizer`
- `augmented_minimizer`
- drive scaling
- state reset / detach / clamp

It should expose:

- `set_drive(h_prev) -> x_k`
- `output_state() -> h_k`
- `output_layer()`
- `free_layers()`
- `resistive_params()`
- `capture_free_state()`
- `restore_free_state(cache)`

I would not make the trainer manipulate raw `energy.free_layers()` directly once the real implementation begins. That is fine for tests, but not for full training.

### B. Block EP Layer

Add a block-local EP helper that mirrors the old DRN estimator.

Suggested object:

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

Suggested dataclasses:

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
```

This layer should:

1. prepare the nudging force from `output_cotangent`
2. restore the free state
3. run `+beta` equilibrium with the block augmented minimizer
4. restore the free state
5. run `-beta` equilibrium with the block augmented minimizer
6. compute centered parameter gradients using the plain energy
7. compute `delta_drive`

The trainer should not know how the block solver works internally.

### C. Digital Backward Layer

The digital side should remain explicit and local.

For the final head:

- use ordinary autograd to get:
  - head parameter gradients
  - `delta_h_K`

For each intermediate digital block:

- recompute `x_k = drive_scale * ff_k(h_{k-1})`
- use a local VJP with `delta_drive_k`

Suggested helper:

```python
def vjp_ff_block(block, h_prev, delta_drive):
    ...
    return ff_param_grads, delta_h_prev
```

This should write gradients into:

- `nn.Parameter.grad` for `ff`
- `nn.Parameter.grad` for `drive_scale_raw` if enabled

The key rule is:

- do not differentiate through the DRN solver
- do not keep one giant graph over the whole free forward
- always recompute the local digital mapping with detached cached inputs

### D. Trainer Layer

Add a new trainer separate from `BPTrainer`.

Suggested class:

```python
class HybridEPTrainer(BPTrainer):
    ...
```

I would reuse as much of `BPTrainer` as possible:

- checkpointing
- history
- logging
- evaluation
- optimizer / scheduler setup
- TensorBoard summaries

But override the training step completely.

## 4. Exact Flow Per Training Step

### Step 1. Free Forward Pass

Run the model forward once and cache per block:

- `h_prev`
- `drive`
- `free_state`
- `output`

For a network with a final head, also cache:

- `h_K`
- `logits`

I would add a dedicated model method:

```python
model.forward_free_with_cache(inputs, num_iterations=...)
```

returning something like:

```python
NetworkFreeCache(
    block_caches=[...],
    final_hidden=h_K,
    logits=...,
)
```

This avoids the trainer reaching into too many model internals.

### Step 2. Final Head Backward

Using the cached final hidden state:

```python
hK = cache.final_hidden.detach().clone().requires_grad_(True)
logits = model.head(hK)
loss = criterion(logits, targets)
delta_h_K, *head_grads = torch.autograd.grad(loss, (hK, *head.parameters()))
```

Write `head_grads` into `param.grad`.

### Step 3. Reverse Sweep Over Blocks

For `k = K, K-1, ..., 1`:

1. call block-local EP:
   - input:
     - free cache for block `k`
     - `delta_h_k`
   - output:
     - DRN parameter grads
     - `delta_drive_k`

2. write DRN grads into `param.state.grad`

3. if the digital frontend before that DRN block is trainable:
   - run local VJP through `F_k`
   - get:
     - FF parameter grads
     - `delta_h_{k-1}`

4. continue to the previous block

### Step 4. Optimizer Step

Exactly like the old DRN flow:

- `optimizer.step()`
- clamp resistive params
- detach states
- scheduler step if configured

## 5. How To Compute `delta_x_k`

This is the main new quantity compared to the original whole-model DRN EP code.

For the current block design, the DRN input coupling is:

```text
E_drive = - <z_1, x_k>
```

where `z_1` is the first analog layer and `x_k` is the injected feedforward current.

Therefore:

```text
∂E / ∂x_k = - z_1
```

and centered EP gives:

```text
delta_x_k = (z_1^- - z_1^+) / (2β)
```

So for dense and conv blocks with this same input-coupling convention, we do not need a new symbolic derivative engine first.

The block EP helper can compute `delta_drive` directly from the first analog layer states of the two nudged equilibria.

This should still be exposed behind a method, for future generality:

```python
block.energy_input_grad_from_state(state, drive)
```

But for v1 it can use the closed-form shortcut.

## 6. How Nudging Should Be Defined

The nudging term should be explicit and linear:

```text
E_nudge = β <output_k(s), delta_h_k>
```

This means:

- only the linear coefficient changes
- no cost-induced quadratic term is added
- the semantics are exactly injected current / fixed force

This matches the recent `Nudging(mode="current")` path already added in:

- [augmented.py](/home/filip/digital_drn/energy/augmented.py)

The main remaining cleanup is conceptual:

- `AugmentedFunction` should be thought of as `energy + output_force`
- not `energy + cost`

That is already the direction of the current implementation.

## 7. Recommended Class Additions

### On blocks

Add these methods to both dense and conv block types:

```python
def capture_free_cache(self, h_prev) -> BlockFreeCache:
    ...

def restore_free_cache(self, cache: BlockFreeCache) -> None:
    ...

def augmented_minimizer(self) -> QuadraticMinimizer:
    ...

def first_free_layer(self):
    ...

def output_from_state(self, state):
    ...
```

Even if `output_from_state(state)` is trivial for now, it should exist. Later, some blocks may expose only a slice of the internal state.

### New helper module

Add a new file, something like:

- `ep_block.py`

holding:

- `BlockFreeCache`
- `BlockEPResult`
- `BlockEquilibriumProp`
- `vjp_ff_block(...)`

This keeps EP logic out of `block.py` and out of `trainer.py`.

### New trainer

Add:

- `ep_trainer.py`

holding:

- `HybridEPTrainer`

and make `experiment.py` build it when:

- `algorithm.name == "ep"`

## 8. Recommended Phases

### Phase 0. Stabilize the current prototype

Already partially done:

- current-force nudging exists
- last-head BP + last-DRN EP test exists
- two-block frozen-prefix overfit test exists

Before full trainer work, keep these as regression tests.

### Phase 1. Formalize block-local EP

Implement:

- `BlockFreeCache`
- `BlockEPResult`
- `BlockEquilibriumProp`

Replace direct test-time manipulation of free layers with this helper.

Success criterion:

- existing single-block and two-block prototype tests still pass

### Phase 2. Last-block trainer path

Implement a minimal `HybridEPTrainer` that trains only:

- final digital head
- final DRN block

with all earlier blocks frozen

Success criterion:

- can overfit the same small toy problem used in the current test
- trainer path works through the normal config / experiment entrypoint

### Phase 3. Add one-step reverse coupling

Implement local VJP through the digital block just before the last DRN block:

- produce FF parameter gradients
- produce `delta_h_{K-1}`

Success criterion:

- a 2-block network can overfit with:
  - last DRN by EP
  - preceding digital FF by explicit VJP

### Phase 4. Full reverse sweep

Generalize the trainer loop over every block:

- `K, K-1, ..., 1`

Success criterion:

- a multi-block dense model can overfit a tiny synthetic dataset
- then a tiny MNIST subset

### Phase 5. Conv / CIFAR path

Once dense full-network EP works:

- extend the same machinery to `ConvDigitalDRNBlock`
- then to mixed `DigitalAnalogNet` models used for CIFAR

## 9. What I Would Not Do Yet

To keep the implementation coherent, I would avoid these in the first full-network EP version:

- analog readout special-casing
- non-centered EP variants
- `second_fn` / alternative estimator path
- end-to-end autograd through solver trajectories
- mixing blockwise EP and implicit backward in the same block
- batchnorm-stat updates during local recomputation

For networks that contain BatchNorm in FF modules, freeze BN running stats during the EP training step.

## 10. Main Code Changes

If implemented cleanly, the file-level change set should look like this:

### New files

- `ep_block.py`
- `ep_trainer.py`
- maybe `ep_types.py` if the cache/result dataclasses become large

### Existing files to extend

- `block.py`
- `conv_block.py`
- `augmented.py`
- `network.py`
- `network_digital_analog.py`
- `experiment.py`
- `tests/...`

### Existing files that should remain mostly untouched

- `core/minimizer.py`
- `core/interaction.py`
- `core/parameter.py`
- `core/layer.py`

The solver and variable wrappers are already the right substrate. The main missing piece is orchestration.

## 11. Recommended First Real Milestone

If I had to pick the next concrete milestone, it would be:

1. implement `BlockEquilibriumProp`
2. implement `HybridEPTrainer` for:
   - head BP
   - last DRN EP
   - earlier blocks frozen
3. verify on a tiny MNIST subset
4. only then add the digital VJP recursion

That gives a real training path quickly, without committing too early to a messy trainer rewrite.

## 12. Bottom Line

The structure should be:

- block-local DRN EP that stays close to the old DRN design
- explicit digital VJP coupling between blocks
- one trainer that orchestrates the reverse sweep

In short:

```text
free forward pass with caches
-> head BP
-> last block EP
-> FF VJP
-> previous block EP
-> FF VJP
-> ...
-> optimizer step
```

This keeps:

- the DRN side energy-based
- the digital side explicit and modular
- the implementation close to the original DRN abstractions
- the coupling across multiple blocks fully explicit
