# Digital DRN Coupling And Gradient Flow For Attention Design

This note is meant as a handoff for designing multihead attention inside the current `digital_drn` setup.

The key point is:

- a `DigitalDRNBlock` is not a mixed analog-digital blob
- it is a very specific split:
  - a digital frontend `ff`
  - a positive scalar `drive_scale`
  - a block-local DRN energy
  - a relaxation solver

So the present architecture is:

- digital module computes a drive
- drive is injected as current into the first free DRN layer
- the DRN block relaxes to equilibrium
- the block output is the final free layer state

This is the interface that any attention mechanism must fit.

## 1. What A Block Does Today

The runtime wrapper is [`DigitalDRNBlock`](/home/filip/digital_drn/blocks/base.py).

Each block owns:

- `ff`: an ordinary differentiable digital module
- `_drive_scale_raw`: a learned scalar parameter
- `energy`: a block-local DRN energy
- `inference_minimizer`
- `training_minimizer`

The actual injected drive is:

```python
drive = block.drive_scale * block.ff(h_prev)
```

where:

- `drive_scale = softplus(_drive_scale_raw)`
- `h_prev` is the exposed output of the previous block

Relevant code:

- [`DigitalDRNBlock.set_drive(...)`](/home/filip/digital_drn/blocks/base.py#L79)
- [`DigitalDRNBlock.forward(...)`](/home/filip/digital_drn/blocks/base.py#L205)

So the digital block does not directly set analog states.
It only produces a current-like forcing tensor.

## 2. How The FF Block Acts On The DRN Block

The coupling term is [`FFCurrentInteraction`](/home/filip/digital_drn/energy/block_interactions.py).

Its energy contribution is:

```python
E_drive(z1, x) = - <z1, x>
```

where:

- `z1` is the first free DRN layer
- `x` is the injected drive tensor from `drive_scale * ff(h_prev)`

Relevant code:

- [`FFCurrentInteraction.eval(...)`](/home/filip/digital_drn/energy/block_interactions.py#L19)
- [`FFCurrentInteraction._grad_layer(...)`](/home/filip/digital_drn/energy/block_interactions.py#L29)

This means:

- the drive acts only on the first free analog layer
- its layer gradient is:
  - `dE_drive/dz1 = -x`
- so in equilibrium, the first free layer feels a direct external current equal to the FF output

That is the present meaning of “the FF block acts on the DRN block with currents.”

There is no direct digital coupling into deeper free layers.
All deeper influence happens indirectly through the DRN energy and relaxation.

## 3. What Lives Inside The Block Energy

For dense blocks, the energy is [`DenseDRNBlockEnergy`](/home/filip/digital_drn/energy/block_energy.py#L191).

It contains:

- the drive interaction on the first free layer
- bias interactions on nonlinear free layers
- dense resistive interactions between adjacent free layers
- optional local nonlinear interactions

So a dense block is:

- `drive -> first free layer`
- then resistive pairwise couplings propagate that effect through the block

The output exposed to the next block is:

- the final free layer state
- [`BaseBlockEnergy.output_state(...)`](/home/filip/digital_drn/energy/block_energy.py#L170)

Therefore the inter-block chain is:

```text
h_{k-1}
  -> ff_k(h_{k-1})
  -> drive current x_k
  -> injected into first free layer z_{k,1}
  -> block equilibrium
  -> exposed output h_k = z_{k,last}
```

## 4. What The Full Network Coupling Is

At the network level, [`DigitalAnalogNet`](/home/filip/digital_drn/models/network_digital_analog.py) is just a chain of these blocks plus an optional digital head.

Forward pass:

```text
h0 = input
h1 = block1(h0)
h2 = block2(h1)
...
logits = head(hK)
```

Important consequence:

- DRN blocks do not couple to each other analog-to-analog
- coupling between blocks is always mediated by a digital map `ff_k`
- then converted into a current injection for the next block

For attention design, this is the main architectural fact.

## 5. Two Backward Paths Exist Today

There are two different training/analysis paths.

### 5.1 Ordinary BP

If you run:

```python
logits = model(inputs, reset=True, num_iterations=...)
loss = criterion(logits, targets)
loss.backward()
```

then autograd differentiates through:

- the digital frontends
- the learned drive scales
- the unrolled DRN equilibrium iterations
- the head

This is ordinary BPTT through the finite relaxation rollout.

Where gradients land:

- FF parameters:
  - normal `nn.Parameter.grad`
- drive scale:
  - `block._drive_scale_raw.grad`
- DRN parameters:
  - `param.state.grad`

The grouping helper in [`loaded_model_beta_sweep.py`](/home/filip/digital_drn/training/loaded_model_beta_sweep.py) uses exactly this convention.

### 5.2 Hybrid Explicit EP

The explicit hybrid algorithm is in [`hybrid_backward_explicit(...)`](/home/filip/digital_drn/training/ep_network.py#L113).

It is not “global EP over the whole network.”
It is:

- ordinary autograd through the digital head
- reverse sweep over blocks
- EP inside each DRN block
- explicit VJP through each digital frontend

So the backward split is:

- DRN parameters: from EP
- digital coupling: from ordinary VJP

## 6. Exact Hybrid Backward Flow

### Step A: Free Forward With Cache

[`forward_free_with_cache(...)`](/home/filip/digital_drn/training/ep_network.py#L60) runs the free forward pass and stores, per block:

- `h_prev`
- `drive`
- all free analog states
- block output

This cache matters because block-local EP keeps the free drive fixed.

### Step B: Backprop Through The Final Digital Head

[`backward_head(...)`](/home/filip/digital_drn/training/ep_network.py#L84) computes:

- head parameter gradients
- `delta_h_K = dL/dh_K`

So the last DRN block receives a fixed output cotangent.

### Step C: EP Inside One DRN Block

[`BlockEquilibriumProp.compute_gradients(...)`](/home/filip/digital_drn/training/ep_block.py#L90) does:

1. prepare current-force nudging on the block output layer
2. restore the free analog state and free drive
3. run `+beta` nudged equilibrium
4. restore the free analog state and free drive
5. run `-beta` nudged equilibrium
6. compute centered parameter gradients
7. compute a drive cotangent `delta_drive`

Important current implementation detail:

- only current-force nudging exists in `digital_drn`
- see [`Nudging`](/home/filip/digital_drn/energy/augmented.py#L10)

The nudging energy is:

```text
E_aug = E_block - beta <z_out, force>
```

and `force` is set to `-output_gradient`, so effectively the block is nudged by the externally supplied output cotangent.

### Step D: What EP Returns

For block `k`, EP returns:

- `param_grads`
- `delta_drive`
- plus/minus states
- plus/minus outputs

The DRN parameter gradient is the centered energy-gradient difference:

```text
g_theta = [dE/dtheta(s+) - dE/dtheta(s-)] / (2 beta)
```

Relevant code:

- [`BlockEquilibriumProp._energy_param_grads(...)`](/home/filip/digital_drn/training/ep_block.py#L39)
- [`BlockEquilibriumProp.compute_gradients(...)`](/home/filip/digital_drn/training/ep_block.py#L113)

### Step E: Why `delta_drive` Is Simple

Because the drive coupling is linear:

```text
E_drive(z1, x) = -<z1, x>
```

we have:

```text
dE/dx = -z1
```

So the centered cotangent wrt the injected drive is:

```text
delta_drive = (z1^- - z1^+) / (2 beta)
```

Relevant code:

- [`BlockEquilibriumProp.compute_gradients(...)`](/home/filip/digital_drn/training/ep_block.py#L121)

This is extremely important for future attention:

- the analog block only has to return a cotangent wrt its injected drive
- the upstream digital module can then be handled entirely by standard VJP

### Step F: VJP Through The Digital Frontend

[`vjp_ff_block(...)`](/home/filip/digital_drn/training/ep_digital.py#L37) computes the vector-Jacobian product through:

```text
x_k = drive_scale_k * ff_k(h_{k-1})
```

given the cotangent `delta_drive`.

It returns:

- FF parameter gradients
- drive-scale gradient
- `delta_h_prev`

where:

```text
delta_h_prev = J_{h_prev} (drive_scale * ff(h_prev))^T delta_drive
```

This `delta_h_prev` becomes the nudging cotangent for the previous DRN block.

So the whole reverse sweep is:

```text
delta_h_K from head BP
for k = K ... 1:
    EP on DRN block k -> (g_theta_k, delta_drive_k)
    VJP through digital ff_k -> (g_ff_k, g_scale_k, delta_h_{k-1})
```

## 7. What Is Actually Verified In Tests

The key network-level test is:

- [`tests/test_ep_network.py`](/home/filip/digital_drn/tests/test_ep_network.py)

What it verifies:

- head gradients from `hybrid_backward_explicit(...)` match ordinary BP
- FF parameter gradients match ordinary BP
- drive-scale gradients match ordinary BP

This was checked in:

- ordinary dense drive mode
- signed-drive mode

So the digital coupling path is not just a design sketch. It is tested.

The signed-drive wrapper is in:

- [`MirrorSignedDriveFrontend`](/home/filip/digital_drn/frontends.py#L53)

and maps:

```text
u -> [u, -u]
```

This is useful when the first free analog layer is organized into excitatory/inhibitory halves and the digital module should emit a paired signed current.

## 8. What This Means For Multihead Attention

The safest interpretation is:

- multihead attention should live on the digital side
- it should replace or extend `ff`
- it should still output a drive tensor with the shape expected by the first free analog layer

That means the natural interface is:

```text
h_prev
  -> attention-based digital module
  -> projected drive x_k
  -> current injection into first free DRN layer
```

So attention is not “inside the analog energy” by default.
It is part of the digital coupling map that produces currents for the DRN block.

## 9. The Cleanest Attention Design Target

To stay maximally compatible with the current implementation, an attention frontend should satisfy:

1. It is an ordinary differentiable PyTorch module.
2. It consumes the previous block output `h_prev`.
3. It outputs a tensor with the exact shape required by the first free layer of the target DRN block.
4. Its parameters are trained by the digital VJP path.
5. The DRN block continues to treat that output only as an injected current.

In other words, the target abstraction is:

```text
drive_k = AttentionFrontend_k(h_{k-1})
E_k(..., drive_k) = E_internal_k(...) - <z_{k,1}, drive_k>
```

This preserves the whole existing hybrid algorithm.

## 10. Consequences For Possible MHA Variants

### Variant A: Attention As A Drop-In FF Replacement

This is the most natural option.

Replace:

```text
ff_k : h_{k-1} -> drive_k
```

with:

```text
ff_k = MHA + projection-to-drive
```

Then nothing in the DRN code has to change.

### Variant B: Attention Over Tokenized Features, Then Current Projection

If `h_prev` should be interpreted as a sequence or spatial token set, the attention module can:

- tokenize / reshape `h_prev`
- run Q/K/V attention digitally
- project the attended features into the first free-layer shape
- hand that result to the DRN as the injected current

Again, this still fits the existing interface.

### Variant C: Attention Inside The Analog Energy

This does **not** fit the current setup directly.

To do that, attention would need to be represented as a new analog energy interaction, not as a digital frontend.

That would require:

- a new energy term
- new local coefficients or generic autograd fallback for the solver
- a new EP derivative path inside the block

That is much more invasive.

So unless there is a strong reason to make attention itself analog, the best current design is to keep attention in `ff`.

## 11. Recommended Design Constraints For ChatGPT Pro

If ChatGPT Pro is asked to design multihead attention for `digital_drn`, it should assume:

- DRN blocks are equilibrium modules with fixed external current injection
- the digital coupling module is free to be arbitrarily expressive, as long as it outputs a drive tensor
- the analog side currently only knows how to consume that drive as a linear forcing term on the first free layer
- backward propagation should stay split as:
  - EP for DRN parameters
  - autograd VJP for digital attention / frontend parameters

The question to solve is therefore:

```text
How should an attention module map h_prev to a drive tensor that is useful for the next DRN block?
```

not:

```text
How do we make the DRN solver itself perform multihead attention?
```

## 12. Short Practical Summary

- Current inter-block coupling is digital-to-current, not analog-to-analog.
- `ff` acts on a DRN block only by generating an injected current for the first free layer.
- The analog block returns:
  - its exposed output `h_k`
  - and, under EP, a cotangent `delta_drive_k` wrt that injected current.
- The digital module then uses an ordinary VJP to:
  - update its own parameters
  - propagate the cotangent to the previous block output.

That makes multihead attention feasible right now as a digital coupling module that emits block drive currents.
