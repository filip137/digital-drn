# Explicit Gradient Coupling Theory

This note summarizes the theory for hybrid BP-EP training in an alternating digital/DRN network.

The key idea is:

- each DRN block is trained with centered EP on a block-local augmented energy
- the coupling between DRN blocks is handled explicitly by backpropagating a cotangent through the digital feedforward block between them

This is the design target for `digital_drn`.

## Default Update Mode

For `digital_drn`, the default DRN minimizer/update mode should always be:

```text
asynchronous
```

That applies to:

- free-phase equilibria
- nudged equilibria used for EP
- ordinary model-building defaults unless a test or experiment explicitly overrides them

So the theory and the implementation should both assume asynchronous updates as the baseline operating mode.

## 1. Model Structure

Assume the model is an alternating chain:

```text
digital block F_1
DRN block E_1
digital block F_2
DRN block E_2
...
digital readout F_N
```

For block `k`:

- `h_{k-1}` = exposed output of the previous block
- `x_k = F_k(h_{k-1}; ω_k)` = digital feedforward drive into the DRN block
- `s_k* = argmin_s E_k(s, θ_k, x_k)` = free equilibrium of DRN block `k`
- `h_k = output_k(s_k*)` = exposed DRN output passed to the next digital block or head

The final readout computes:

```text
logits = F_N(h_K; ω_N)
loss = ℓ(logits, y)
```

## 2. Final Head Gradient

At the output, ordinary backpropagation is used through the digital head:

```text
delta_h_K = ∇_{h_K} ℓ(logits, y)
g_{ω_N}   = ∇_{ω_N} ℓ(logits, y)
```

So the last DRN block receives a fixed free-phase output cotangent `delta_h_K`.

## 3. Block-Local Augmented Energy

For DRN block `k`, the nudged phases are defined by a block-local augmented energy:

```text
E_k(s, θ_k, x_k*) + β <output_k(s), delta_h_k>
```

where:

- `x_k*` is the fixed free-phase drive of the block
- `delta_h_k` is the output cotangent used to nudge the block
- `β` is the centered EP strength

This matches the old DRN design principle:

```text
augmented_energy = energy + nudging_interaction
```

but the nudging interaction is now driven by an externally supplied output cotangent rather than a local cost function.

## 4. Centered EP On One DRN Block

For block `k`, define the centered nudged equilibria:

```text
s_k^{+} = argmin_s [ E_k(s, θ_k, x_k*) + β <output_k(s), delta_h_k> ]
s_k^{-} = argmin_s [ E_k(s, θ_k, x_k*) - β <output_k(s), delta_h_k> ]
```

Then the DRN parameter gradient is estimated by:

```text
g_{θ_k}
=
[ ∂E_k/∂θ_k (s_k^{+}, x_k*) - ∂E_k/∂θ_k (s_k^{-}, x_k*) ] / (2β)
```

The block also returns a backward signal with respect to its injected drive:

```text
Δx_k
=
[ ∂E_k/∂x_k (s_k^{+}, x_k*) - ∂E_k/∂x_k (s_k^{-}, x_k*) ] / (2β)
```

This `Δx_k` is the cotangent that must be propagated through the preceding digital block.

## 5. What `Δx_k` Means

`Δx_k` is not the nudging signal for the previous block directly.

It is the EP-estimated cotangent with respect to the drive input `x_k` of DRN block `k`.

Only after passing it backward through the digital block `F_k` do we obtain the nudging signal for the previous DRN block.

This distinction matters:

- `Δx_k` lives at the input of DRN block `k`
- `delta_h_{k-1}` lives at the output of DRN block `k-1`

## 6. Explicit BP Through The Digital Block

The digital coupling is:

```text
x_k = F_k(h_{k-1}; ω_k)
```

Given `Δx_k`, the explicit backward pass through `F_k` is:

```text
g_{ω_k}     = (∂F_k/∂ω_k)^T Δx_k
delta_h_{k-1} = (∂F_k/∂h_{k-1})^T Δx_k
```

Equivalently, in Jacobian notation:

```text
g_{ω_k}   = J_{ω_k} F_k(h_{k-1})^T Δx_k
delta_h_{k-1} = J_{h_{k-1}} F_k(h_{k-1})^T Δx_k
```

So yes: the nudging signal for the previous block is obtained by multiplying `Δx_k` by the transpose Jacobian of the digital feedforward map with respect to the previous DRN output.

This is the vector-Jacobian product:

```text
delta_h_{k-1} = VJP(F_k, h_{k-1}, Δx_k)
```

## 7. Reverse Sweep Across The Whole Network

The full reverse algorithm is:

```text
delta_h_K = ∇_{h_K} ℓ(logits, y)
g_{ω_N}   = ∇_{ω_N} ℓ(logits, y)

for k = K, K-1, ..., 1:
    s_k^{+} = argmin_s [ E_k(s, θ_k, x_k*) + β <output_k(s), delta_h_k> ]
    s_k^{-} = argmin_s [ E_k(s, θ_k, x_k*) - β <output_k(s), delta_h_k> ]

    g_{θ_k} =
        [ ∂E_k/∂θ_k (s_k^{+}, x_k*) - ∂E_k/∂θ_k (s_k^{-}, x_k*) ] / (2β)

    Δx_k =
        [ ∂E_k/∂x_k (s_k^{+}, x_k*) - ∂E_k/∂x_k (s_k^{-}, x_k*) ] / (2β)

    g_{ω_k}, delta_h_{k-1} =
        VJP(F_k, inputs=(h_{k-1}, ω_k), cotangent=Δx_k)
```

So:

- EP provides `g_{θ_k}` and `Δx_k`
- explicit digital BP provides `g_{ω_k}` and `delta_h_{k-1}`

## 8. Common Linear-Coupling Shortcut

In many DRN blocks, the energy has the standard input coupling:

```text
E_k(s, θ, x) = Ē_k(s, θ) - <z_{1,k}, x>
```

where `z_{1,k}` is the first free analog layer.

Then:

```text
∂E_k / ∂x = -z_{1,k}
```

and therefore:

```text
Δx_k = (z_{1,k}^{-} - z_{1,k}^{+}) / (2β)
```

So in the linear-coupling case, `Δx_k` can be computed directly from the two nudged first-layer states without introducing a separate symbolic derivative path.

## 9. Last Block Vs Earlier Blocks

The DRN blocks themselves do not need different nudging semantics for the last block and earlier blocks.

They all use:

```text
E_k(s, θ_k, x_k*) + β <output_k(s), delta_h_k>
```

The only difference is where `delta_h_k` comes from:

- last block:
  - `delta_h_K = ∇_{h_K} ℓ(logits, y)`
- earlier blocks:
  - `delta_h_k = J_{h_k} F_{k+1}(h_k)^T Δx_{k+1}`

So the trainer computes the source of the signal, but the DRN block only receives a fixed output cotangent to nudge with.

## 10. Mapping To `digital_drn`

The natural mapping in the current codebase is:

- runtime block:
  - [DigitalDRNBlock](/home/filip/digital_drn/blocks/base.py)
- block energy:
  - [DenseDRNBlockEnergy](/home/filip/digital_drn/energy/block_energy.py)
  - [ConvDRNBlockEnergy](/home/filip/digital_drn/energy/block_energy.py)
  - [ConvDenseDRNBlockEnergy](/home/filip/digital_drn/energy/block_energy.py)
- block-local augmented function:
  - [AugmentedFunction](/home/filip/digital_drn/energy/augmented.py)
- block-local nudging interaction:
  - [Nudging](/home/filip/digital_drn/energy/augmented.py)
- top-level container:
  - [DigitalDRNNet](/home/filip/digital_drn/models/network.py)
  - [DigitalAnalogNet](/home/filip/digital_drn/models/network_digital_analog.py)

This means the intended implementation is:

1. free forward through the model with block-local caches
2. head gradient by ordinary autograd
3. reverse sweep over blocks:
   - centered EP on the DRN block
   - VJP through the digital feedforward block

## 11. Implementation Consequence

The correct implementation target is not:

- one global cost-based augmented function over the whole network

but rather:

- one augmented function per DRN block
- each nudged by a fixed output cotangent supplied by the trainer

So the trainer owns:

- the global loss
- the digital VJPs
- the reverse block ordering

And each DRN block owns:

- its local free equilibrium
- its local nudged equilibria
- its centered EP gradient estimate

That is the clean hybrid BP-EP decomposition for `digital_drn`.
