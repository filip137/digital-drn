# Single FF + Energy Block

This note explains how a single feedforward-plus-energy block works, what the relevant abstraction levels are, and how equilibrium is found.

The discussion uses two closely related viewpoints:

- the `hybrid_bp_ep_official` Hopfield / ff-EBM block formulation
- the DRN resistive-energy formulation in `server_code/model/resistive`

These are not implemented with the same abstractions, but they represent the same high-level idea:

- there is a fixed input or upstream drive
- there is a set of local state variables
- there is a scalar function that couples them
- equilibrium is found by iteratively updating the local state variables

## 1. The Core Idea

A single FF + energy block has:

- an input `x`
- internal state variables `z1, z2, ...`
- parameters `theta`
- a scalar coupling function

The block output is usually the final state, for example `z_last`.

Conceptually:

- the feedforward part supplies a drive from `x`
- the energy part couples that drive to the local states
- equilibrium is the state configuration where the local updates have settled

## 2. Abstraction Levels

It helps to separate the block into layers of abstraction.

### Level A: Model / Chain Level

At the highest level, multiple blocks are composed in a chain.

In `hybrid_bp_ep_official`, the chain forward pass does:

- start with `h = input`
- run one block on `h`
- set `h = block_output`
- pass `h` to the next block

Relevant code:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L60)

So the current block input is:

- the raw image for block 1
- the previous block's final state for later blocks

### Level B: Block Level

At the block level, we have:

- fixed block input `x`
- local states `neurons = [z1, z2, ...]`
- block-specific parameters
- a local relaxation procedure

In `hybrid_bp_ep_official`, this is the `HopfieldBlock` abstraction:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L113)

In DRN, there is no first-class standalone "single block" object in the same sense. The nearest equivalent is:

- a selected subset of layers and interactions inside `DeepResistiveEnergy`
- together with a minimizer acting on the corresponding free layers

### Level C: Scalar Function Level

This is the function that defines the coupling inside the block.

There are two variants:

- BP-EP block formulation:
  - the scalar function is `Phi`
- DRN formulation:
  - the scalar function is the physical energy `E`

In the BP-EP implementation:

- `Phi` is the primitive function of the block
- neuron updates and EP gradients are derived from it

Relevant code:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L269)

In the DRN implementation:

- the total energy is a sum of interaction terms
- per-layer local coefficients are derived from the energy

Relevant code:

- [interaction.py](/home/filip/server_code/model/function/interaction.py#L829)

### Level D: Interaction Level

At this level, the block scalar function is decomposed into pairwise or local interaction terms.

Examples:

- BP-EP:
  - `feedforward(x) * z1`
  - `W(z1) * z2`
  - `W(z2) * z3`
- DRN:
  - dense resistive interaction
  - convolutional resistive interaction
  - bias interaction
  - diode interaction

Relevant code:

- BP-EP VGG primitive:
  [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L351)
- BP-EP ResNet primitive:
  [resnet.py](/home/filip/hybrid_bp_ep_official/src/models/resnet.py#L163)
- DRN dense interaction:
  [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L9)
- DRN convolution interaction:
  [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L160)

### Level E: Local Updater Level

At this level, each free layer gets an update rule.

Examples:

- BP-EP autograd stepper:
  - update from `dPhi/dz`
- BP-EP analytical stepper:
  - update from explicit bottom-up and top-down terms
- DRN quadratic updater:
  - update from local quadratic coefficients `a` and `b`

Relevant code:

- BP-EP fixed-point stepper:
  [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L458)
- BP-EP analytical stepper:
  [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L378)
- DRN updater base:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L6)
- DRN quadratic updater:
  [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L31)

### Level F: Equilibrium Scheduler Level

This is the outer loop that decides:

- how many update sweeps to run
- in what order layers are updated
- whether to stop by fixed iteration count or tolerance

Relevant code:

- BP-EP block relaxation:
  [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L170)
- DRN minimizer equilibrium loop:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L178)

## 3. How a Single FF + Energy Block Works in `hybrid_bp_ep_official`

The BP-EP code treats the feedforward part as absorbed into the block primitive.

### 3.1 Input to the Block

The chain passes `h` into the block as `x`.

Relevant code:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L76)
- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L78)

### 3.2 Feedforward Coupling

For a VGG block, the first coupling term is:

`sum(feedforward(x) * z1)`

and later terms are:

`sum(W1(z1) * z2)`, `sum(W2(z2) * z3)`, and so on.

Relevant code:

- [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L351)
- [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L355)
- [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L358)

For a ResNet block, there is an additional residual coupling term:

- [resnet.py](/home/filip/hybrid_bp_ep_official/src/models/resnet.py#L179)
- [resnet.py](/home/filip/hybrid_bp_ep_official/src/models/resnet.py#L182)

### 3.3 What the Scalar Function Is

The block scalar function is `Phi`.

This `Phi` is used for:

- neuron updates
- EP gradient computation

Relevant code:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L269)
- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L504)

### 3.4 How Equilibrium Is Found

The block forward method does:

1. keep `x` fixed
2. repeatedly call the stepper on the neuron list
3. optionally stop on tolerance
4. return the relaxed neuron list

Relevant code:

- [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L170)

The two stepper variants are:

- autograd:
  - compute `dPhi/dz`
  - use those derivatives as the updates
- analytical:
  - compute the same update structure explicitly

Relevant code:

- autograd stepper:
  [base.py](/home/filip/hybrid_bp_ep_official/src/models/base.py#L458)
- analytical stepper:
  [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L396)

For the first local state, the bottom-up term is literally the feedforward transform:

- [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L495)
- [vgg.py](/home/filip/hybrid_bp_ep_official/src/models/vgg.py#L501)

## 4. How a Single FF + Energy Block Works in DRN

The DRN implementation uses a different abstraction.

There is usually not an explicit block object with:

- `x`
- `Phi`
- `stepper`

Instead, the equivalent block behavior is distributed across:

- a subset of layers inside `DeepResistiveEnergy`
- the interactions coupling those layers
- a minimizer that updates the free layers

### 4.1 What Plays the Role of the Feedforward Part

There is no separate feedforward module in the same way as BP-EP.

Instead:

- the input or upstream state is written into a fixed boundary layer
- the first free layer is coupled to that fixed source by resistive interactions

For example, `ResistiveInputLayer.set_input()` writes the source state:

- [layer.py](/home/filip/server_code/model/resistive/layer.py#L31)

And the first hidden layer is coupled to the input through interactions such as `DenseResistive` or `ConvResistive`:

- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L9)
- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L160)

So the "feedforward drive" is represented by the interaction between:

- a fixed pre-synaptic source state
- a free post-synaptic state

### 4.2 What the Scalar Function Is

The scalar function is the physical energy `E`, implemented as a sum of interactions.

Relevant code:

- [network.py](/home/filip/server_code/model/resistive/network.py#L25)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L849)

### 4.3 How the Local Updaters See the Energy

The minimizer does not directly inspect every interaction.

Instead, for each free layer it asks the function for:

- `a_coef_fn(layer)`
- `b_coef_fn(layer)`

Those functions are sums of the corresponding interaction-level contributions.

Relevant code:

- [interaction.py](/home/filip/server_code/model/function/interaction.py#L907)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L924)

### 4.4 How Equilibrium Is Found

The flow is:

1. choose the function to minimize
   - either `energy_fn`
   - or `AugmentedFunction(energy_fn, cost_fn)` in EP
2. construct `QuadraticMinimizer(fn, free_layers, ...)`
3. create one updater per free layer
4. iterate the update schedule
5. at each local update:
   - compute `a` and `b`
   - compute the minimizer for that layer
   - assign the new state
   - apply the layer activation / clamp
6. repeat until the chosen number of sweeps is reached

Relevant code:

- minimizer creation in training:
  [drn_config.py](/home/filip/server_code/papers/fast-drn/training/drn_config.py#L233)
- equilibrium loop:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L178)
- one update step:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L220)
- updater selection:
  [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L873)

## 5. The Simplest Local Equilibrium Update

In the quadratic case, the updater assumes the energy is locally:

`E(z) = a z^2 + b z + c`

for that layer coordinate, so the local minimizer is:

`z* = -b / (2a)`

Relevant code:

- [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L73)

After that, the layer activation is applied:

- perfect-diode style layers clamp excitatory and inhibitory halves
- other nonlinearities may leave the value unclipped and encode the nonlinearity in `pre_activate()`

Relevant code:

- [layer.py](/home/filip/server_code/model/resistive/layer.py#L70)

## 6. Equilibrium in One Sentence

For a single FF + energy block, equilibrium is found by:

- keeping the incoming drive fixed
- repeatedly updating the free internal states according to the scalar coupling function
- stopping after the scheduled sweeps or a convergence test

In BP-EP, the scalar function is usually `Phi`.  
In DRN, it is the resistive energy `E`.

## 7. Practical Comparison

### BP-EP block

- first-class block object
- explicit `Phi`
- explicit `stepper`
- chain passes `x` from block to block

### DRN block-equivalent

- no first-class standalone block abstraction
- scalar function is the energy `E`
- equilibrium is owned by the minimizer
- "feedforward" behavior is represented by fixed source layers coupled through interactions

## 8. Why This Matters for Integration

If DRN is inserted into a FF + energy block framework, there are two possible interpretations:

- keep the BP-EP abstraction
  - derive a `Phi`-compatible block formulation
- keep the DRN abstraction
  - use energy `E` plus a minimizer-driven equilibrium loop

The second option is deeper, because the block no longer centers around `Phi` and a fixed-point stepper. It centers around:

- the energy definition
- the free-layer set
- the per-layer updaters
- the equilibrium scheduler

## 9. Proposed `DigitalDRNBlock` Structure

For the `DigitalDRNBlock` design discussed here, the coupling should not use the standard DRN input-layer pattern.

The intended behavior is:

- the digital / FF part computes an injected current
- that current is added as a forcing term on the first free DRN layer
- the first free DRN layer is still free and is decided by equilibrium

So the block should be modeled as:

`E_block(z; theta, i_ff) = E_internal(z; theta) + E_inject(i_ff, z1)`

where:

- `z = [z1, z2, ...]` are the free DRN states
- `theta` are the trainable resistive parameters
- `i_ff` is the current produced by the digital / FF path

The natural linear injection term is:

`E_inject(i_ff, z1) = - <i_ff, z1>`

This means:

- the FF block does not directly overwrite `z1`
- the FF block supplies a current term
- the minimizer determines `z1` from the total energy

### 9.1 Recommended Object Split

The clean split is:

- `DigitalDRNNet(nn.Module)`
  - top-level chained model
  - owns a `ModuleList` of blocks
- `DigitalDRNBlock(nn.Module)`
  - block wrapper used by the model
  - owns the FF map, block energy, and minimizer(s)
- `DenseDRNBlockEnergy(SumSeparableFunction)`
  - block-local energy object
  - owns free layers, trainable parameters, and interactions
- `FFCurrentInteraction(LFunction)`
  - non-trainable interaction that injects the FF current into the first free layer

This keeps the responsibilities separate:

- `nn.Module` wrapper:
  integration with PyTorch and block chaining
- energy object:
  defines the scalar function being minimized
- minimizer:
  computes equilibrium
- current interaction:
  couples FF output to the first free DRN state

### 9.2 Recommended File Structure

One reasonable layout is:

```text
digital_drn/
  block.py
  block_energy.py
  block_interactions.py
  network.py
```

with responsibilities:

- `block.py`
  - `DigitalDRNBlock(nn.Module)`
- `block_energy.py`
  - `DenseDRNBlockEnergy(SumSeparableFunction)`
- `block_interactions.py`
  - `FFCurrentInteraction(LFunction)`
- `network.py`
  - `DigitalDRNNet(nn.Module)`

### 9.3 `DigitalDRNBlock(nn.Module)`

This should be the main PyTorch-facing block abstraction.

It should contain:

- `self.ff`
  - an ordinary digital / feedforward `nn.Module`
  - maps incoming block input `h` to an injected current `i_ff`
- `self.energy`
  - a `DenseDRNBlockEnergy`
- `self.minimizer_inference`
  - equilibrium solver for inference / free phase
- `self.minimizer_training`
  - optional separate equilibrium solver for training / BP

Conceptually:

```python
class DigitalDRNBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.ff = ...
        self.energy = DenseDRNBlockEnergy(...)
        self.minimizer_inference = QuadraticMinimizer(
            fn=self.energy,
            free_layers=self.energy.free_layers(),
            ...
        )
        self.minimizer_training = QuadraticMinimizer(
            fn=self.energy,
            free_layers=self.energy.free_layers(),
            ...
        )

    def set_drive(self, h):
        i_ff = self.ff(h)
        self.energy.set_drive(i_ff)

    def reset_state(self, batch_size, device):
        self.energy.reset_free_layers(batch_size, device)

    def equilibrate(self, training=False):
        minimizer = self.minimizer_training if training else self.minimizer_inference
        minimizer.compute_equilibrium()

    def forward(self, h, reset=False, training_dynamics=False):
        if reset:
            self.reset_state(h.size(0), h.device)
        self.set_drive(h)
        self.equilibrate(training=training_dynamics)
        return self.energy.output_state()
```

### 9.4 `DenseDRNBlockEnergy`

This object should not own the training logic.

It should only define:

- the free layers `z1, z2, ...`
- the resistive weights and biases
- the list of internal interactions
- the `FFCurrentInteraction`
- helper accessors for:
  - `free_layers()`
  - `set_drive(i_ff)`
  - `reset_free_layers(batch_size, device)`
  - `output_state()`

Conceptually:

```python
class DenseDRNBlockEnergy(SumSeparableFunction):
    def __init__(self, ...):
        self.z1 = NonlinearResistiveLayer(...)
        self.z2 = NonlinearResistiveLayer(...)
        self.z_out = LinearLayer(...)

        self.drive = FFCurrentInteraction(self.z1)

        self.weights = ...
        self.biases = ...

        interactions = [
            self.drive,
            ... BiasInteraction(...),
            ... DenseResistive(...),
            ... diode interactions ...
        ]

        layers = [self.z1, self.z2, self.z_out]
        params = [...]

        super().__init__(layers, params, interactions)

    def free_layers(self):
        return self.layers()

    def set_drive(self, current):
        self.drive.set_current(current)

    def reset_free_layers(self, batch_size, device):
        for layer in self.layers():
            layer.init_state(batch_size, device)

    def output_state(self):
        return self.layers()[-1].state
```

### 9.5 `FFCurrentInteraction`

This interaction should be linear in the first free layer, so it should follow the `LFunction` abstraction:

- no trainable parameters
- one target layer: `z1`
- stores the current tensor `i_ff`
- contributes:
  - energy term `- <i_ff, z1>`
  - layer gradient / `b` coefficient `-i_ff`

Conceptually:

```python
class FFCurrentInteraction(LFunction):
    def __init__(self, layer):
        self._layer = layer
        self._current = None
        super().__init__([layer], [])

    def set_current(self, current):
        self._current = current

    def eval(self):
        return -self._layer.state.mul(self._current).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise KeyError(layer)
        return lambda: -self._current
```

Because `LFunction.b_coef_fn(layer)` is defined from `grad_layer_fn(layer)`, this interaction automatically injects the FF current into the first layer's local linear coefficient.

Relevant code:

- `LFunction`:
  [interaction.py](/home/filip/server_code/model/function/interaction.py#L315)
- `SumSeparableFunction.a_coef_fn` / `b_coef_fn`:
  [interaction.py](/home/filip/server_code/model/function/interaction.py#L907)
  [interaction.py](/home/filip/server_code/model/function/interaction.py#L924)

### 9.6 Where the Feedforward Network Lives

The digital / FF network should live inside the block wrapper as a normal `nn.Module`.

That means:

- the energy object should not define the digital map
- the minimizer should not define the digital map
- the outer block should compute the current drive and then launch the equilibrium solver

So the correct mental model is:

- `ff(h)` computes `i_ff`
- `energy.set_drive(i_ff)` injects that current into the first free DRN layer
- `QuadraticMinimizer.compute_equilibrium()` solves for the block state
- the final layer state is the block output

### 9.7 Where the Minimizer Lives

The minimizer should live inside `DigitalDRNBlock`, not inside `DenseDRNBlockEnergy`.

Reason:

- the energy object defines what is minimized
- the minimizer defines how equilibrium is computed
- the block wrapper decides when to reset, when to run inference dynamics, and when to run training dynamics

So:

- `DenseDRNBlockEnergy`
  - energy definition only
- `DigitalDRNBlock`
  - equilibrium logic

### 9.8 Top-Level Network Structure

The full digital resistive model can then be written as a chain of such blocks:

```python
class DigitalDRNNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h
```

So the FF network is naturally distributed across the blocks:

- each block has its own digital projection
- each block injects current into its own DRN energy
- each block outputs its relaxed final state to the next block

### 9.9 Main Engineering Rule

For this design, do not try to force the block through the existing DRN `Network.set_input()` abstraction.

That abstraction assumes:

- a special input layer
- direct writing of a boundary state

But the proposed `DigitalDRNBlock` needs:

- injected current from a digital map
- no direct overwrite of the first free DRN state
- equilibrium determined by the energy plus the forcing term

So the coupling point should be:

- a dedicated interaction like `FFCurrentInteraction`

not:

- direct state assignment into the first free layer
