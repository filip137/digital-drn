# DRN Structure

This note summarizes how the DRN stack is organized, with emphasis on the flow from `DeepResistiveEnergy` to `QuadraticMinimizer`.

## High-Level Objects

- `DeepResistiveEnergy`
  - The high-level energy-function object for a deep resistive network.
  - It builds:
    - layers
    - parameters
    - interactions
  - File: [network.py](/home/filip/server_code/model/resistive/network.py)

- `Network`
  - A thin wrapper around the energy function.
  - It identifies the input layer and the free layers.
  - File: [network.py](/home/filip/server_code/model/function/network.py)

- `QuadraticMinimizer`
  - The object that computes equilibria.
  - It does not live inside `DeepResistiveEnergy`; it is constructed from a function and a list of free layers.
  - File: [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py)

## What `DeepResistiveEnergy` Contains

`DeepResistiveEnergy` is a `SumSeparableFunction`. It defines the scalar function being minimized.

At construction time it creates:

- the input layer
- optional convolution / pooling layers
- hidden nonlinear resistive layers
- the output layer
- conductance weights and biases
- interaction terms such as:
  - `DenseResistive`
  - `ConvResistive`
  - pooling interactions
  - nonlinear diode interactions

Then it passes all of this to `SumSeparableFunction`.

Relevant code:

- layer / parameter / interaction construction:
  [network.py](/home/filip/server_code/model/resistive/network.py#L31)
- conversion into a sum-separable function:
  [network.py](/home/filip/server_code/model/resistive/network.py#L261)

## What `SumSeparableFunction` Provides

`SumSeparableFunction` is the bridge between the energy definition and the minimizer.

It provides:

- `eval()`
  - returns the total energy as the sum of all interaction energies
- `grad_layer_fn(layer)`
  - sums layer-gradient contributions from all interactions touching that layer
- `grad_param_fn(param)`
  - sums parameter-gradient contributions from all interactions touching that parameter
- `a_coef_fn(layer)`
  - sums all quadratic coefficients `a` for that layer
- `b_coef_fn(layer)`
  - sums all linear coefficients `b` for that layer

Relevant code:

- [interaction.py](/home/filip/server_code/model/function/interaction.py#L829)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L849)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L858)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L876)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L907)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L924)

## What the Individual Resistive Interactions Provide

Each interaction contributes to the local update rules.

For example, `DenseResistive` provides:

- `eval()`
- `a_coef_fn(layer)`
- `b_coef_fn(layer)`
- `grad_param_fn(param)`

So each layer's local quadratic approximation is assembled by summing the relevant interaction contributions.

Relevant code:

- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L9)
- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L37)
- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L58)
- [interaction.py](/home/filip/server_code/model/resistive/interaction.py#L66)

## Role of `Network`

`Network(energy_fn)` does not define the energy and does not compute equilibrium.

It only:

- stores the wrapped function
- exposes the full layer list
- marks the input layer
- defines `free_layers = all layers except the input layer`
- sets the input tensor into the input layer

Relevant code:

- [network.py](/home/filip/server_code/model/function/network.py#L22)
- [network.py](/home/filip/server_code/model/function/network.py#L36)
- [network.py](/home/filip/server_code/model/function/network.py#L44)

## Flow from `DeepResistiveEnergy` to `QuadraticMinimizer`

### 1. Build the energy function

Create:

- `energy_fn = DeepResistiveEnergy(...)`

This defines the layers, params, and interactions.

### 2. Wrap it as a network

Create:

- `network = Network(energy_fn)`

This gives access to:

- `params = energy_fn.params()`
- `layers = energy_fn.layers()`
- `free_layers = network.free_layers()`

Example usage:

- [drn_config.py](/home/filip/server_code/papers/fast-drn/training/drn_config.py#L224)
- [drn_config.py](/home/filip/server_code/papers/fast-drn/training/drn_config.py#L227)

### 3. Choose the function minimized during training

There are two common cases:

- EP:
  - minimize the augmented function `E + beta * C`
  - this is built as `AugmentedFunction(energy_fn, cost_fn)`
- BP:
  - use plain `energy_fn`

Example:

- EP path:
  [drn_config.py](/home/filip/server_code/papers/fast-drn/training/drn_config.py#L233)
- BP path:
  [drn_config.py](/home/filip/server_code/papers/fast-drn/training/drn_config.py#L250)

### 4. Construct the minimizer

Create:

- `QuadraticMinimizer(fn=..., free_layers=..., num_iterations=..., mode=..., non_linearity=...)`

`QuadraticMinimizer` chooses one updater per free layer based on the configured nonlinearity:

- `QuadraticUpdater`
- `AdaptiveQuadraticUpdater`
- `QuadraticDoubleDiodeUpdaterOffset`
- `ExponentialDoubleDiodeUpdater`
- `ExponentialSingleDiodeUpdater`
- `HardSigmoidUpdater`

Relevant code:

- [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L873)

### 5. Each updater binds to the function coefficients

For a given layer, the updater stores references to:

- `fn.a_coef_fn(layer)`
- `fn.b_coef_fn(layer)`

So the minimizer does not manually inspect the interactions itself. It queries the function abstraction, which already aggregates all interaction contributions.

Relevant code:

- [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L21)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L919)
- [interaction.py](/home/filip/server_code/model/function/interaction.py#L936)

### 6. Equilibrium computation

Calling:

- `energy_minimizer.compute_equilibrium()`

does the following:

- creates an iteration schedule from `mode`
- repeatedly calls `step(layer_group)`
- each updater computes a new pre-activation
- the new value is written into `layer.state`
- `layer.activate()` is applied
- the final layer states are returned

Relevant code:

- equilibrium loop:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L178)
- single update step:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L220)
- update schedule:
  [minimizer.py](/home/filip/server_code/model/minimizer/minimizer.py#L229)

## Local Update Rule

In the simplest quadratic case, each layer is updated using:

`z* = -b / (2a)`

where `a` and `b` are the coefficients of the local quadratic form induced by the total function.

Relevant code:

- [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L31)
- [minimizer.py](/home/filip/server_code/model/resistive/minimizer.py#L73)

After that, the layer activation is applied:

- perfect diode layers clamp excitatory and inhibitory halves
- other nonlinearities may leave the value unclipped and push the nonlinearity into `pre_activate()`

Relevant code:

- [layer.py](/home/filip/server_code/model/resistive/layer.py#L70)

## Short Summary

The separation of responsibilities is:

- `DeepResistiveEnergy`:
  defines the energy landscape
- `SumSeparableFunction`:
  aggregates interaction-level energy and coefficient contributions
- `Network`:
  identifies input and free layers
- `QuadraticMinimizer`:
  creates layer updaters and iterates them
- updaters:
  compute local closed-form state updates from the current aggregated coefficients

So the correct mental model is:

`DeepResistiveEnergy` does not itself "call the updaters".  
Instead, `QuadraticMinimizer` is built from the function defined by `DeepResistiveEnergy`, and equilibrium is obtained by iterating the minimizer's per-layer updaters.
