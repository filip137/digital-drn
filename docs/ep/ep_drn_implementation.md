# EP In The Initial DRN Code

This note summarizes how the original DRN code computed equilibrium-propagation (EP) gradients with two nudging phases.

The reference implementation is in:

- [server_code/training/sgd.py](/home/filip/server_code/training/sgd.py)
- [server_code/model/function/cost.py](/home/filip/server_code/model/function/cost.py)

## High-Level Structure

The original DRN code used a classical energy-based setup:

- energy function: `E(theta, s)`
- cost function: `C(s)`
- augmented function: `E(theta, s) + beta * C(s)`

This augmentation is implemented explicitly with:

- `Nudging` in [sgd.py](/home/filip/server_code/training/sgd.py#L10)
- `AugmentedFunction` in [sgd.py](/home/filip/server_code/training/sgd.py#L61)

`AugmentedFunction.eval()` returns:

```text
E(theta, s) + beta * C(s)
```

where `beta` is stored as `nudging`.

## Two-Phase EP Gradient Rule

The main estimator is `EquilibriumProp` in [sgd.py](/home/filip/server_code/training/sgd.py#L150).

The documented estimator is:

```text
[ dE(theta, s(beta_2)) / dtheta - dE(theta, s(beta_1)) / dtheta ] / (beta_2 - beta_1)
```

where:

```text
s(beta) = argmin_s [ E(theta, s) + beta * C(s) ]
```

This is implemented in `compute_gradient()` in [sgd.py](/home/filip/server_code/training/sgd.py#L247).

## Phase Flow

### 1. Start from the free state

The method assumes the network is already at the free equilibrium for `beta = 0`.

It stores:

```python
layers_free = [layer.state for layer in self._layers]
```

in [sgd.py](/home/filip/server_code/training/sgd.py#L261).

### 2. First nudged phase

It sets:

```python
self._augmented_fn.nudging = self._first_nudging
layers_first = self._energy_minimizer.compute_equilibrium()
```

in [sgd.py](/home/filip/server_code/training/sgd.py#L262).

### 3. Reset to the free state

Before the second phase, it restores the free state:

```python
for layer, state in zip(self._layers, layers_free):
    layer.state = state
```

in [sgd.py](/home/filip/server_code/training/sgd.py#L266).

### 4. Second nudged phase

It sets:

```python
self._augmented_fn.nudging = self._second_nudging
layers_second = self._energy_minimizer.compute_equilibrium()
```

in [sgd.py](/home/filip/server_code/training/sgd.py#L267).

### 5. Build the EP gradient estimate

The standard formula is implemented in `_standard_param_grads()` in [sgd.py](/home/filip/server_code/training/sgd.py#L357).

It:

1. sets the layers to `layers_first`
2. computes `dE/dtheta`
3. sets the layers to `layers_second`
4. computes `dE/dtheta`
5. divides the difference by `(beta_2 - beta_1)`

Concretely:

```python
param_grads = [
    (second - first) / (self._second_nudging - self._first_nudging)
    for first, second in zip(grads_first, grads_second)
]
```

from [sgd.py](/home/filip/server_code/training/sgd.py#L381).

So the parameter gradient is a finite-difference estimate between two nudged equilibria.

## Which Two Nudging Values Were Used?

This depends on `variant`, set in `_set_nudgings()` in [sgd.py](/home/filip/server_code/training/sgd.py#L430):

- `positive`:
  - `beta_1 = 0`
  - `beta_2 = +beta`
- `negative`:
  - `beta_1 = -beta`
  - `beta_2 = 0`
- `centered`:
  - `beta_1 = -beta`
  - `beta_2 = +beta`

The centered version is the genuine two-sided estimate.

## Cost Function Handling

The output cost lived in a separate `CostFunction`.

For example, `SquaredError` is in [cost.py](/home/filip/server_code/model/function/cost.py#L85).

Targets were set by:

```python
cost_fn.set_target(label)
```

which converts class labels to one-hot targets in [cost.py](/home/filip/server_code/model/function/cost.py#L41).

So the nudging force came from the cost function through the augmented energy, not from a manual backward pass through logits.

## Direct Cost Gradients On Cost Parameters

There is one extra detail in the original implementation.

At the top of `compute_gradient()`, it also computes:

```python
cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]
```

in [sgd.py](/home/filip/server_code/training/sgd.py#L258).

This handles the case where the cost function itself contains parameters, such as a readout layer.

The final return is:

```python
return param_grads + cost_grads
```

from [sgd.py](/home/filip/server_code/training/sgd.py#L276).

So:

- energy parameters get EP finite-difference gradients
- cost/readout parameters can get direct gradients from `C`

## Alternative Formula

There is also an optional `use_alternative_formula=True` mode.

That path is implemented in `_alternative_param_grads()` in [sgd.py](/home/filip/server_code/training/sgd.py#L385).

Instead of differencing `dE/dtheta` directly, it:

- computes a direction in state space:

```text
(s(beta_2) - s(beta_1)) / (beta_2 - beta_1)
```

- resets layers to the free state
- applies a second-derivative-based parameter update through `second_fn(...)`

For most practical discussion of the old DRN code, the standard formula above is the main one.

## Detailed Gradients

`detailed_gradients()` in [sgd.py](/home/filip/server_code/training/sgd.py#L302) computes the same idea over full trajectories instead of just final equilibria:

- trajectory under `beta_1`
- trajectory under `beta_2`
- finite-difference layer and parameter gradients at each time step

This was used for analysis and plotting rather than ordinary training.

## Short Summary

The original DRN EP implementation did the following:

1. Assume the network is at the free equilibrium.
2. Run one nudged equilibrium with `beta_1`.
3. Reset back to the free state.
4. Run a second nudged equilibrium with `beta_2`.
5. Estimate parameter gradients from the difference of energy gradients:

```text
grad_theta ≈ [ dE/dtheta at s(beta_2) - dE/dtheta at s(beta_1) ] / (beta_2 - beta_1)
```

For centered EP, this becomes a symmetric estimate using `-beta` and `+beta`.

## Difference From `hybrid_bp_ep_official`

This is different from the later blockwise EP code in `/home/filip/hybrid_bp_ep_official`.

There:

- each block exposed a primitive `Phi(...)`
- EP was applied block-by-block in reverse order
- an upstream error current was propagated between blocks

In the initial DRN code:

- EP was implemented at the level of a global energy function plus cost function
- the two nudging phases were applied directly through the augmented energy
- gradients came from finite differences of `dE/dtheta` between the two equilibria

So the original DRN code was closer to classical EP, while `hybrid_bp_ep_official` used a blockwise EP construction.
