# Digital DRN Status And Library Plan

This note summarizes:

- what was implemented in the current `digital_drn` prototype
- what was actually tested
- what still needs to be done to turn it into a functioning library

## 1. Current Prototype Status

The current prototype exists under `/home/filip/digital_drn`.

Implemented core pieces:

- `DigitalDRNBlock`
  - owns:
    - a feedforward / digital block `ff`
    - a learnable positive drive scale
    - a block-local resistive energy
    - a `QuadraticMinimizer`
- `DenseDRNBlockEnergy`
  - dense-only resistive core for now
  - free layers + dense resistive interactions + biases + optional nonlinear interactions
- `FFCurrentInteraction`
  - injects the FF output as current into the first free resistive layer
- `DigitalDRNNet`
  - generic chain of blocks
- `SequentialDigitalDRNNet`
  - simple builder for stacked blocks

Current training path:

- standard autograd BP through the unrolled equilibrium solve
- resistive parameter tensors are optimized directly via their `.state`
- explicit helper methods exist for:
  - `set_device(...)`
  - `enable_resistive_grad_(...)`
  - `optimizer_tensors()`
  - `clamp_resistive_params_()`
  - `detach_state_()`

Current FF support:

- dense FF path:
  - `Flatten -> Linear -> activation`
- conv FF path:
  - repeated `Conv2d -> activation -> optional MaxPool`
  - then `Flatten -> Linear -> activation`

Current resistive side support:

- dense resistive interactions only
- block-local hidden/output equilibrium
- optional reused DRN nonlinear interaction modes are wired, but not all were benchmarked seriously
- minimal vendored DRN core now lives under `digital_drn/core`
  - variables
  - layers
  - parameters
  - interactions
  - minimizers

Packaging status:

- `pyproject.toml` is now present
- the package installs with:
  - `pip install -e .`
- runtime imports no longer depend on `/home/filip/server_code`

## 2. What Was Tested

Unit / integration tests:

- test suite currently passes:
  - `7 passed`
- covered behaviors include:
  - FF current interaction
  - block-local energy wiring
  - block forward/reset behavior
  - sequential network chaining
  - BP gradient flow into:
    - FF parameters
    - resistive parameters
    - learnable drive scale
  - conv FF frontend shape/path

Training / smoke experiments that were actually run:

### Dense prototype

- short MNIST smoke, dense-only, 1 block, `4` equilibrium iterations
  - stable BP training confirmed
- full MNIST, dense-only, `10` epochs
  - FF activation: `tanh`
  - test accuracy: `0.9251`
  - test loss: `0.2647`
  - learned drive scale: `1.034514`

### Conv FF + resistive block

- full MNIST, conv FF frontend, `100` epochs
  - FF activation: `tanh`
  - conv FF:
    - `Conv(1->32, 3x3, pad=1) -> MaxPool(2)`
    - `Conv(32->64, 3x3, pad=1) -> MaxPool(2)`
    - `Flatten -> Linear -> tanh`
  - resistive block:
    - one hidden resistive layer of size `128`
    - output layer of size `10`
    - `4` equilibrium iterations
  - test accuracy: `0.9890`
  - test loss: `0.1263`
  - learned drive scale: `1.838454`

### Nonlinearity comparison already checked

- in a short-run comparison, `perfect_diode` did not beat the current linear resistive setup
- that does **not** mean diode-style nonlinearities are useless
- it only means they are not yet tuned or properly integrated for this prototype

## 3. What The Prototype Still Is Not

This is still a prototype, not yet a clean library.

Current limitations:

- Resistive parameters are not native `nn.Parameter`s.
- There is no stable checkpoint / restore API for the full model.
- There is no official trainer abstraction.
- There is no CLI / experiment config system.
- There is no proper model zoo:
  - no `DigitalDRNVGG`
  - no ResNet-style variant
  - no conv resistive block yet
- There is no EP implementation in this package yet.
- There is no proper benchmark matrix across seeds / datasets / nonlinearities.

## 4. What Is Needed For A Functioning Library

The work should be done in roughly this order.

### A. Make it self-contained

Completed in this pass:

- Removed the runtime dependency on `/home/filip/server_code`.
- Vendored the minimal DRN primitives needed by the prototype:
  - layers
  - parameters
  - interactions
  - minimizers
- Added a package layout with:
  - `pyproject.toml`
  - installable module metadata
  - pinned core dependencies

What still remains in this area:

- decide whether to keep the current flat package-root layout or move to a `src/` layout later
- trim or harden the vendored core API for public use
- document optional Lambert-W support for exponential diode modes

### B. Formalize parameter and checkpoint handling

- Decide whether resistive parameters stay as custom wrapper objects or become registered `nn.Parameter`s.
- Add consistent save/load support for:
  - FF parameters
  - resistive parameter states
  - learned drive scales
- Add a full-model checkpoint API.

Right now training works, but serialization is not yet library-grade.

### C. Add stable training interfaces

- Add a trainer module for BP:
  - train loop
  - eval loop
  - metric logging
  - gradient clipping if needed
  - checkpointing
- Add reproducibility controls:
  - seeds
  - deterministic options
  - device selection
- Add configuration objects or dataclasses for:
  - block config
  - network config
  - trainer config

### D. Build the intended model hierarchy

- Keep the current generic base classes:
  - `DigitalDRNNet`
  - `DigitalDRNBlock`
- Add concrete subclasses/builders:
  - `DigitalDRNVGG`
  - VGG-style block builder
  - later, maybe residual variants

This matches the original intention of mirroring the `hybrid_bp_ep_official` structure.

### E. Improve the resistive side

- Add conv resistive blocks, not only conv FF frontends.
- Revisit diode / nonlinear interaction modes:
  - `lpw_diode`
  - `hard_sigmoid`
  - `double_diode_quadratic`
  - `double_diode_exponential`
  - `single_diode_exponential`
- Add paired signed-current injection when using diode-style hidden structure.
- Test multi-block equilibrium behavior, not only the 1-block MNIST setup.

### F. Add algorithms beyond plain BP

- Add explicit BPTT-facing utilities if detailed trajectory training is needed.
- Add EP support if the goal is to reconnect to the original BP-EP direction.
- Decide whether EP stays block-local or becomes a full-network algorithm in this package.

### G. Add documentation and examples

- Minimal README:
  - what the library is
  - what “digital DRN” means here
  - how to train a model
  - how to define a new block/network
- Example scripts:
  - dense MNIST
  - conv-FF MNIST
  - later: VGG-style model

## 5. Recommended Immediate Next Steps

If work resumes tomorrow, the most useful next steps are:

1. Make the package self-contained.
2. Add checkpoint/save-load support.
3. Turn the MNIST runner into a reusable trainer.
4. Add a proper VGG-style subclass on top of the current generic base classes.
5. Then revisit true resistive nonlinearities and conv resistive blocks.

## 6. Bottom Line

What already works:

- generic equilibrium blocks
- FF current injection
- autograd BP through equilibrium
- dense MNIST training
- conv FF + resistive output training
- strong MNIST result (`98.9%`) in the current conv-FF prototype

What still makes it a prototype instead of a library:

- no proper serialization/config/trainer API
- no finalized model hierarchy
- no fully developed resistive nonlinear/block variants
