# CIFAR-10 Overfit Debug

## Goal

Debug CIFAR-10 trainability by asking a simple question:

- Can the model memorize `128` training samples?

For these notes, the main success criterion is:

- train accuracy close to `100%`

Validation accuracy on a separate `128`-sample test subset is recorded, but it is **not** the overfit criterion.

## Important note about model file names

Some YAML file names were reused while iterating on the architecture. The authoritative record of what actually ran is:

- the saved `experiment_config.json`
- inside each run directory under `simulation_results/<model-selection>/<timestamp-host>/`

In particular, runs under `simulation_results/conv_cifar10_1block/...` do **not** all correspond to the same architecture.

## Failed cases

### 0. Loaded-model beta sweep on the best overfit-debug checkpoint

Case:

- [loaded_model_beta_sweep](/home/filip/digital_drn/cases/conv_cases/cifar10_overfit_debug/loaded_model_beta_sweep/README.md)

Checkpoint:

- [checkpoint_best.pt](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2/checkpoint_best.pt)

Interpretation:

- On the loaded CIFAR checkpoint, the head gradients match exactly across the tested beta grid, while the DRN parameter groups remain the limiting factor at small beta.
- The free-vs-nudged displacement is large but nearly flat across beta in this diagnostic, so the beta sensitivity shows up more clearly in the gradient comparison than in the state displacement.

### 1. Original 2-block digital/analog CIFAR model

Model family:

- [cifar10_digital_analog_v0.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_digital_analog_v0.yaml)

Representative run:

- [20260403-153953-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0/20260403-153953-nom-cool-2)

Final metrics:

- train loss: `1.3679`
- train accuracy: `0.4453`
- best train accuracy: `0.5312`
- val loss: `2.7052`
- val accuracy: `0.2578`

Interpretation:

- This model did **not** memorize the `128`-sample training subset.
- The original `digital -> drn -> digital -> drn + digital head` formulation was not trainable enough in this setup.

### 2. One-block `F1 -> E1 -> head` with conv DRN block

Model family:

- [cifar10_digital_analog_v0_1block.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_digital_analog_v0_1block.yaml)

Representative run:

- [20260403-161550-integnano-akib](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_1block/20260403-161550-integnano-akib)

Final metrics:

- train loss: `1.9701`
- train accuracy: `0.2266`
- best train accuracy: `0.3047`
- val loss: `2.2566`
- val accuracy: `0.2031`

Interpretation:

- Even the simpler `digital -> conv DRN -> digital head` formulation did not overfit.
- This pointed to the first digital-to-conv-analog handoff as a likely weak point.

### 3. Overfit-debug learning-rate sweeps did not fix the issue

Model family:

- [cifar10_digital_analog_v0_overfit_debug.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_digital_analog_v0_overfit_debug.yaml)

Run family:

- [simulation_results/cifar10_digital_analog_v0_overfit_debug](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug)

Observed range across runs:

- best train accuracy between `0.1328` and `0.3203`
- final val accuracy between `0.0859` and `0.2578`

Representative runs:

- [20260403-155821-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-155821-nom-cool-2)
- [20260403-160125-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160125-nom-cool-2)
- [20260403-160450-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_overfit_debug/20260403-160450-nom-cool-2)

Interpretation:

- Simple FF/DRN learning-rate changes were not enough to rescue the failing architecture family.

### 4. Pure digital MLP-style ablation was not a strong baseline

Model family:

- [cifar10_digital_mlp_v0.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_digital_mlp_v0.yaml)

Representative run:

- [20260403-160918-integnano-akib](/home/filip/digital_drn/simulation_results/cifar10_digital_mlp_v0/20260403-160918-integnano-akib)

Final metrics:

- train loss: `1.5572`
- train accuracy: `0.4297`
- best train accuracy: `0.5078`
- val loss: `2.2734`
- val accuracy: `0.2734`

Interpretation:

- This ablation also failed to memorize the tiny subset.
- It is therefore not a strong control baseline for judging the analog blocks.

### 5. Conv-only analog readout was weak

Representative run:

- [20260403-162253-nom-cool-2](/home/filip/digital_drn/simulation_results/conv_cifar10_1block/20260403-162253-nom-cool-2)

Saved model name in the run config:

- `cifar10_digital_analog_v0_conv_cifar10_1block`

Observed outcome:

- best train accuracy: `0.3516`

Interpretation:

- A conv analog readout without a dense analog readout layer was weak in this early formulation.

### 6. Adding a final digital `10 -> 10` head on top of a working mixed-analog readout hurt overfitting

Model family:

- [cifar10_mixed_analog_2block_plus_digital_head.yaml](/home/filip/digital_drn/hydra_conf/model/cifar10_mixed_analog_2block_plus_digital_head.yaml)

Representative run:

- [20260403-201207-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_mixed_analog_2block_plus_digital_head/20260403-201207-nom-cool-2)

Final metrics:

- train loss: `0.7589`
- train accuracy: `0.8359`
- best train accuracy: `0.8672`
- val loss: `2.3162`
- val accuracy: `0.2266`

Direct comparison:

- the matching analog-readout model without the extra digital head reached `1.0000` train accuracy
- adding the digital `10 -> 10` head reduced the best train accuracy to `0.8672`
- rerunning the same idea with an explicit `cross_entropy_readout` head also underperformed, reaching only `0.7891` best train accuracy in [20260403-201750-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_mixed_analog_2block_plus_digital_head/20260403-201750-nom-cool-2)

Interpretation:

- This did not fully break learning, but it clearly damaged the tiny-subset overfit behavior.
- The result supports the hypothesis that an FF output head on top of the final analog readout is a weak design choice in this current CIFAR setup.

## Working controls / counterexamples

These runs matter because they show that the CIFAR data pipeline and the training code are not fundamentally broken.

### A. Digital conv + dense DRN readout overfits

Representative run:

- [20260403-161323-integnano-akib](/home/filip/digital_drn/simulation_results/conv_cifar10_1block/20260403-161323-integnano-akib)

Saved model name in the run config:

- `sequential_digital_drn`

Observed outcome:

- best train accuracy: `1.0000`

Meaning:

- A CIFAR model with a digital conv frontend and a dense DRN readout can memorize the `128`-sample subset.

### B. One-block mixed analog readout overfits

Representative run:

- [20260403-163610-nom-cool-2](/home/filip/digital_drn/simulation_results/conv_cifar10_1block/20260403-163610-nom-cool-2)

Saved model name in the run config:

- `cifar10_conv_dense_analog_1block`

Final metrics:

- train loss: `0.0211`
- train accuracy: `1.0000`
- val loss: `3.3956`
- val accuracy: `0.1562`

Meaning:

- `digital conv -> analog conv -> analog dense readout` in a single block can memorize the tiny subset.

### C. Two-block mixed analog readout overfits

Representative run:

- [20260403-200547-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_mixed_analog_2block_readout/20260403-200547-nom-cool-2)

Saved model name in the run config:

- `cifar10_mixed_analog_2block_readout`

Final metrics:

- train loss: `0.0246`
- train accuracy: `1.0000`
- val loss: `3.8399`
- val accuracy: `0.1875`

Meaning:

- A `digital -> drn -> digital -> drn(readout)` formulation can also memorize the tiny subset when the last analog block uses a dense readout state.

### D. Original `v0` architecture overfits once the output head is replaced with a flat readout

Representative run:

- [20260403-203315-nom-cool-2](/home/filip/digital_drn/simulation_results/cifar10_digital_analog_v0_flat_readout/20260403-203315-nom-cool-2)

Saved model name in the run config:

- `cifar10_digital_analog_v0_flat_readout`

Final metrics:

- train loss: `0.0102`
- train accuracy: `0.9922`
- val loss: `10.2137`
- val accuracy: `0.1562`

Observed behavior:

- the run reached `1.0000` train accuracy by epoch `49`
- it stayed near-perfect for most of training

Meaning:

- the original `v0` block structure was not the core problem
- replacing the pooled `64 -> 10` digital head with a flat cross-entropy readout was enough to make the model memorize the `128`-sample subset
- this strongly points to the original output head as the main bottleneck

## Current interpretation

The broad statement:

- "digital/analog coupling is broken"

is too strong.

What the results support more specifically is:

- some early CIFAR formulations with conv DRN blocks do **not** overfit
- a conv analog block followed directly by a digital head was particularly weak
- adding an **analog dense readout state** makes the block much easier to optimize
- even when a two-block mixed analog model already works, adding a final digital `10 -> 10` head on top hurts overfitting noticeably
- the original `v0` architecture can overfit once its pooled output head is replaced by a flat readout
- the problem is therefore not DRN itself, and not any digital-to-analog coupling in general
- the failing cases seem to be **specific block/readout formulations**, with the final FF head now looking like a particularly suspicious component

## Practical takeaway

For CIFAR-10 prototyping, the safer block template is:

- digital conv frontend
- analog conv state
- analog dense readout state
- no extra FF head on top unless there is a strong reason to include it

Or, if you keep the original `v0` two-block conv-DRN stack, use a stronger final readout than:

- `adaptive_avg_pool2d -> linear 64 -> 10`

This worked both:

- in a single-block model
- and in a two-block `digital -> drn -> digital -> drn(readout)` model

## Next useful comparisons

- Compare conv-only analog readout against conv+dense analog readout under the same digital frontend.
- Keep using the `128`-sample overfit test as the first filter before full CIFAR runs.
- When a formulation overfits, only then move on to real generalization experiments.
