# EP Checkpoint Beta x Amplification Sweep

- checkpoint: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/checkpoint_best.pt`
- config: `/home/filip/digital_drn/cases/conv_cases/cifar10_wider_hardsigmoid_ep_bp_100epochs/runs/cifar10_drn_only_signed_norm_readout_wider_hardsigmoid_va4_ca1_adhoc_voff1p0_ep/20260408-163101-nom-cool-2/experiment_config.json`
- device: `cuda`
- betas: `[0.0001, 0.001, 0.01, 0.1]`
- voltage_amps: `[1.0, 2.0, 4.0, 8.0]`
- current_amp: `1.0`

## Conclusion

For this deeper hard-sigmoid EP checkpoint, `beta` is not the main problem once it is at least `1e-3`.

- with no amplification (`voltage_amp = 1`, `current_amp = 1`), the BP-vs-EP gradient match is already good at `beta = 1e-3`
  - `overall_cosine = 0.991569`
  - `overall_relative_error = 0.1358`
- increasing `beta` further to `1e-2` or `1e-1` changes little in the unamplified regime
- the dominant degradation comes from amplification
  - `va = 2`: `overall_cosine = 0.899009`, `overall_relative_error = 2.4672`
  - `va = 4`: `overall_cosine = 0.808004`, `overall_relative_error = 11.8662`
  - `va = 8`: `overall_cosine = 0.735633`, `overall_relative_error = 46.7924`

So the practical read is:

1. `beta = 1e-4` is too small.
2. `beta >= 1e-3` is already sufficient in the no-amplification case.
3. The remaining failure mode is amplification, not beta choice.

## Overall Cosine

| voltage_amp | beta | overall_cosine | block_0_drn | block_1_drn | disp_mean |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0001 | 0.804504 | 0.933439 | 0.428183 | 0.170431 |
| 1 | 0.001 | 0.991569 | 0.997188 | 0.940894 | 0.170431 |
| 1 | 0.01 | 0.997190 | 0.998126 | 0.982219 | 0.170431 |
| 1 | 0.1 | 0.997302 | 0.998155 | 0.982992 | 0.170454 |
| 2 | 0.0001 | 0.695360 | 0.816001 | 0.242561 | 0.324228 |
| 2 | 0.001 | 0.893110 | 0.994422 | 0.808645 | 0.324228 |
| 2 | 0.01 | 0.899009 | 0.998116 | 0.986045 | 0.324228 |
| 2 | 0.1 | 0.899018 | 0.998087 | 0.990149 | 0.324249 |
| 4 | 0.0001 | 0.488951 | 0.604632 | 0.140668 | 0.444320 |
| 4 | 0.001 | 0.796911 | 0.985845 | 0.611593 | 0.444320 |
| 4 | 0.01 | 0.808004 | 0.998534 | 0.974617 | 0.444320 |
| 4 | 0.1 | 0.808165 | 0.998691 | 0.989374 | 0.444333 |
| 8 | 0.0001 | 0.318547 | 0.429770 | 0.108995 | 0.469740 |
| 8 | 0.001 | 0.712979 | 0.966730 | 0.508048 | 0.469740 |
| 8 | 0.01 | 0.735633 | 0.998402 | 0.956920 | 0.469740 |
| 8 | 0.1 | 0.735997 | 0.998862 | 0.986200 | 0.469747 |
