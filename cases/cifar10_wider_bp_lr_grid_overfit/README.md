# Widened BP LR Grid Overfit

- config: `cifar10_drn_only_signed_norm_readout_wider`
- device: `cuda`
- epochs: `100`
- train_subset: `128`
- test_subset: `128`
- digital_mults: `[1.0, 3.0, 10.0]`
- drn_mults: `[1.0, 3.0, 10.0]`

## Ranked By Train Best

| digital_mult | drn_mult | ff_lr | drn_lr | head_lr | train_best | val_best | train_epoch | val_epoch | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 10 | 0.003 | 0.001 | 0.0009 | 0.7188 | 0.3125 | 81 | 19 | `20260409-170607-integnano-akib` |
| 1 | 3 | 0.001 | 0.0003 | 0.0003 | 0.6953 | 0.3203 | 96 | 31 | `20260409-165901-integnano-akib` |
| 3 | 3 | 0.003 | 0.0003 | 0.0009 | 0.6953 | 0.2969 | 98 | 28 | `20260409-170422-integnano-akib` |
| 1 | 10 | 0.001 | 0.001 | 0.0003 | 0.6875 | 0.3281 | 94 | 19 | `20260409-170051-integnano-akib` |
| 10 | 10 | 0.01 | 0.001 | 0.003 | 0.5547 | 0.2812 | 93 | 43 | `20260409-171117-integnano-akib` |
| 1 | 1 | 0.001 | 0.0001 | 0.0003 | 0.5078 | 0.3281 | 90 | 56 | `20260409-165708-integnano-akib` |
| 3 | 1 | 0.003 | 0.0001 | 0.0009 | 0.5078 | 0.3203 | 96 | 38 | `20260409-170234-integnano-akib` |
| 10 | 3 | 0.01 | 0.0003 | 0.003 | 0.4766 | 0.2969 | 98 | 40 | `20260409-170929-integnano-akib` |
| 10 | 1 | 0.01 | 0.0001 | 0.003 | 0.1562 | 0.0781 | 3 | 1 | `20260409-170753-integnano-akib` |
