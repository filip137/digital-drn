# OPT MLP DRN Active-Layer Curriculum

Date: 2026-05-06

Branch: `opt-partial-fine-tuning`

## Summary

Added a one-by-one activation curriculum for OPT MLP-to-DRN joint training. The model can now construct and initialize all selected DRN replacements, but only allow a staged subset to actually replace the frozen teacher MLP path during training.

For last-three replacement, the intended schedule is:

```text
[11] -> [10, 11] -> [9, 10, 11]
```

## New Training Flags

```bash
--active_layer_schedule 'last:1;last:2;last:3'
--replacement_schedule 0.1,0.3,0.6,1.0
```

`active_layer_schedule` controls which DRN layers are allowed to replace their teacher MLP at each training stage. `replacement_schedule` controls the stochastic probability of using the DRN path within the currently active layers.

## Behavior

- Inactive replacement layers use the frozen teacher MLP path.
- Active replacement layers sample teacher path vs DRN path according to the replacement probability.
- Validation and test force all selected DRNs active with replacement probability `1.0`, so reported metrics are for the fully replaced student.
- Metrics now record active replacement fraction, student replacement fraction, and mean replacement probability.

## Verification

Focused OPT tests passed:

```text
24 passed
```

Implementation commit:

```text
bbb474b Add active-layer OPT DRN curriculum
```
