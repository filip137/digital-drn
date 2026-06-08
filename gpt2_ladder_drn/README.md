# GPT-2 Ladder DRN

Small, inspectable experiments for GPT-2 small, LoRA, Ladder Side-Tuning, and a DRN-style ladder side block. The implementation favors explicit PyTorch modules and shape/debug tests over throughput.

## What Is Included

- GPT-2-small style causal LM with tied embeddings and optional hidden-state returns.
- Optional Hugging Face GPT-2 checkpoint loading when `transformers` is installed.
- Tiny Shakespeare style next-token datasets with GPT-2 `tiktoken` tokenization when available and a character fallback.
- LoRA for `nn.Linear`, with frozen base weights.
- Ladder Side-Tuning with no-grad frozen backbone activations, gated ladder mixing, side-only output, and residual-logit output.
- DRN-LST side blocks using the repository's `TokenwiseDRNMLP`, which drives a
  `DigitalDRNBlock` and solves the free-layer equilibrium with unrolled
  asynchronous block coordinate descent.

## Quick Checks

```bash
pytest gpt2_ladder_drn/tests
```

Debug smoke train without a data file:

```bash
python gpt2_ladder_drn/train.py --mode full_finetune --debug --max_steps 50 --eval_interval 10 --block_size 32
```

Evaluate a frozen pretrained GPT-2 base, if dependencies and checkpoint access are available:

```bash
python gpt2_ladder_drn/train.py --mode eval_base --data data/tiny_shakespeare.txt --pretrained gpt2
```

## Training Commands

LoRA:

```bash
python gpt2_ladder_drn/train.py \
  --mode lora \
  --data data/tiny_shakespeare.txt \
  --pretrained gpt2 \
  --block_size 256 \
  --batch_size 8 \
  --lr 3e-4 \
  --max_steps 2000 \
  --lora_rank 8 \
  --lora_alpha 16
```

Ladder Side-Tuning:

```bash
python gpt2_ladder_drn/train.py \
  --mode lst \
  --data data/tiny_shakespeare.txt \
  --pretrained gpt2 \
  --block_size 256 \
  --batch_size 8 \
  --lr 3e-4 \
  --max_steps 2000 \
  --lst_reduction 8 \
  --lst_side_layers 12 \
  --lst_temperature 0.1 \
  --lst_output_mode side_only
```

DRN-LST:

```bash
python gpt2_ladder_drn/train.py \
  --mode lst_drn \
  --data data/tiny_shakespeare.txt \
  --pretrained gpt2 \
  --block_size 256 \
  --batch_size 8 \
  --lr 1e-4 \
  --max_steps 2000 \
  --lst_reduction 8 \
  --lst_side_layers 6 \
  --drn_iter 4 \
  --grad_clip 1.0 \
  --lst_output_mode residual_logits
```

## Results Table

Tiny Shakespeare run from 2026-04-26, using pretrained `gpt2`, `block_size=256`, `batch_size=8`, `eval_iters=20`, and `max_steps=2000`.
Run artifacts were written to `/tmp/gpt2_ladder_drn_experiments/main_20260426-000307`.
The non-DRN rows below are current. The DRN-LST rows are pending a rerun after
the side block was corrected to use the repo's coordinate-descent DRN instead
of the earlier fixed-point proxy.

| Mode | Trainable params | Peak GPU memory | Val loss | Val PPL | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen GPT-2 | 0 | | 3.3928 | 29.7499 | baseline eval only |
| LoRA rank 8 | 811,008 | 3392 MB | 2.4941 | 12.1103 | best logged loss 2.4764 at step 1750 |
| LST r=8, 12 side layers | 2,376,493 | 2602 MB | 2.8627 | 17.5091 | `side_only` |
| LST r=16, 6 side layers | 465,751 | 2566 MB | 2.9224 | 18.5866 | `side_only` |
| DRN-LST r=8, 6 side layers | pending | pending | pending | pending | rerun with coordinate-descent DRN |
| DRN-LST r=8, 12 side layers | pending | pending | pending | pending | rerun with coordinate-descent DRN |

Generated samples are written beside each checkpoint as `sample_step_*.txt`.

Final-step sample excerpts:

```text
LoRA rank 8:
To be, or not to be
The last of these things.

LST r=8, 12 side layers:
To be, or not to be, thou wild widow,
Counces me, tell me, how she parted from me;

LST r=16, 6 side layers:
To be, or not to be, king.

DRN-LST sample excerpts should be regenerated after rerunning the corrected
coordinate-descent DRN implementation.
```

## Math Summary

LoRA keeps `W` frozen and trains low-rank adapters:

```text
y = W x + (alpha / r) B A x
```

Ladder Side-Tuning freezes the backbone:

```text
h_i^B = F_i(h_{i-1}^B), with no gradient through F_i
u_i = P_i stopgrad(h_i^B)
mu_i = sigmoid(alpha_i / T)
m_i = mu_i u_i + (1 - mu_i) s_{i-1}
s_i = G_i(m_i)
logits = LMHead(U s_L)
```

In DRN-LST, the frozen backbone acts on the side DRN through that ladder mix:
`u_i` is part of the tensor passed to the side block, and `TokenwiseDRNMLP`
converts that tensor into the injected current. Gradients update the side
projections, side attention, DRN digital frontend/drive scale, and DRN
resistive parameter states, but they do not propagate into the frozen GPT-2
backbone.

DRN-LST replaces the side block MLP path with a tokenwise DRN simulation. The
side tensor is flattened from `(batch, seq_len, d_side)` to
`(batch * seq_len, d_side)`, then the digital frontend injects current into the
first DRN free layer:

```text
current = softplus(drive_scale_raw) * ff(x)
E_drive = - <z_1, current>
z_l <- activation(-b_l / (2 a_l))
G_i(x) = x + DRN_i(LN(x))
```

GPT-2 DRN-LST side blocks use explicit signed-drive encoding by default:
`ff(x)` is mirrored as `[ff(x), -ff(x)]` before it drives the first DRN free
layer. Use `--no-drn_signed_drive` only for the unsigned ablation.
The internal DRN hidden/free-layer width is `4 * d_side` by default and can be
changed with `--drn_hidden_multiplier`.

The coordinate-descent schedule defaults to `asynchronous`, and the default DRN
nonlinearity is `perfect_diode`. Training uses ordinary backpropagation through
the unrolled coordinate-descent updates. Implicit differentiation or EqProp can
be added later.
