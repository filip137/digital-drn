from __future__ import annotations

import torch

from digital_drn.models.digital_transformer import DigitalGPTConfig, SmallDigitalGPT


def test_small_digital_gpt_forward_shape():
    cfg = DigitalGPTConfig(vocab_size=64, seq_len=16, d_model=32, n_heads=4, n_layers=2)
    model = SmallDigitalGPT(cfg)

    inputs = torch.randint(0, cfg.vocab_size, (4, 16), dtype=torch.long)
    logits = model(inputs)

    assert logits.shape == (4, 16, cfg.vocab_size)
    assert logits.dtype == torch.float32
