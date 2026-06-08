import torch

from gpt2_ladder_drn import DebugGPT2Config, GPT2LMHeadModel


def test_gpt2_debug_shapes_and_hidden_states():
    torch.manual_seed(0)
    cfg = DebugGPT2Config(dropout=0.0)
    model = GPT2LMHeadModel(cfg)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    out = model(input_ids, targets=input_ids, return_hidden_states=True)

    assert out["logits"].shape == (batch_size, seq_len, cfg.vocab_size)
    assert out["loss"] is not None
    assert len(out["hidden_states"]) == cfg.n_layer + 1
    assert out["hidden_states"][0].shape == (batch_size, seq_len, cfg.n_embd)
