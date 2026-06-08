from gpt2_ladder_drn import DebugGPT2Config, GPT2LMHeadModel, LadderSideGPT2, apply_lora


def test_lora_freezes_all_non_lora_parameters():
    cfg = DebugGPT2Config(dropout=0.0)
    model = GPT2LMHeadModel(cfg)
    apply_lora(model, target_modules=("c_attn",), r=4, alpha=8)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]

    assert trainable
    assert all(".A.weight" in name or ".B.weight" in name for name in trainable)


def test_ladder_freezes_base_and_keeps_side_trainable():
    cfg = DebugGPT2Config(dropout=0.0)
    base = GPT2LMHeadModel(cfg)
    model = LadderSideGPT2(base, reduction_factor=8, num_side_layers=2)

    assert all(not param.requires_grad for param in model.base.parameters())
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(not name.startswith("base.") for name in trainable)
