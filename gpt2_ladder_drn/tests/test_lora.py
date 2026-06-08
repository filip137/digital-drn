import torch

from gpt2_ladder_drn import DebugGPT2Config, GPT2LMHeadModel, LoRALinear, apply_lora


def test_lora_forward_backward_keeps_base_grad_empty():
    torch.manual_seed(1)
    cfg = DebugGPT2Config(dropout=0.0)
    model = GPT2LMHeadModel(cfg)
    apply_lora(model, target_modules=("c_attn",), r=4, alpha=8)

    assert any(isinstance(module, LoRALinear) for module in model.modules())

    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss = model(input_ids, targets=input_ids)["loss"]
    loss.backward()

    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module.A.weight.grad is not None
            assert module.B.weight.grad is not None
            assert module.base.weight.grad is None
            if module.base.bias is not None:
                assert module.base.bias.grad is None
