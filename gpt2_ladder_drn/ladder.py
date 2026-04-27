from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from digital_drn.models.tokenwise import TokenwiseDRNMLP

from .config import GPT2Config
from .drn import PureDRNSideBlock, SideDRNBlock
from .model_gpt2 import CausalSelfAttention, GPT2LMHeadModel, LayerNorm, MLP


class SideTransformerBlock(nn.Module):
    def __init__(self, side_config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(side_config.n_embd, bias=side_config.bias)
        self.attn = CausalSelfAttention(side_config)
        self.ln_2 = LayerNorm(side_config.n_embd, bias=side_config.bias)
        self.mlp = MLP(side_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LadderSideGPT2(nn.Module):
    def __init__(
        self,
        base: GPT2LMHeadModel,
        reduction_factor: int = 8,
        num_side_layers: int | None = None,
        temperature: float = 0.1,
        output_mode: str = "side_only",
        side_block_type: str = "transformer",
        drn_iter: int = 8,
        drn_damping: float = 0.5,
    ) -> None:
        super().__init__()
        if reduction_factor <= 0:
            raise ValueError("reduction_factor must be strictly positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive.")
        if output_mode not in {"side_only", "residual_logits"}:
            raise ValueError("output_mode must be 'side_only' or 'residual_logits'.")
        if side_block_type not in {"transformer", "drn", "pure_drn"}:
            raise ValueError("side_block_type must be 'transformer', 'drn', or 'pure_drn'.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

        self.reduction_factor = int(reduction_factor)
        self.temperature = float(temperature)
        self.output_mode = output_mode
        self.side_block_type = side_block_type
        self.d_model = int(base.config.n_embd)
        self.num_base_layers = int(base.config.n_layer)
        self.num_side_layers = int(num_side_layers if num_side_layers is not None else self.num_base_layers)
        if not 1 <= self.num_side_layers <= self.num_base_layers:
            raise ValueError("num_side_layers must lie in [1, base.config.n_layer].")
        if self.d_model % self.reduction_factor != 0:
            raise ValueError("base hidden size must be divisible by reduction_factor.")
        self.d_side = self.d_model // self.reduction_factor
        self.base_indices = _layer_indices(self.num_base_layers, self.num_side_layers)

        side_config = GPT2Config(
            vocab_size=base.config.vocab_size,
            block_size=base.config.block_size,
            n_layer=self.num_side_layers,
            n_head=choose_side_heads(self.d_side),
            n_embd=self.d_side,
            dropout=base.config.dropout,
            bias=base.config.bias,
        )
        self.down_projs = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_side) for _ in range(self.num_side_layers + 1)]
        )
        self.side_blocks = nn.ModuleList(
            [
                _build_side_block(side_config, side_block_type, drn_iter=drn_iter, drn_damping=drn_damping)
                for _ in range(self.num_side_layers)
            ]
        )
        self.gates = nn.Parameter(torch.zeros(self.num_side_layers + 1))
        self.up_proj = nn.Linear(self.d_side, self.d_model)
        self.side_ln_f = LayerNorm(self.d_side, bias=base.config.bias)
        if self.output_mode == "residual_logits":
            self.gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("gamma", None)

    def train(self, mode: bool = True) -> "LadderSideGPT2":
        super().train(mode)
        self.base.eval()
        return self

    def drn_mlps(self) -> list[TokenwiseDRNMLP]:
        return [module for module in self.side_blocks.modules() if isinstance(module, TokenwiseDRNMLP)]

    def resistive_param_states(self) -> list[torch.Tensor]:
        return [state for mlp in self.drn_mlps() for state in mlp.resistive_param_states()]

    def named_resistive_parameters(self):
        for block_idx, block in enumerate(self.side_blocks):
            for module_name, module in block.named_modules():
                if isinstance(module, TokenwiseDRNMLP):
                    prefix = f"side_blocks.{block_idx}"
                    if module_name:
                        prefix = f"{prefix}.{module_name}"
                    for name, tensor in module.named_resistive_parameters():
                        yield f"{prefix}.{name}", tensor

    def optimizer_tensors(self) -> list[torch.Tensor]:
        return list(self.parameters()) + self.resistive_param_states()

    def enable_resistive_grad_(self, enabled: bool = True) -> "LadderSideGPT2":
        for mlp in self.drn_mlps():
            mlp.enable_resistive_grad_(enabled)
        return self

    def zero_resistive_grad_(self, set_to_none: bool = True) -> "LadderSideGPT2":
        for mlp in self.drn_mlps():
            mlp.zero_resistive_grad_(set_to_none=set_to_none)
        return self

    def clamp_resistive_params_(self) -> "LadderSideGPT2":
        for mlp in self.drn_mlps():
            mlp.clamp_resistive_params_()
        return self

    def detach_state_(self) -> "LadderSideGPT2":
        for mlp in self.drn_mlps():
            mlp.detach_state_()
        return self

    def forward(
        self,
        input_ids: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        with torch.no_grad():
            base_out = self.base(input_ids, targets=None, return_hidden_states=True)

        hidden_states = [h.detach() for h in base_out["hidden_states"]]
        base_logits = base_out["logits"].detach()

        gate0 = torch.sigmoid(self.gates[0] / self.temperature)
        s = gate0 * self.down_projs[0](hidden_states[0])
        side_hidden_states: list[torch.Tensor] | None = [s] if return_hidden_states else None

        for j, block in enumerate(self.side_blocks):
            base_idx = self.base_indices[j]
            u = self.down_projs[j + 1](hidden_states[base_idx])
            mu = torch.sigmoid(self.gates[j + 1] / self.temperature)
            mix = mu * u + (1.0 - mu) * s
            s = block(mix)
            if side_hidden_states is not None:
                side_hidden_states.append(s)

        side_h = self.side_ln_f(s)
        side_d = self.up_proj(side_h)
        side_logits = self.base.lm_head(side_d)
        if self.output_mode == "residual_logits":
            logits = base_logits + self.gamma * side_logits
        else:
            logits = side_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return {"logits": logits, "loss": loss, "hidden_states": side_hidden_states}


def choose_side_heads(d_side: int) -> int:
    for n_head in (4, 8, 6, 3, 2, 1):
        if d_side % n_head == 0:
            return n_head
    raise ValueError(f"Could not choose a side attention head count for d_side={d_side}.")


def structural_init_from_backbone(
    side_model: LadderSideGPT2,
    base_model: GPT2LMHeadModel,
    method: str = "magnitude",
) -> None:
    if method != "magnitude":
        raise ValueError("Only magnitude structural initialization is implemented.")
    # TODO: Fisher structural initialization requires sample gradients from data.
    with torch.no_grad():
        for down_proj in side_model.down_projs:
            row_scores = base_model.transformer.wte.weight.abs().sum(dim=0)
            cols = torch.topk(row_scores, k=side_model.d_side).indices.sort().values
            down_proj.weight.zero_()
            down_proj.bias.zero_()
            down_proj.weight[:, cols] = torch.eye(side_model.d_side, device=down_proj.weight.device)


def _layer_indices(num_base_layers: int, num_side_layers: int) -> list[int]:
    return [max(1, round((j + 1) * num_base_layers / num_side_layers)) for j in range(num_side_layers)]


def _build_side_block(
    side_config: GPT2Config,
    side_block_type: str,
    drn_iter: int,
    drn_damping: float,
) -> nn.Module:
    if side_block_type == "transformer":
        return SideTransformerBlock(side_config)
    if side_block_type == "drn":
        return SideDRNBlock(side_config, drn_iter=drn_iter, drn_damping=drn_damping)
    if side_block_type == "pure_drn":
        return PureDRNSideBlock(side_config, drn_iter=drn_iter, drn_damping=drn_damping)
    raise ValueError(f"Unknown side block type: {side_block_type}")
