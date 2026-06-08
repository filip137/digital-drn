from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from digital_drn.models.tokenwise import TokenwiseDRNMLP

from .config import GPT2Config
from .drn import PureDRNSideBlock, SideDRNBlock
from .model_gpt2 import CausalSelfAttention, GPT2LMHeadModel, LayerNorm, MLP


_SIDE_BLOCK_ALIASES = {
    "transformer": "digital",
    "digital": "digital",
    "drn": "drn_hybrid_attn",
    "drn_hybrid_attn": "drn_hybrid_attn",
    "pure_drn": "drn_pure",
    "drn_pure": "drn_pure",
}
_LADDER_INJECTION_MODES = {"pre_drn_mix", "post_drn_residual"}
_BACKBONE_TAP_KINDS = {"block_output", "attn_residual", "mlp_input", "mlp_delta"}
_INITIAL_SIDE_STATE_MODES = {"gated", "full_tap"}


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
        ladder_injection_mode: str = "pre_drn_mix",
        backbone_tap_kind: str = "block_output",
        post_drn_feedback: bool = True,
        post_drn_alpha_init: float | None = None,
        post_ladder_lambda_init: float | None = None,
        residual_gamma_init: float | None = None,
        logit_gate_alpha_init: float = 0.0,
        tap_indices: list[int] | tuple[int, ...] | None = None,
        initial_side_state_mode: str = "gated",
        drn_signed_drive: bool = True,
        drn_hidden_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if reduction_factor <= 0:
            raise ValueError("reduction_factor must be strictly positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive.")
        if output_mode not in {"side_only", "residual_logits", "gated_logits"}:
            raise ValueError("output_mode must be 'side_only', 'residual_logits', or 'gated_logits'.")
        if side_block_type not in _SIDE_BLOCK_ALIASES:
            valid_side_blocks = "', '".join(sorted(_SIDE_BLOCK_ALIASES))
            raise ValueError(f"side_block_type must be one of '{valid_side_blocks}'.")
        if ladder_injection_mode not in _LADDER_INJECTION_MODES:
            raise ValueError("ladder_injection_mode must be 'pre_drn_mix' or 'post_drn_residual'.")
        if initial_side_state_mode not in _INITIAL_SIDE_STATE_MODES:
            valid_init_modes = "', '".join(sorted(_INITIAL_SIDE_STATE_MODES))
            raise ValueError(f"initial_side_state_mode must be one of '{valid_init_modes}'.")
        if backbone_tap_kind not in _BACKBONE_TAP_KINDS:
            valid_taps = "', '".join(sorted(_BACKBONE_TAP_KINDS))
            raise ValueError(f"backbone_tap_kind must be one of '{valid_taps}'.")
        if backbone_tap_kind != "block_output":
            raise NotImplementedError("Only backbone_tap_kind='block_output' is implemented.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

        self.reduction_factor = int(reduction_factor)
        self.temperature = float(temperature)
        self.output_mode = output_mode
        self.side_block_type = _SIDE_BLOCK_ALIASES[side_block_type]
        self.ladder_injection_mode = ladder_injection_mode
        self.backbone_tap_kind = backbone_tap_kind
        self.post_drn_feedback = bool(post_drn_feedback)
        self.initial_side_state_mode = initial_side_state_mode
        self.drn_signed_drive = bool(drn_signed_drive)
        if drn_hidden_multiplier <= 0:
            raise ValueError("drn_hidden_multiplier must be strictly positive.")
        self.drn_hidden_multiplier = int(drn_hidden_multiplier)
        self.d_model = int(base.config.n_embd)
        self.num_base_layers = int(base.config.n_layer)
        explicit_tap_indices = _validate_tap_indices(tap_indices, self.num_base_layers)
        if explicit_tap_indices is not None and num_side_layers is not None:
            if len(explicit_tap_indices) != int(num_side_layers):
                raise ValueError("num_side_layers must match len(tap_indices) when both are provided.")
        self.num_side_layers = (
            len(explicit_tap_indices)
            if explicit_tap_indices is not None
            else int(num_side_layers if num_side_layers is not None else self.num_base_layers)
        )
        if not 1 <= self.num_side_layers <= self.num_base_layers:
            raise ValueError("num_side_layers must lie in [1, base.config.n_layer].")
        if self.d_model % self.reduction_factor != 0:
            raise ValueError("base hidden size must be divisible by reduction_factor.")
        self.d_side = self.d_model // self.reduction_factor
        self.base_indices = (
            explicit_tap_indices
            if explicit_tap_indices is not None
            else _layer_indices(self.num_base_layers, self.num_side_layers)
        )

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
                _build_side_block(
                    side_config,
                    self.side_block_type,
                    drn_iter=drn_iter,
                    drn_damping=drn_damping,
                    drn_signed_drive=self.drn_signed_drive,
                    drn_hidden_multiplier=self.drn_hidden_multiplier,
                )
                for _ in range(self.num_side_layers)
            ]
        )
        if self.ladder_injection_mode == "pre_drn_mix":
            self.gates = nn.Parameter(torch.zeros(self.num_side_layers + 1))
        else:
            self.register_parameter("gates", None)
        self.up_proj = nn.Linear(self.d_side, self.d_model)
        self.side_ln_f = LayerNorm(self.d_side, bias=base.config.bias)
        if self.output_mode == "residual_logits":
            if residual_gamma_init is None:
                residual_gamma_init = 1.0e-3 if self.ladder_injection_mode == "post_drn_residual" else 0.1
            self.gamma = nn.Parameter(torch.tensor(float(residual_gamma_init)))
        else:
            self.register_parameter("gamma", None)
        if self.output_mode == "gated_logits":
            self.logit_gate = nn.Parameter(torch.tensor(float(logit_gate_alpha_init)))
        else:
            self.register_parameter("logit_gate", None)
        if self.ladder_injection_mode == "post_drn_residual":
            if post_drn_alpha_init is None:
                post_drn_alpha_init = 0.1 if self.output_mode == "residual_logits" else 1.0
            if post_ladder_lambda_init is None:
                post_ladder_lambda_init = 0.1 if self.output_mode == "residual_logits" else 1.0
            self.tap_norms = nn.ModuleList(
                [LayerNorm(self.d_side, bias=base.config.bias) for _ in range(self.num_side_layers + 1)]
            )
            self.tap_adapters = nn.ModuleList([nn.Identity() for _ in range(self.num_side_layers + 1)])
            if self.side_block_type == "digital":
                self.register_parameter("post_drn_alphas", None)
            else:
                self.post_drn_alphas = nn.Parameter(
                    torch.full((self.num_side_layers,), float(post_drn_alpha_init))
                )
            self.post_ladder_lambdas = nn.Parameter(
                torch.full((self.num_side_layers + 1,), float(post_ladder_lambda_init))
            )
        else:
            self.tap_norms = nn.ModuleList()
            self.tap_adapters = nn.ModuleList()
            self.register_parameter("post_drn_alphas", None)
            self.register_parameter("post_ladder_lambdas", None)
        self.last_ladder_diagnostics: dict[str, float | str | bool] = {
            "ladder_injection_mode": self.ladder_injection_mode,
            "post_drn_feedback": self.post_drn_feedback,
            "initial_side_state_mode": self.initial_side_state_mode,
            "drn_signed_drive": self.drn_signed_drive,
            "drn_hidden_multiplier": self.drn_hidden_multiplier,
        }

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

    def collect_ladder_diagnostics(self) -> dict[str, float | str | bool]:
        return dict(self.last_ladder_diagnostics)

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

        if self.ladder_injection_mode == "post_drn_residual":
            final_side_state, side_hidden_states = self._forward_post_drn_residual(
                hidden_states,
                return_hidden_states=return_hidden_states,
            )
        else:
            final_side_state, side_hidden_states = self._forward_pre_drn_mix(
                hidden_states,
                return_hidden_states=return_hidden_states,
            )

        side_h = self.side_ln_f(final_side_state)
        side_d = self.up_proj(side_h)
        side_logits = self.base.lm_head(side_d)
        if self.output_mode == "residual_logits":
            logits = base_logits + self.gamma * side_logits
        elif self.output_mode == "gated_logits":
            rho = torch.sigmoid(self.logit_gate / self.temperature)
            logits = rho * side_logits + (1.0 - rho) * base_logits
        else:
            logits = side_logits

        self.last_ladder_diagnostics["z_side_norm"] = float(side_logits.detach().norm().item())
        if self.gamma is not None:
            self.last_ladder_diagnostics["gamma"] = float(self.gamma.detach().item())
        if self.logit_gate is not None:
            rho = torch.sigmoid(self.logit_gate.detach() / self.temperature)
            self.last_ladder_diagnostics["logit_gate_alpha"] = float(self.logit_gate.detach().item())
            self.last_ladder_diagnostics["logit_gate_rho"] = float(rho.item())

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return {"logits": logits, "loss": loss, "hidden_states": side_hidden_states}

    def _forward_pre_drn_mix(
        self,
        hidden_states: list[torch.Tensor],
        *,
        return_hidden_states: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if self.gates is None:
            raise RuntimeError("pre_drn_mix requires ladder gates.")
        if self.initial_side_state_mode == "full_tap":
            s = self.down_projs[0](hidden_states[0])
        else:
            gate0 = torch.sigmoid(self.gates[0] / self.temperature)
            s = gate0 * self.down_projs[0](hidden_states[0])
        side_hidden_states: list[torch.Tensor] | None = [s] if return_hidden_states else None

        for j, block in enumerate(self.side_blocks):
            base_idx = self.base_indices[j]
            tap_proj = self.down_projs[j + 1](hidden_states[base_idx])
            mu = torch.sigmoid(self.gates[j + 1] / self.temperature)
            mix = mu * tap_proj + (1.0 - mu) * s
            s = block(mix)
            if side_hidden_states is not None:
                side_hidden_states.append(s)

        self.last_ladder_diagnostics = {
            "ladder_injection_mode": self.ladder_injection_mode,
            "post_drn_feedback": self.post_drn_feedback,
            "initial_side_state_mode": self.initial_side_state_mode,
            "drn_signed_drive": self.drn_signed_drive,
            "drn_hidden_multiplier": self.drn_hidden_multiplier,
        }
        return s, side_hidden_states

    def _forward_post_drn_residual(
        self,
        hidden_states: list[torch.Tensor],
        *,
        return_hidden_states: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if self.post_ladder_lambdas is None:
            raise RuntimeError("post_drn_residual scales are not initialized.")
        if self.side_block_type != "digital" and self.post_drn_alphas is None:
            raise RuntimeError("post_drn_residual DRN side blocks require alpha scales.")

        c0 = self._adapt_tap(0, hidden_states[0])
        s = self.post_ladder_lambdas[0] * c0
        read_state = s
        side_hidden_states: list[torch.Tensor] | None = [read_state] if return_hidden_states else None
        drn_update_ratios: list[float] = []
        tap_update_ratios: list[float] = []

        for j, block in enumerate(self.side_blocks):
            base_idx = self.base_indices[j]
            c_j = self._adapt_tap(j + 1, hidden_states[base_idx])
            q_j = s
            if self.side_block_type == "digital":
                s_drn_j = block(q_j)
            else:
                s_drn_j, q_j, d_j = block.forward_post_drn_residual(q_j, self.post_drn_alphas[j])
                drn_update_ratios.append(_safe_norm_ratio(self.post_drn_alphas[j] * d_j, q_j))
            ladder_update = self.post_ladder_lambdas[j + 1] * c_j
            s_combined = s_drn_j + ladder_update
            read_state = s_combined
            s = s_combined if self.post_drn_feedback else s_drn_j
            tap_update_ratios.append(_safe_norm_ratio(ladder_update, s_drn_j))
            if side_hidden_states is not None:
                side_hidden_states.append(read_state)

        self.last_ladder_diagnostics = {
            "ladder_injection_mode": self.ladder_injection_mode,
            "post_drn_feedback": self.post_drn_feedback,
            "initial_side_state_mode": self.initial_side_state_mode,
            "drn_signed_drive": self.drn_signed_drive,
            "drn_hidden_multiplier": self.drn_hidden_multiplier,
            "mean_abs_lambda": float(self.post_ladder_lambdas.detach().abs().mean().item()),
            "mean_drn_update_to_q_norm": _mean_or_nan(drn_update_ratios),
            "mean_ladder_update_to_s_drn_norm": _mean_or_nan(tap_update_ratios),
        }
        if self.post_drn_alphas is not None:
            self.last_ladder_diagnostics["mean_abs_alpha"] = float(
                self.post_drn_alphas.detach().abs().mean().item()
            )
        return read_state, side_hidden_states

    def _adapt_tap(self, tap_slot: int, tap: torch.Tensor) -> torch.Tensor:
        tap_proj = self.down_projs[tap_slot](tap.detach())
        return self.tap_adapters[tap_slot](self.tap_norms[tap_slot](tap_proj))


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
    if method not in {"magnitude", "magnitude_attn", "magnitude_pruned"}:
        raise ValueError("structural init method must be 'magnitude', 'magnitude_attn', or 'magnitude_pruned'.")
    with torch.no_grad():
        cols = _magnitude_channel_indices(base_model, side_model.d_side)
        _init_selector_projections(side_model, cols)
        if method == "magnitude_attn":
            _copy_pruned_attention_side_blocks(side_model, base_model, cols)
        elif method == "magnitude_pruned":
            _copy_pruned_digital_side_blocks(side_model, base_model, cols)


def _validate_tap_indices(
    tap_indices: list[int] | tuple[int, ...] | None,
    num_base_layers: int,
) -> list[int] | None:
    if tap_indices is None:
        return None
    indices = [int(index) for index in tap_indices]
    if not indices:
        raise ValueError("tap_indices must not be empty.")
    if indices != sorted(indices):
        raise ValueError("tap_indices must be sorted in ascending order.")
    if len(set(indices)) != len(indices):
        raise ValueError("tap_indices must not contain duplicates.")
    bad = [index for index in indices if index < 1 or index > num_base_layers]
    if bad:
        raise ValueError(f"tap_indices must lie in [1, {num_base_layers}], got {bad}.")
    return indices


def _layer_indices(num_base_layers: int, num_side_layers: int) -> list[int]:
    return [max(1, round((j + 1) * num_base_layers / num_side_layers)) for j in range(num_side_layers)]


def _magnitude_channel_indices(base_model: GPT2LMHeadModel, d_side: int) -> torch.Tensor:
    row_scores = base_model.transformer.wte.weight.detach().abs().sum(dim=0)
    return torch.topk(row_scores, k=d_side).indices.sort().values


def _init_selector_projections(side_model: LadderSideGPT2, cols: torch.Tensor) -> None:
    for down_proj in side_model.down_projs:
        row_idx = torch.arange(side_model.d_side, device=down_proj.weight.device)
        cols_on_device = cols.to(down_proj.weight.device)
        down_proj.weight.zero_()
        if down_proj.bias is not None:
            down_proj.bias.zero_()
        down_proj.weight[row_idx, cols_on_device] = 1.0

    row_idx = cols.to(side_model.up_proj.weight.device)
    col_idx = torch.arange(side_model.d_side, device=side_model.up_proj.weight.device)
    side_model.up_proj.weight.zero_()
    if side_model.up_proj.bias is not None:
        side_model.up_proj.bias.zero_()
    side_model.up_proj.weight[row_idx, col_idx] = 1.0


def _copy_pruned_digital_side_blocks(
    side_model: LadderSideGPT2,
    base_model: GPT2LMHeadModel,
    cols: torch.Tensor,
) -> None:
    if side_model.side_block_type != "digital":
        raise ValueError("magnitude_pruned structural init is only available for digital side blocks.")

    _copy_layer_norm(base_model.transformer.ln_f, side_model.side_ln_f, cols)
    mlp_cols = _expanded_mlp_indices(cols, side_model.d_model)
    qkv_cols = _expanded_qkv_indices(cols, side_model.d_model)

    for side_block, base_hidden_index in zip(side_model.side_blocks, side_model.base_indices):
        if not isinstance(side_block, SideTransformerBlock):
            raise ValueError("magnitude_pruned structural init expects SideTransformerBlock modules.")
        base_block = base_model.transformer.h[base_hidden_index - 1]
        _copy_layer_norm(base_block.ln_1, side_block.ln_1, cols)
        _copy_layer_norm(base_block.ln_2, side_block.ln_2, cols)
        _copy_linear(base_block.attn.c_attn, side_block.attn.c_attn, out_idx=qkv_cols, in_idx=cols)
        _copy_linear(base_block.attn.c_proj, side_block.attn.c_proj, out_idx=cols, in_idx=cols)
        _copy_linear(base_block.mlp.c_fc, side_block.mlp.c_fc, out_idx=mlp_cols, in_idx=cols)
        _copy_linear(base_block.mlp.c_proj, side_block.mlp.c_proj, out_idx=cols, in_idx=mlp_cols)


def _copy_pruned_attention_side_blocks(
    side_model: LadderSideGPT2,
    base_model: GPT2LMHeadModel,
    cols: torch.Tensor,
) -> None:
    if side_model.side_block_type != "drn_hybrid_attn":
        raise ValueError("magnitude_attn structural init is only available for DRN hybrid-attention side blocks.")

    _copy_layer_norm(base_model.transformer.ln_f, side_model.side_ln_f, cols)
    qkv_cols = _expanded_qkv_indices(cols, side_model.d_model)

    for side_block, base_hidden_index in zip(side_model.side_blocks, side_model.base_indices):
        if not isinstance(side_block, SideDRNBlock):
            raise ValueError("magnitude_attn structural init expects SideDRNBlock modules.")
        base_block = base_model.transformer.h[base_hidden_index - 1]
        _copy_layer_norm(base_block.ln_1, side_block.ln_1, cols)
        _copy_layer_norm(base_block.ln_2, side_block.ln_2, cols)
        _copy_linear(base_block.attn.c_attn, side_block.attn.c_attn, out_idx=qkv_cols, in_idx=cols)
        _copy_linear(base_block.attn.c_proj, side_block.attn.c_proj, out_idx=cols, in_idx=cols)


def _expanded_qkv_indices(cols: torch.Tensor, d_model: int) -> torch.Tensor:
    return torch.cat([cols + offset * d_model for offset in range(3)])


def _expanded_mlp_indices(cols: torch.Tensor, d_model: int) -> torch.Tensor:
    return torch.cat([cols + offset * d_model for offset in range(4)])


def _copy_layer_norm(src: LayerNorm, dst: LayerNorm, idx: torch.Tensor) -> None:
    dst.weight.copy_(_select_1d(src.weight, idx, dst.weight))
    if dst.bias is not None:
        if src.bias is None:
            dst.bias.zero_()
        else:
            dst.bias.copy_(_select_1d(src.bias, idx, dst.bias))


def _copy_linear(
    src: nn.Linear,
    dst: nn.Linear,
    *,
    out_idx: torch.Tensor,
    in_idx: torch.Tensor,
) -> None:
    weight = src.weight.detach()
    weight = weight.index_select(0, out_idx.to(weight.device))
    weight = weight.index_select(1, in_idx.to(weight.device))
    if tuple(weight.shape) != tuple(dst.weight.shape):
        raise ValueError(f"Cannot copy pruned weight with shape {tuple(weight.shape)} into {tuple(dst.weight.shape)}.")
    dst.weight.copy_(weight.to(device=dst.weight.device, dtype=dst.weight.dtype))
    if dst.bias is not None:
        if src.bias is None:
            dst.bias.zero_()
        else:
            bias = src.bias.detach().index_select(0, out_idx.to(src.bias.device))
            dst.bias.copy_(bias.to(device=dst.bias.device, dtype=dst.bias.dtype))


def _select_1d(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return src.detach().index_select(0, idx.to(src.device)).to(device=dst.device, dtype=dst.dtype)


def _build_side_block(
    side_config: GPT2Config,
    side_block_type: str,
    drn_iter: int,
    drn_damping: float,
    drn_signed_drive: bool = True,
    drn_hidden_multiplier: int = 4,
) -> nn.Module:
    if side_block_type == "digital":
        return SideTransformerBlock(side_config)
    if side_block_type == "drn_hybrid_attn":
        return SideDRNBlock(
            side_config,
            drn_iter=drn_iter,
            drn_damping=drn_damping,
            signed_drive=drn_signed_drive,
            hidden_multiplier=drn_hidden_multiplier,
        )
    if side_block_type == "drn_pure":
        return PureDRNSideBlock(
            side_config,
            drn_iter=drn_iter,
            drn_damping=drn_damping,
            signed_drive=drn_signed_drive,
            hidden_multiplier=drn_hidden_multiplier,
        )
    raise ValueError(f"Unknown side block type: {side_block_type}")


def _safe_norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    with torch.no_grad():
        denom = float(denominator.detach().norm().item())
        if denom <= 1.0e-12:
            return float("nan")
        return float(numerator.detach().norm().item() / denom)


def _mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if value == value]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))
