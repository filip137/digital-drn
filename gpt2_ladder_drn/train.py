from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gpt2_ladder_drn.config import DebugGPT2Config, GPT2Config
from gpt2_ladder_drn.data import load_tiny_shakespeare
from gpt2_ladder_drn.eval import evaluate
from gpt2_ladder_drn.generate import generate
from gpt2_ladder_drn.ladder import LadderSideGPT2
from gpt2_ladder_drn.lora import apply_lora
from gpt2_ladder_drn.model_gpt2 import GPT2LMHeadModel
from gpt2_ladder_drn.utils import (
    cuda_peak_memory_mb,
    cycle,
    get_device,
    save_json,
    set_seed,
    timestamp,
)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = _make_config(args)
    tokenizer = "char" if args.debug and args.pretrained is None and args.tokenizer == "gpt2" else args.tokenizer
    train_dataset, val_dataset, encode, decode = load_tiny_shakespeare(
        args.data,
        tokenizer=tokenizer,
        train_frac=args.train_frac,
        block_size=args.block_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    train_iter = cycle(train_loader)

    model = _build_model(args, config).to(device)
    if hasattr(model, "enable_resistive_grad_"):
        model.enable_resistive_grad_(True)
    trainable_params = _trainable_tensors(model)
    output_dir = Path(args.output_dir) / f"{args.mode}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "mode": args.mode,
        "config": config.__dict__,
        "tokenizer": tokenizer,
        "trainable_params": _count_tensors(trainable_params),
        "total_params": _count_tensors(_all_model_tensors(model)),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "run_metadata.json", metadata)

    if args.mode == "eval_base":
        metrics = evaluate(model, val_loader, device, max_batches=args.eval_iters)
        save_json(output_dir / "eval.json", metrics)
        print(metrics)
        return

    if not trainable_params:
        raise RuntimeError("No trainable parameters for training mode.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    last_log_time = time.time()
    last_log_tokens = 0
    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, max_batches=args.eval_iters)
            best_val = min(best_val, val_metrics["loss"])
            sample = _sample_text(model, encode, decode, device, args)
            _save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, step, args, val_metrics)
            (output_dir / f"sample_step_{step:06d}.txt").write_text(sample, encoding="utf-8")
            elapsed = max(1.0e-6, time.time() - last_log_time)
            tokens = last_log_tokens
            last_log_time = time.time()
            last_log_tokens = 0
            print(
                {
                    "step": step,
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["ppl"],
                    "tokens_per_second": tokens / elapsed,
                    "peak_memory_mb": cuda_peak_memory_mb(),
                }
            )
            if step == args.max_steps:
                break

        model.train()
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)
        out = model(x, targets=y)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        if hasattr(model, "clamp_resistive_params_"):
            model.clamp_resistive_params_()
        if hasattr(model, "detach_state_"):
            model.detach_state_()
        last_log_tokens += x.numel()

    final_metrics = {"best_val_loss": best_val, "peak_memory_mb": cuda_peak_memory_mb()}
    save_json(output_dir / "final_metrics.json", final_metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full_finetune", "lora", "lst", "lst_drn", "eval_base"], required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tokenizer", choices=["gpt2", "char"], default="gpt2")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--output_dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="c_attn,c_proj")
    parser.add_argument("--lst_reduction", type=int, default=8)
    parser.add_argument("--lst_side_layers", type=int, default=None)
    parser.add_argument("--lst_temperature", type=float, default=0.1)
    parser.add_argument("--lst_output_mode", choices=["side_only", "residual_logits"], default="side_only")
    parser.add_argument("--lst_side_block_type", choices=["transformer", "drn", "pure_drn"], default=None)
    parser.add_argument("--drn_iter", type=int, default=4)
    parser.add_argument("--drn_damping", type=float, default=0.5)
    parser.add_argument("--sample_prompt", default="To be, or not to be")
    parser.add_argument("--sample_tokens", type=int, default=80)
    return parser.parse_args()


def _make_config(args: argparse.Namespace) -> GPT2Config:
    if args.debug:
        cfg = DebugGPT2Config(block_size=args.block_size)
        return GPT2Config(**cfg.__dict__)
    return GPT2Config(block_size=max(args.block_size, 1))


def _build_model(args: argparse.Namespace, config: GPT2Config) -> torch.nn.Module:
    if args.pretrained and not args.debug:
        base = GPT2LMHeadModel.from_hf_gpt2(args.pretrained)
    else:
        base = GPT2LMHeadModel(config)

    if args.mode == "full_finetune":
        return base
    if args.mode == "eval_base":
        for param in base.parameters():
            param.requires_grad = False
        return base
    if args.mode == "lora":
        targets = tuple(item.strip() for item in args.lora_targets.split(",") if item.strip())
        return apply_lora(base, target_modules=targets, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    if args.mode in {"lst", "lst_drn"}:
        side_block_type = args.lst_side_block_type
        if side_block_type is None:
            side_block_type = "transformer" if args.mode == "lst" else "drn"
        return LadderSideGPT2(
            base,
            reduction_factor=args.lst_reduction,
            num_side_layers=args.lst_side_layers,
            temperature=args.lst_temperature,
            output_mode=args.lst_output_mode,
            side_block_type=side_block_type,
            drn_iter=args.drn_iter,
            drn_damping=args.drn_damping,
        )
    raise ValueError(f"Unsupported mode: {args.mode}")


def _all_model_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    tensors = list(model.parameters())
    if hasattr(model, "resistive_param_states"):
        tensors.extend(model.resistive_param_states())
    return _dedupe_tensors(tensors)


def _trainable_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    if hasattr(model, "optimizer_tensors"):
        tensors = list(model.optimizer_tensors())
    else:
        tensors = list(model.parameters())
    return [tensor for tensor in _dedupe_tensors(tensors) if tensor.requires_grad]


def _dedupe_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    unique = []
    seen = set()
    for tensor in tensors:
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tensor)
    return unique


def _count_tensors(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)


def _sample_text(model, encode, decode, device: torch.device, args: argparse.Namespace) -> str:
    input_ids = torch.tensor([encode(args.sample_prompt)], dtype=torch.long, device=device)
    output = generate(model, input_ids, max_new_tokens=args.sample_tokens, temperature=0.8, top_k=50)
    return decode(output[0].detach().cpu())


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
        "metrics": metrics,
    }
    if hasattr(model, "named_resistive_parameters"):
        checkpoint["resistive_parameters"] = {
            name: tensor.detach().cpu() for name, tensor in model.named_resistive_parameters()
        }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
