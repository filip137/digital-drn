from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models import DRNGPTConfig, SmallDRNGPT
from ..training.mqar import IGNORE_INDEX, SmallGPTMQARDataset


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def get_lr(step: int, *, lr: float, warmup_steps: int, max_steps: int, lr_schedule: str) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return lr * step / warmup_steps
    if lr_schedule == "constant":
        return lr
    if max_steps <= warmup_steps:
        return lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * lr * (1.0 + math.cos(math.pi * progress))


def cycle(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def configure_optimizer(model: SmallDRNGPT, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.optimizer_param_groups(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )


@torch.no_grad()
def evaluate(
    model: SmallDRNGPT,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    num_iterations: int | None,
) -> dict[str, float]:
    model.eval()
    device_type = device.type
    amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    loss_sum = 0.0
    query_count = 0
    correct_count = 0
    exact_count = 0
    example_count = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(input_ids, reset=True, num_iterations=num_iterations)
            batch_loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        model.detach_state_()

        mask = labels != IGNORE_INDEX
        pred = logits.argmax(dim=-1)
        correct = (pred == labels) & mask
        loss_sum += float(batch_loss_sum.item())
        query_count += int(mask.sum().item())
        correct_count += int(correct.sum().item())
        exact_count += int(((pred == labels) | (~mask)).all(dim=1).sum().item())
        example_count += int(labels.size(0))

    return {
        "loss": loss_sum / max(query_count, 1),
        "query_acc": correct_count / max(query_count, 1),
        "exact_acc": exact_count / max(example_count, 1),
    }


def _make_run_dir(root: str | Path, *, timestamp: bool) -> Path:
    root_path = Path(root).expanduser()
    if not timestamp:
        root_path.mkdir(parents=True, exist_ok=True)
        return root_path
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{socket.gethostname()}"
    run_dir = root_path / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digital-drn-mqar-benchmark",
        description="Train SmallDRNGPT on the small_gpt-compatible MQAR benchmark.",
    )

    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_kv_pairs", type=int, default=4)
    parser.add_argument("--train_examples", type=int, default=50_000)
    parser.add_argument("--val_examples", type=int, default=5_000)
    parser.add_argument("--train_seed", type=int, default=123)
    parser.add_argument("--val_seed", type=int, default=456)
    parser.add_argument("--power_a", type=float, default=0.01)
    parser.add_argument("--random_non_queries", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tie_weights", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--drn_hidden_dim", type=int, default=None)
    parser.add_argument("--drn_num_iterations", type=int, default=6)
    parser.add_argument("--drn_mode", type=str, default="asynchronous")
    parser.add_argument("--drn_non_linearity", type=str, default="perfect_diode")
    parser.add_argument("--drn_ff_activation", type=str, default="identity")
    parser.add_argument("--drn_signed_drive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drn_weight_gains", type=float, default=0.1)
    parser.add_argument("--drn_bias_gain", type=float, default=0.0)
    parser.add_argument("--drn_voltage_amp", type=float, default=1.0)
    parser.add_argument("--drn_current_amp", type=float, default=1.0)
    parser.add_argument("--drn_learn_drive_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drn_init_drive_scale", type=float, default=1.0)
    parser.add_argument("--drn_weight_min", type=float, default=None)
    parser.add_argument("--drn_weight_max", type=float, default=None)
    parser.add_argument("--drn_weight_init_mode", type=str, default="kaiming_uniform")
    parser.add_argument("--drn_reset_each_forward", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--lr_schedule", type=str, choices=("constant", "cosine"), default="constant")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--train_num_iterations", type=int, default=6)
    parser.add_argument("--eval_num_iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="simulation_results/mqar_small_gpt_drn")
    parser.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _config_payload(args: argparse.Namespace, model_cfg: DRNGPTConfig) -> dict[str, Any]:
    return {
        "data": {
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "num_kv_pairs": args.num_kv_pairs,
            "train_examples": args.train_examples,
            "val_examples": args.val_examples,
            "train_seed": args.train_seed,
            "val_seed": args.val_seed,
            "power_a": args.power_a,
            "random_non_queries": args.random_non_queries,
        },
        "model": asdict(model_cfg),
        "train": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "lr_schedule": args.lr_schedule,
            "grad_clip": args.grad_clip,
            "eval_every": args.eval_every,
            "train_num_iterations": args.train_num_iterations,
            "eval_num_iterations": args.eval_num_iterations,
            "seed": args.seed,
            "device": args.device,
            "num_workers": args.num_workers,
            "amp": args.amp,
            "out_dir": args.out_dir,
            "save_best": args.save_best,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    seed_all(int(args.seed))
    device = resolve_device(args.device)
    if device.type != "cuda":
        print(f"warning: running on {device}; pass --device cuda for the intended benchmark.", flush=True)
    pin_memory = device.type == "cuda"
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_data = SmallGPTMQARDataset(
        vocab_size=args.vocab_size,
        num_examples=args.train_examples,
        seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        seed=args.train_seed,
        power_a=args.power_a,
        random_non_queries=args.random_non_queries,
    )
    val_data = SmallGPTMQARDataset(
        vocab_size=args.vocab_size,
        num_examples=args.val_examples,
        seq_len=args.seq_len,
        num_kv_pairs=args.num_kv_pairs,
        seed=args.val_seed,
        power_a=args.power_a,
        random_non_queries=args.random_non_queries,
    )

    loader_generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    model_cfg = DRNGPTConfig(
        vocab_size=int(args.vocab_size),
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        mlp_ratio=int(args.mlp_ratio),
        dropout=float(args.dropout),
        bias=bool(args.bias),
        tie_weights=bool(args.tie_weights),
        drn_hidden_dim=args.drn_hidden_dim,
        drn_num_iterations=int(args.drn_num_iterations),
        drn_mode=str(args.drn_mode),
        drn_non_linearity=str(args.drn_non_linearity),
        drn_ff_activation=args.drn_ff_activation,
        drn_signed_drive=bool(args.drn_signed_drive),
        drn_weight_gains=float(args.drn_weight_gains),
        drn_bias_gain=float(args.drn_bias_gain),
        drn_voltage_amp=float(args.drn_voltage_amp),
        drn_current_amp=float(args.drn_current_amp),
        drn_learn_drive_scale=bool(args.drn_learn_drive_scale),
        drn_init_drive_scale=float(args.drn_init_drive_scale),
        drn_weight_min=args.drn_weight_min,
        drn_weight_max=args.drn_weight_max,
        drn_weight_init_mode=str(args.drn_weight_init_mode),
        drn_reset_each_forward=bool(args.drn_reset_each_forward),
    )
    model = SmallDRNGPT(model_cfg)
    model.set_device(device)
    model.enable_resistive_grad_()
    optimizer = configure_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_dir = _make_run_dir(args.out_dir, timestamp=bool(args.timestamp))
    config_payload = _config_payload(args, model_cfg)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "cwd": str(Path.cwd()),
                "checkpoint_dir": str(run_dir.resolve()),
                "output_dir": str(Path(args.out_dir).expanduser().resolve()),
                "entrypoint": "digital-drn-mqar-benchmark",
            },
            handle,
            indent=2,
        )

    print(
        f"device={device} params={model.num_parameters():,} "
        f"train_examples={len(train_data):,} val_examples={len(val_data):,} "
        f"run_dir={run_dir.resolve()}",
        flush=True,
    )

    train_iter = cycle(train_loader)
    train_losses: list[float] = []
    best_query_acc = -1.0
    history: list[dict[str, float | int]] = []
    device_type = device.type
    amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    for step in range(1, int(args.max_steps) + 1):
        model.train()
        lr = get_lr(
            step,
            lr=float(args.lr),
            warmup_steps=int(args.warmup_steps),
            max_steps=int(args.max_steps),
            lr_schedule=str(args.lr_schedule),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids, labels = next(train_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(input_ids, reset=True, num_iterations=args.train_num_iterations)
            loss = masked_cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.optimizer_tensors(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        model.clamp_resistive_params_()
        model.detach_state_()
        train_losses.append(float(loss.item()))

        should_eval = step == 1 or step % int(args.eval_every) == 0 or step == int(args.max_steps)
        if should_eval:
            val = evaluate(
                model,
                val_loader,
                device,
                amp_enabled=amp_enabled,
                num_iterations=args.eval_num_iterations,
            )
            train_loss = sum(train_losses) / len(train_losses)
            train_losses.clear()
            epoch = step * int(args.batch_size) / int(args.train_examples)
            row = {
                "step": step,
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_query_acc": val["query_acc"],
                "val_exact_acc": val["exact_acc"],
            }
            history.append(row)
            with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

            print(
                f"step={step:05d} lr={lr:.3e} train_loss={train_loss:.4f} "
                f"val_loss={val['loss']:.4f} val_query_acc={val['query_acc']:.4f} "
                f"val_exact_acc={val['exact_acc']:.4f}",
                flush=True,
            )

            if val["query_acc"] > best_query_acc:
                best_query_acc = val["query_acc"]
                if bool(args.save_best):
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "resistive_state": [
                                {
                                    "name": getattr(param, "name", f"param_{idx}"),
                                    "state": param.state.detach().cpu(),
                                }
                                for idx, param in enumerate(model.resistive_params())
                            ],
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config_payload,
                            "val_metrics": val,
                        },
                        run_dir / "best.pt",
                    )

    print(f"done: best_val_query_acc={best_query_acc:.4f} run_dir={run_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
