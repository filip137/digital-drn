from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .calibration import RunningScalarStats, scales_from_calibration
from .data import load_explicit_text_datasets, load_text_datasets
from .model import load_opt_causal_lm
from .teacher import TeacherLayerActivations, collect_teacher_layer_activations, default_probe_layers, opt_num_layers


CACHE_VERSION = 1
ACTIVATION_KEYS = ("z", "r", "a", "h_next", "next_ln")
TRAINING_KEYS = ("z", "r", "a", "h_next")


class CachedLayerActivationDataset(Dataset):
    """Layer-specific view over a sharded teacher activation cache."""

    def __init__(self, cache_dir: str | Path, *, split: str, layer_index: int, max_open_shards: int = 8) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.layer_index = int(layer_index)
        self.max_open_shards = max(1, int(max_open_shards))
        self.metadata = load_cache_metadata(self.cache_dir)
        _validate_cache_version(self.metadata)

        split_meta = self.metadata.get("splits", {}).get(self.split)
        if split_meta is None:
            raise ValueError(f"Cache has no split '{self.split}'.")
        if str(self.layer_index) not in self.metadata.get("layers", []):
            available = ", ".join(self.metadata.get("layers", []))
            raise ValueError(f"Cache has no layer {self.layer_index}; available layers: {available}.")

        self._entries: list[tuple[Path, int]] = []
        for shard_meta in split_meta.get("shards", []):
            rel_path = shard_meta["path"]
            path = self.cache_dir / rel_path
            num_sequences = int(shard_meta["num_sequences"])
            self._entries.extend((path, row_idx) for row_idx in range(num_sequences))
        if not self._entries:
            raise ValueError(f"Cache split '{self.split}' is empty.")
        self._shards: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, row_idx = self._entries[int(index)]
        shard = self._load_shard(path)
        layer_payload = shard["layers"][str(self.layer_index)]
        return {key: layer_payload[key][row_idx].clone() for key in ACTIVATION_KEYS if key in layer_payload}

    def _load_shard(self, path: Path) -> dict[str, Any]:
        cached = self._shards.get(path)
        if cached is not None:
            self._shards.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._shards[path] = payload
        self._shards.move_to_end(path)
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return payload


def batch_to_activations(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> TeacherLayerActivations:
    return TeacherLayerActivations(
        z=batch["z"].to(device=device, dtype=dtype, non_blocking=True),
        r=batch["r"].to(device=device, dtype=dtype, non_blocking=True),
        a=batch["a"].to(device=device, dtype=dtype, non_blocking=True),
        h_next=batch["h_next"].to(device=device, dtype=dtype, non_blocking=True),
        next_ln=(
            batch["next_ln"].to(device=device, dtype=dtype, non_blocking=True)
            if "next_ln" in batch
            else None
        ),
    )


def write_activation_cache(
    *,
    teacher: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_indices: list[int],
    output_dir: str | Path,
    device: torch.device,
    model_name: str,
    block_size: int,
    source: str,
    test_loader: DataLoader | None = None,
    sequence_stride: int = 1,
    mlp_input_mode: str = "normalized",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    dtype: str = "float16",
    max_quantile_samples: int = 250_000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be one of: float16, bfloat16, float32.")
    tensor_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    layer_indices = sorted({int(idx) for idx in layer_indices})
    stats = {
        layer_idx: {
            "z": RunningScalarStats(max_quantile_samples=max_quantile_samples),
            "r": RunningScalarStats(max_quantile_samples=max_quantile_samples),
        }
        for layer_idx in layer_indices
    }

    teacher.to(device)
    teacher.eval()
    splits: dict[str, Any] = {}
    split_specs: list[tuple[str, DataLoader, int | None]] = [
        ("train", train_loader, max_train_batches),
        ("val", val_loader, max_val_batches),
    ]
    if test_loader is not None:
        split_specs.append(("test", test_loader, max_test_batches))
    for split, loader, max_batches in split_specs:
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shards = []
        sequences = 0
        for shard_idx, (x, _y) in enumerate(loader):
            if max_batches is not None and shard_idx >= max_batches:
                break
            x = x.to(device)
            activations = collect_teacher_layer_activations(
                teacher,
                x,
                layer_indices,
                mlp_input_mode=mlp_input_mode,
            )
            layer_payload = {}
            for layer_idx, layer_acts in activations.items():
                stats[layer_idx]["z"].update(layer_acts.z)
                stats[layer_idx]["r"].update(layer_acts.r)
                layer_payload[str(layer_idx)] = {
                    key: getattr(layer_acts, key).detach().to("cpu", dtype=tensor_dtype).contiguous()
                    for key in TRAINING_KEYS
                }
                if layer_acts.next_ln is not None:
                    layer_payload[str(layer_idx)]["next_ln"] = (
                        layer_acts.next_ln.detach().to("cpu", dtype=tensor_dtype).contiguous()
                    )
            rel_path = Path(split) / f"shard_{shard_idx:06d}.pt"
            torch.save(
                {
                    "input_ids": x.detach().cpu().long(),
                    "layers": layer_payload,
                },
                output / rel_path,
            )
            num_sequences = int(x.size(0))
            sequences += num_sequences
            shards.append(
                {
                    "path": str(rel_path),
                    "num_sequences": num_sequences,
                    "sequence_length": int(x.size(1)),
                }
            )
        splits[split] = {"num_sequences": sequences, "num_shards": len(shards), "shards": shards}

    calibration = {
        "layers": {
            str(layer_idx): {
                "z": layer_stats["z"].as_dict(),
                "r": layer_stats["r"].as_dict(),
            }
            for layer_idx, layer_stats in stats.items()
        },
        "max_quantile_samples": int(max_quantile_samples),
        "mlp_input_mode": mlp_input_mode,
        "scale_source": "q0_999_abs",
    }
    metadata = {
        "cache_version": CACHE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_name": model_name,
        "source": source,
        "block_size": int(block_size),
        "layers": [str(idx) for idx in layer_indices],
        "activation_keys": list(ACTIVATION_KEYS),
        "dtype": dtype,
        "mlp_input_mode": mlp_input_mode,
        "sequence_stride": int(sequence_stride),
        "splits": splits,
        "calibration": calibration,
    }
    save_cache_metadata(output, metadata)
    return metadata


def load_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(cache_dir) / "metadata.json").read_text(encoding="utf-8"))


def save_cache_metadata(cache_dir: str | Path, metadata: dict[str, Any]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibration_from_cache(cache_dir: str | Path) -> dict[str, Any]:
    metadata = load_cache_metadata(cache_dir)
    _validate_cache_version(metadata)
    return metadata["calibration"]


def cached_scales(cache_dir: str | Path, layer_index: int, *, source: str = "q0_999_abs") -> tuple[float, float]:
    return scales_from_calibration(calibration_from_cache(cache_dir), layer_index, source=source, default=1.0)


def _validate_cache_version(metadata: dict[str, Any]) -> None:
    version = int(metadata.get("cache_version", -1))
    if version != CACHE_VERSION:
        raise ValueError(f"Unsupported activation cache version {version}; expected {CACHE_VERSION}.")


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _get_device(args.device)
    tokenizer = "char" if args.debug and args.tokenizer == "auto" else args.tokenizer
    explicit_splits = args.train_data is not None or args.val_data is not None or args.test_data is not None
    if explicit_splits:
        if args.train_data is None or args.val_data is None:
            raise ValueError("--train_data and --val_data must be provided together.")
        train_dataset, val_dataset, test_dataset, _encode, _decode = load_explicit_text_datasets(
            args.train_data,
            args.val_data,
            args.test_data,
            model_name=args.model_name,
            tokenizer=tokenizer,
            block_size=args.block_size,
            vocab_size=128 if args.debug else None,
        )
        source = json.dumps(
            {
                "train": str(args.train_data),
                "val": str(args.val_data),
                "test": str(args.test_data) if args.test_data is not None else None,
            },
            sort_keys=True,
        )
    else:
        train_dataset, val_dataset, _encode, _decode = load_text_datasets(
            args.data,
            model_name=args.model_name,
            tokenizer=tokenizer,
            train_frac=args.train_frac,
            block_size=args.block_size,
            vocab_size=128 if args.debug else None,
        )
        test_dataset = None
        source = str(args.data) if args.data is not None else "default"
    train_dataset = _stride_dataset(train_dataset, args.sequence_stride)
    val_dataset = _stride_dataset(val_dataset, args.sequence_stride)
    if test_dataset is not None:
        test_dataset = _stride_dataset(test_dataset, args.sequence_stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if test_dataset is not None
        else None
    )
    teacher = _build_teacher(args).to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    layer_indices = _parse_layers(args.layers, opt_num_layers(teacher))
    metadata = write_activation_cache(
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        layer_indices=layer_indices,
        output_dir=args.output_dir,
        device=device,
        model_name=args.model_name,
        block_size=args.block_size,
        source=source,
        sequence_stride=args.sequence_stride,
        mlp_input_mode=args.mlp_input_mode,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
        dtype=args.dtype,
        max_quantile_samples=args.max_quantile_samples,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _stride_dataset(dataset: Dataset, stride: int) -> Dataset:
    stride = int(stride)
    if stride <= 1:
        return dataset
    return Subset(dataset, range(0, len(dataset), stride))


def _build_teacher(args: argparse.Namespace) -> torch.nn.Module:
    if not args.debug:
        return load_opt_causal_lm(args.model_name)
    try:
        from transformers import OPTConfig, OPTForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to construct debug OPT teachers.") from exc

    cfg = OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=args.block_size,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="relu",
        do_layer_norm_before=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        word_embed_proj_dim=32,
    )
    return OPTForCausalLM(cfg)


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "default":
        return default_probe_layers(num_layers)
    if raw == "all":
        return list(range(num_layers))
    values: list[int] = []
    for part in (item.strip() for item in raw.split(",") if item.strip()):
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending layer range '{part}'.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    bad = [idx for idx in values if idx < 0 or idx >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}.")
    return sorted(set(values))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--train_data", type=Path, default=None)
    parser.add_argument("--val_data", type=Path, default=None)
    parser.add_argument("--test_data", type=Path, default=None)
    parser.add_argument("--tokenizer", choices=["auto", "char"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--mlp_input_mode", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--sequence_stride", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--max_quantile_samples", type=int, default=250_000)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.sequence_stride <= 0:
        raise ValueError("--sequence_stride must be positive.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max_train_batches must be positive when provided.")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max_val_batches must be positive when provided.")
    if args.max_test_batches is not None and args.max_test_batches <= 0:
        raise ValueError("--max_test_batches must be positive when provided.")
    explicit_splits = args.train_data is not None or args.val_data is not None or args.test_data is not None
    if explicit_splits and (args.train_data is None or args.val_data is None):
        raise ValueError("--train_data and --val_data must be provided together.")
    if explicit_splits and args.data is not None:
        raise ValueError("Use either --data or explicit --train_data/--val_data/--test_data splits, not both.")
    return args


def _get_device(raw: str | None) -> torch.device:
    if raw is not None:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


if __name__ == "__main__":
    main()
