from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MQARConfig:
    """Configuration for synthetic multi-query associative recall data."""

    num_pairs: int = 4
    num_queries: int = 4
    num_keys: int = 32
    num_values: int = 32
    vocab_size: int | None = None
    seq_len: int | None = None
    train_samples: int = 2048
    eval_samples: int = 512
    batch_size: int = 32
    num_workers: int = 0
    seed: int | None = None
    ignore_index: int = -100
    shuffle_train: bool = True

    @property
    def required_seq_len(self) -> int:
        return 2 * int(self.num_pairs) + int(self.num_queries)

    @property
    def resolved_seq_len(self) -> int:
        return int(self.seq_len if self.seq_len is not None else self.required_seq_len)

    @property
    def resolved_vocab_size(self) -> int:
        minimum = int(self.num_keys) + int(self.num_values)
        return int(self.vocab_size if self.vocab_size is not None else minimum)

    @property
    def value_offset(self) -> int:
        return int(self.num_keys)

    def validate(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be strictly positive.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be strictly positive.")
        if self.num_keys < self.num_pairs:
            raise ValueError("num_keys must be at least num_pairs so each sample can use unique keys.")
        if self.num_values <= 0:
            raise ValueError("num_values must be strictly positive.")
        if self.resolved_vocab_size < self.num_keys + self.num_values:
            raise ValueError("vocab_size must be at least num_keys + num_values.")
        if self.resolved_seq_len < self.required_seq_len:
            raise ValueError("seq_len must be at least 2 * num_pairs + num_queries.")
        if self.train_samples <= 0:
            raise ValueError("train_samples must be strictly positive.")
        if self.eval_samples <= 0:
            raise ValueError("eval_samples must be strictly positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")


def mqar_config_from_mapping(config: Mapping[str, Any]) -> MQARConfig:
    field_names = {field.name for field in fields(MQARConfig)}
    values = {key: value for key, value in config.items() if key in field_names}

    aliases = {
        "num_train_samples": "train_samples",
        "num_eval_samples": "eval_samples",
        "test_samples": "eval_samples",
    }
    for source, target in aliases.items():
        if source in config and config[source] is not None:
            values[target] = config[source]

    if config.get("train_subset") is not None:
        values["train_samples"] = config["train_subset"]
    if config.get("test_subset") is not None:
        values["eval_samples"] = config["test_subset"]

    mqar_config = MQARConfig(**values)
    mqar_config.validate()
    return mqar_config


def generate_mqar_batch(
    config: MQARConfig,
    *,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    config.validate()
    if num_samples <= 0:
        raise ValueError("num_samples must be strictly positive.")

    seq_len = config.resolved_seq_len
    vocab_size = config.resolved_vocab_size
    pair_prefix_len = 2 * config.num_pairs
    query_start = pair_prefix_len
    query_stop = query_start + config.num_queries

    inputs = torch.randint(0, vocab_size, (num_samples, seq_len), generator=generator, dtype=torch.long)
    targets = torch.full((num_samples, seq_len), int(config.ignore_index), dtype=torch.long)

    for sample_idx in range(num_samples):
        pair_keys = torch.randperm(config.num_keys, generator=generator, dtype=torch.long)[: config.num_pairs]
        pair_values = (
            torch.randint(0, config.num_values, (config.num_pairs,), generator=generator, dtype=torch.long)
            + config.value_offset
        )
        query_pair_indices = torch.randint(
            0,
            config.num_pairs,
            (config.num_queries,),
            generator=generator,
            dtype=torch.long,
        )

        inputs[sample_idx, :pair_prefix_len:2] = pair_keys
        inputs[sample_idx, 1:pair_prefix_len:2] = pair_values
        inputs[sample_idx, query_start:query_stop] = pair_keys[query_pair_indices]
        targets[sample_idx, query_start:query_stop] = pair_values[query_pair_indices]

    return inputs, targets


class MQARDataset(Dataset):
    """Pre-generated synthetic MQAR samples for deterministic smoke runs."""

    def __init__(self, config: MQARConfig, *, split: str = "train") -> None:
        super().__init__()
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'.")
        config.validate()
        self.config = config
        self.split = split

        seed = 0 if config.seed is None else int(config.seed)
        if split == "eval":
            seed += 1
        generator = torch.Generator().manual_seed(seed)
        num_samples = int(config.train_samples if split == "train" else config.eval_samples)
        self.inputs, self.targets = generate_mqar_batch(
            config,
            num_samples=num_samples,
            generator=generator,
        )

    def __len__(self) -> int:
        return int(self.inputs.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        return self.inputs[idx], self.targets[idx]


def build_mqar_dataloaders(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    merged = dict(config)
    if merged.get("seed") is None and seed is not None:
        merged["seed"] = seed
    mqar_config = mqar_config_from_mapping(merged)

    train_dataset = MQARDataset(mqar_config, split="train")
    eval_dataset = MQARDataset(mqar_config, split="eval")
    generator = None
    if mqar_config.seed is not None:
        generator = torch.Generator().manual_seed(int(mqar_config.seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(mqar_config.batch_size),
        shuffle=bool(mqar_config.shuffle_train),
        num_workers=int(mqar_config.num_workers),
        drop_last=False,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(mqar_config.batch_size),
        shuffle=False,
        num_workers=int(mqar_config.num_workers),
        drop_last=False,
    )
    return train_loader, eval_loader
