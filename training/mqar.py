from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100


@dataclass(frozen=True)
class MQARConfig:
    """Configuration for synthetic multi-query associative recall data."""

    style: str = "compact"
    num_pairs: int = 4
    num_queries: int = 4
    num_kv_pairs: int | None = None
    num_keys: int = 32
    num_values: int = 32
    vocab_size: int | None = None
    seq_len: int | None = None
    train_samples: int = 2048
    eval_samples: int = 512
    train_examples: int | None = None
    val_examples: int | None = None
    batch_size: int = 32
    num_workers: int = 0
    seed: int | None = None
    train_seed: int | None = None
    val_seed: int | None = None
    power_a: float = 0.01
    random_non_queries: bool = True
    ignore_index: int = IGNORE_INDEX
    shuffle_train: bool = True

    @property
    def resolved_num_pairs(self) -> int:
        return int(self.num_kv_pairs if self.num_kv_pairs is not None else self.num_pairs)

    @property
    def resolved_train_samples(self) -> int:
        return int(self.train_examples if self.train_examples is not None else self.train_samples)

    @property
    def resolved_eval_samples(self) -> int:
        return int(self.val_examples if self.val_examples is not None else self.eval_samples)

    @property
    def resolved_train_seed(self) -> int:
        if self.train_seed is not None:
            return int(self.train_seed)
        return 0 if self.seed is None else int(self.seed)

    @property
    def resolved_eval_seed(self) -> int:
        if self.val_seed is not None:
            return int(self.val_seed)
        return self.resolved_train_seed + 1

    @property
    def required_seq_len(self) -> int:
        if self.style == "small_gpt":
            return 4 * self.resolved_num_pairs
        return 2 * int(self.num_pairs) + int(self.num_queries)

    @property
    def resolved_seq_len(self) -> int:
        return int(self.seq_len if self.seq_len is not None else self.required_seq_len)

    @property
    def resolved_vocab_size(self) -> int:
        if self.style == "small_gpt" and self.vocab_size is not None:
            return int(self.vocab_size)
        minimum = int(self.num_keys) + int(self.num_values)
        return int(self.vocab_size if self.vocab_size is not None else minimum)

    @property
    def value_offset(self) -> int:
        return int(self.num_keys)

    def validate(self) -> None:
        if self.style not in {"compact", "small_gpt"}:
            raise ValueError("style must be 'compact' or 'small_gpt'.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.style == "small_gpt":
            _validate_small_gpt_mqar_args(
                vocab_size=self.resolved_vocab_size,
                num_examples=max(self.resolved_train_samples, self.resolved_eval_samples),
                seq_len=self.resolved_seq_len,
                num_kv_pairs=self.resolved_num_pairs,
                power_a=float(self.power_a),
            )
            return

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
        if self.resolved_train_samples <= 0:
            raise ValueError("train_samples must be strictly positive.")
        if self.resolved_eval_samples <= 0:
            raise ValueError("eval_samples must be strictly positive.")


def mqar_config_from_mapping(config: Mapping[str, Any]) -> MQARConfig:
    field_names = {field.name for field in fields(MQARConfig)}
    values = {key: value for key, value in config.items() if key in field_names}

    aliases = {
        "num_train_samples": "train_samples",
        "num_eval_samples": "eval_samples",
        "test_samples": "eval_samples",
        "variant": "style",
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


def _validate_small_gpt_mqar_args(
    vocab_size: int,
    num_examples: int,
    seq_len: int,
    num_kv_pairs: int,
    power_a: float,
) -> None:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if seq_len % 2 != 0:
        raise ValueError("seq_len must be even.")
    if vocab_size <= seq_len:
        raise ValueError("vocab_size must be greater than seq_len.")
    if num_kv_pairs <= 0:
        raise ValueError("num_kv_pairs must be positive.")
    if num_kv_pairs * 4 > seq_len:
        raise ValueError("num_kv_pairs * 4 must be <= seq_len.")
    if power_a < 0:
        raise ValueError("power_a must be non-negative.")

    half_vocab = vocab_size // 2
    num_key_tokens = half_vocab - 1
    num_value_tokens = vocab_size - half_vocab
    if num_key_tokens < num_kv_pairs:
        raise ValueError("not enough key tokens for distinct keys.")
    if num_value_tokens < num_kv_pairs:
        raise ValueError("not enough value tokens for distinct values.")


def generate_small_gpt_mqar(
    *,
    vocab_size: int,
    num_examples: int,
    seq_len: int,
    num_kv_pairs: int,
    seed: int,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Generate the shifted MQAR task used by `/home/filip/small_gpt`.

    Labels are aligned with returned input positions. Query values are written
    into a one-token-ahead label buffer and then shifted back, matching the
    causal-LM contract while keeping the answer token out of the visible input.
    """
    _validate_small_gpt_mqar_args(vocab_size, num_examples, seq_len, num_kv_pairs, power_a)

    generator = torch.Generator().manual_seed(int(seed))

    half_vocab = vocab_size // 2
    key_vocab = torch.arange(1, half_vocab, dtype=torch.long)
    value_vocab = torch.arange(half_vocab, vocab_size, dtype=torch.long)

    internal_len = seq_len + 1
    prefix_len = 2 * num_kv_pairs
    examples = torch.zeros((num_examples, internal_len), dtype=torch.long)
    label_buffer = torch.full(
        (num_examples, internal_len),
        int(ignore_index),
        dtype=torch.long,
    )

    query_slot_count = (seq_len - prefix_len) // 2
    gap_ids = torch.arange(query_slot_count, dtype=torch.long)
    if power_a == 0:
        gap_weights = torch.ones(query_slot_count, dtype=torch.float)
    else:
        gap_distances = torch.arange(1, query_slot_count + 1, dtype=torch.float)
        gap_weights = float(power_a) * gap_distances.pow(float(power_a) - 1.0)

    for row in range(num_examples):
        key_perm = torch.randperm(key_vocab.numel(), generator=generator)[:num_kv_pairs]
        value_perm = torch.randperm(value_vocab.numel(), generator=generator)[:num_kv_pairs]
        keys = key_vocab[key_perm]
        values = value_vocab[value_perm]

        examples[row, :prefix_len:2] = keys
        examples[row, 1:prefix_len:2] = values

        sampled_gaps = gap_ids[
            torch.multinomial(gap_weights, num_kv_pairs, replacement=False, generator=generator)
        ]
        for pair_idx, (key, value) in enumerate(zip(keys, values)):
            query_pos = prefix_len + int(2 * sampled_gaps[pair_idx].item())
            examples[row, query_pos] = key
            label_buffer[row, query_pos + 1] = value

    inputs = examples[:, :-1].clone()
    labels = label_buffer[:, 1:].clone()

    if random_non_queries:
        zero_mask = inputs == 0
        num_zeros = int(zero_mask.sum().item())
        if num_zeros:
            inputs[zero_mask] = torch.randint(
                low=0,
                high=vocab_size,
                size=(num_zeros,),
                generator=generator,
                dtype=torch.long,
            )

    return inputs.long(), labels.long()


class MQARDataset(Dataset):
    """Pre-generated synthetic MQAR samples for deterministic smoke runs."""

    def __init__(self, config: MQARConfig, *, split: str = "train") -> None:
        super().__init__()
        if split not in {"train", "eval"}:
            raise ValueError("split must be 'train' or 'eval'.")
        config.validate()
        self.config = config
        self.split = split

        if config.style == "small_gpt":
            seed = config.resolved_train_seed if split == "train" else config.resolved_eval_seed
            num_samples = config.resolved_train_samples if split == "train" else config.resolved_eval_samples
            self.inputs, self.targets = generate_small_gpt_mqar(
                vocab_size=config.resolved_vocab_size,
                num_examples=num_samples,
                seq_len=config.resolved_seq_len,
                num_kv_pairs=config.resolved_num_pairs,
                seed=seed,
                power_a=float(config.power_a),
                random_non_queries=bool(config.random_non_queries),
                ignore_index=int(config.ignore_index),
            )
        else:
            seed = 0 if config.seed is None else int(config.seed)
            if split == "eval":
                seed += 1
            generator = torch.Generator().manual_seed(seed)
            num_samples = int(config.resolved_train_samples if split == "train" else config.resolved_eval_samples)
            self.inputs, self.targets = generate_mqar_batch(
                config,
                num_samples=num_samples,
                generator=generator,
            )

    def __len__(self) -> int:
        return int(self.inputs.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        return self.inputs[idx], self.targets[idx]

    @property
    def query_mask(self) -> torch.BoolTensor:
        return self.targets != int(self.config.ignore_index)


class SmallGPTMQARDataset(MQARDataset):
    """Dataset wrapper with the same constructor shape as `/home/filip/small_gpt`."""

    def __init__(
        self,
        *,
        vocab_size: int,
        num_examples: int,
        seq_len: int,
        num_kv_pairs: int,
        seed: int,
        power_a: float = 0.01,
        random_non_queries: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        config = MQARConfig(
            style="small_gpt",
            num_kv_pairs=int(num_kv_pairs),
            vocab_size=int(vocab_size),
            seq_len=int(seq_len),
            train_examples=int(num_examples),
            eval_samples=1,
            train_seed=int(seed),
            power_a=float(power_a),
            random_non_queries=bool(random_non_queries),
            ignore_index=int(ignore_index),
        )
        super().__init__(config, split="train")


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
