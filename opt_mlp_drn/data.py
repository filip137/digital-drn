from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_TINY_TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n"
)


class TokenDataset(Dataset):
    def __init__(self, token_ids: torch.LongTensor, block_size: int) -> None:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor.")
        if block_size <= 0:
            raise ValueError("block_size must be strictly positive.")
        if token_ids.numel() < block_size + 1:
            raise ValueError("Need at least block_size + 1 tokens.")

        self.token_ids = token_ids.long().contiguous()
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return self.token_ids.numel() - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y


class CharTokenizer:
    def __init__(self, text: str, vocab_size: int | None = None) -> None:
        chars = sorted(set(text) | {"~"})
        if vocab_size is not None and len(chars) > vocab_size:
            raise ValueError(f"Text has {len(chars)} characters, larger than vocab_size={vocab_size}.")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi["~"]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(int(i), "~") for i in ids)


def load_text_datasets(
    path: str | Path | None,
    *,
    model_name: str,
    tokenizer: str = "auto",
    train_frac: float = 0.9,
    block_size: int = 256,
    vocab_size: int | None = None,
) -> tuple[TokenDataset, TokenDataset, Callable[[str], list[int]], Callable[[list[int] | torch.Tensor], str]]:
    text = Path(path).read_text(encoding="utf-8") if path is not None else DEFAULT_TINY_TEXT
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie in (0, 1).")

    encode, decode = _build_tokenizer(text, model_name=model_name, tokenizer=tokenizer, vocab_size=vocab_size)
    token_ids = torch.tensor(encode(text), dtype=torch.long)
    if token_ids.numel() < 2 * (block_size + 1):
        repeats = (2 * (block_size + 1) // max(1, token_ids.numel())) + 1
        token_ids = token_ids.repeat(repeats)

    split = max(block_size + 1, int(train_frac * token_ids.numel()))
    split = min(split, token_ids.numel() - (block_size + 1))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    return TokenDataset(train_ids, block_size), TokenDataset(val_ids, block_size), encode, decode


def load_explicit_text_datasets(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path | None,
    *,
    model_name: str,
    tokenizer: str = "auto",
    block_size: int = 256,
    vocab_size: int | None = None,
) -> tuple[
    TokenDataset,
    TokenDataset,
    TokenDataset | None,
    Callable[[str], list[int]],
    Callable[[list[int] | torch.Tensor], str],
]:
    train_text = Path(train_path).read_text(encoding="utf-8")
    val_text = Path(val_path).read_text(encoding="utf-8")
    test_text = Path(test_path).read_text(encoding="utf-8") if test_path is not None else None
    tokenizer_text = train_text + val_text + (test_text or "")
    encode, decode = _build_tokenizer(
        tokenizer_text,
        model_name=model_name,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )
    train_dataset = TokenDataset(torch.tensor(encode(train_text), dtype=torch.long), block_size)
    val_dataset = TokenDataset(torch.tensor(encode(val_text), dtype=torch.long), block_size)
    test_dataset = (
        TokenDataset(torch.tensor(encode(test_text), dtype=torch.long), block_size)
        if test_text is not None
        else None
    )
    return train_dataset, val_dataset, test_dataset, encode, decode


def _build_tokenizer(
    text: str,
    *,
    model_name: str,
    tokenizer: str,
    vocab_size: int | None,
) -> tuple[Callable[[str], list[int]], Callable[[list[int] | torch.Tensor], str]]:
    if tokenizer == "auto":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use --tokenizer auto.") from exc

        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        def encode(text_value: str) -> list[int]:
            return hf_tokenizer.encode(text_value, add_special_tokens=False)

        def decode(ids: list[int] | torch.Tensor) -> str:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return hf_tokenizer.decode(ids)

        return encode, decode

    if tokenizer != "char":
        raise ValueError("tokenizer must be 'auto' or 'char'.")
    char_tokenizer = CharTokenizer(text, vocab_size=vocab_size)
    return char_tokenizer.encode, char_tokenizer.decode
