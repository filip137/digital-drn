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
    def __init__(self, text: str) -> None:
        chars = sorted(set(text) | {"~"})
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi["~"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(int(i), "~") for i in ids)


def load_tiny_shakespeare(
    path: str | Path | None,
    tokenizer: str = "gpt2",
    train_frac: float = 0.9,
    block_size: int = 256,
) -> tuple[TokenDataset, TokenDataset, Callable[[str], list[int]], Callable[[list[int] | torch.Tensor], str]]:
    text = Path(path).read_text(encoding="utf-8") if path is not None else DEFAULT_TINY_TEXT
    return build_datasets_from_text(text, tokenizer=tokenizer, train_frac=train_frac, block_size=block_size)


def build_datasets_from_text(
    text: str,
    tokenizer: str = "gpt2",
    train_frac: float = 0.9,
    block_size: int = 256,
) -> tuple[TokenDataset, TokenDataset, Callable[[str], list[int]], Callable[[list[int] | torch.Tensor], str]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must lie in (0, 1).")
    encode, decode = _build_tokenizer(text, tokenizer)
    token_ids = torch.tensor(encode(text), dtype=torch.long)
    if token_ids.numel() < 2 * (block_size + 1):
        repeats = (2 * (block_size + 1) // max(1, token_ids.numel())) + 1
        token_ids = token_ids.repeat(repeats)

    split = max(block_size + 1, int(train_frac * token_ids.numel()))
    split = min(split, token_ids.numel() - (block_size + 1))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    return TokenDataset(train_ids, block_size), TokenDataset(val_ids, block_size), encode, decode


def _build_tokenizer(
    text: str,
    tokenizer: str,
) -> tuple[Callable[[str], list[int]], Callable[[list[int] | torch.Tensor], str]]:
    if tokenizer == "gpt2":
        try:
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            return enc.encode, lambda ids: enc.decode(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        except ImportError:
            pass
    if tokenizer not in {"gpt2", "char"}:
        raise ValueError("tokenizer must be 'gpt2' or 'char'.")

    char_tokenizer = CharTokenizer(text)
    return char_tokenizer.encode, char_tokenizer.decode
