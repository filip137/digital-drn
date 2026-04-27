from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gpt2_ladder_drn.config import DebugGPT2Config, GPT2Config
from gpt2_ladder_drn.data import DEFAULT_TINY_TEXT, build_datasets_from_text
from gpt2_ladder_drn.model_gpt2 import GPT2LMHeadModel
from gpt2_ladder_drn.utils import get_device


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> torch.LongTensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -_context_size(model) :]
        logits = model(idx_cond)["logits"]
        logits = logits[:, -1, :] / max(temperature, 1.0e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)
    return input_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    config = DebugGPT2Config(dropout=0.0) if args.debug else GPT2Config(dropout=0.0)
    model = GPT2LMHeadModel(config).to(device)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model"])

    _, _, encode, decode = build_datasets_from_text(
        DEFAULT_TINY_TEXT + args.prompt,
        tokenizer="gpt2",
        train_frac=0.9,
        block_size=min(32, config.block_size),
    )
    input_ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(decode(output[0].cpu()))


def _context_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "base"):
        config = model.base.config
    return int(config.block_size)


if __name__ == "__main__":
    main()
