from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import load_opt_causal_lm


def main() -> None:
    args = _parse_args()
    if args.mode == "generate_synthetic":
        generate_synthetic_text(args)
    elif args.mode == "real_subset":
        write_token_subset(args.real_path, args.output, args.model_name, args.max_tokens)
    elif args.mode == "mix":
        mix_text(args)
    else:
        raise ValueError(f"Unsupported mode '{args.mode}'.")


@torch.no_grad()
def generate_synthetic_text(args: argparse.Namespace) -> None:
    tokenizer = _tokenizer(args.model_name)
    device = _get_device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = load_opt_causal_lm(args.model_name).to(device)
    model.eval()
    prompts = [prompt.strip() for prompt in args.prompt.split("|||") if prompt.strip()]
    if not prompts:
        prompts = ["The"]
    fallback_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    fallback_id = int(fallback_id if fallback_id is not None else 2)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_id = int(pad_id if pad_id is not None else fallback_id)

    token_ids: list[int] = []
    sequence_index = 0
    while len(token_ids) < args.max_tokens:
        remaining = args.max_tokens - len(token_ids)
        target_len = min(int(args.chunk_tokens), remaining)
        prompt = prompts[sequence_index % len(prompts)]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) or [fallback_id]
        if len(prompt_ids) >= target_len:
            token_ids.extend(prompt_ids[:target_len])
            sequence_index += 1
            continue
        batch_size = min(int(args.generation_batch_size), max(1, (remaining + target_len - 1) // target_len))
        input_ids = torch.tensor([prompt_ids] * batch_size, dtype=torch.long, device=device)
        max_new_tokens = target_len - len(prompt_ids)
        generated = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=max(float(args.temperature), 1.0e-6),
            top_k=max(0, int(args.top_k)),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=None,
        )
        for row in generated.detach().cpu().tolist():
            token_ids.extend(row[:target_len])
            sequence_index += 1
            if len(token_ids) >= args.max_tokens:
                break

    text = tokenizer.decode(token_ids[: args.max_tokens])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


def write_token_subset(input_path: Path, output_path: Path, model_name: str, max_tokens: int) -> None:
    text = input_path.read_text(encoding="utf-8")
    tokenizer = _tokenizer(model_name)
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tokenizer.decode(ids), encoding="utf-8")


def mix_text(args: argparse.Namespace) -> None:
    if args.real_path is None or args.synthetic_path is None:
        raise ValueError("--real_path and --synthetic_path are required for --mode mix.")
    tokenizer = _tokenizer(args.model_name)
    real_ids = tokenizer.encode(args.real_path.read_text(encoding="utf-8"), add_special_tokens=False)
    synthetic_ids = tokenizer.encode(args.synthetic_path.read_text(encoding="utf-8"), add_special_tokens=False)
    real_count = min(len(real_ids), int(round(args.max_tokens * args.real_fraction)))
    synthetic_count = max(0, args.max_tokens - real_count)
    ids = real_ids[:real_count] + synthetic_ids[:synthetic_count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tokenizer.decode(ids), encoding="utf-8")


def _tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to tokenize OPT corpora.") from exc
    return AutoTokenizer.from_pretrained(model_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate_synthetic", "real_subset", "mix"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--real_path", type=Path, default=None)
    parser.add_argument("--synthetic_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_tokens", type=int, default=1_000_000)
    parser.add_argument("--real_fraction", type=float, default=0.5)
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--chunk_tokens", type=int, default=512)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive.")
    if args.chunk_tokens <= 1:
        raise ValueError("--chunk_tokens must be greater than 1.")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation_batch_size must be positive.")
    if not 0.0 <= args.real_fraction <= 1.0:
        raise ValueError("--real_fraction must be in [0, 1].")
    if args.mode == "real_subset" and args.real_path is None:
        raise ValueError("--real_path is required for --mode real_subset.")
    return args


def _get_device(raw: str | None) -> torch.device:
    if raw is not None:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
