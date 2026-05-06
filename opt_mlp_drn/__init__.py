"""OPT MLP-to-DRN replacement and distillation experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _prefer_sibling_digital_drn_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    init_path = repo_root / "__init__.py"
    if not init_path.exists():
        return

    existing = sys.modules.get("digital_drn")
    existing_file = Path(getattr(existing, "__file__", "")).resolve() if existing is not None else None
    if existing_file == init_path.resolve():
        return

    for name in list(sys.modules):
        if name == "digital_drn" or name.startswith("digital_drn."):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        "digital_drn",
        init_path,
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["digital_drn"] = module
    spec.loader.exec_module(module)


_prefer_sibling_digital_drn_source()

from .model import OPTDRNDecoderLayer, OPTMLPDRNForCausalLM, load_opt_causal_lm, parse_layer_indices

__all__ = [
    "OPTDRNDecoderLayer",
    "OPTMLPDRNForCausalLM",
    "load_opt_causal_lm",
    "parse_layer_indices",
]
