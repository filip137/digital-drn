"""OPT MLP-to-DRN replacement and distillation experiments."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def _ensure_local_digital_drn_package() -> None:
    """Use the sibling digital_drn source tree when running from this worktree.

    The repo maps the package name ``digital_drn`` to the repository root in
    ``pyproject.toml``. When the worktree is not installed, a direct
    ``python -m opt_mlp_drn...`` can otherwise resolve ``digital_drn`` from an
    older editable checkout elsewhere on the machine.
    """

    repo_root = Path(__file__).resolve().parents[1]
    existing = sys.modules.get("digital_drn")
    existing_paths = [Path(path).resolve() for path in getattr(existing, "__path__", [])] if existing else []
    if repo_root in existing_paths:
        return

    for module_name in list(sys.modules):
        if module_name.startswith("digital_drn."):
            del sys.modules[module_name]

    package = types.ModuleType("digital_drn")
    package.__file__ = str(repo_root / "__init__.py")
    package.__path__ = [str(repo_root)]
    spec = importlib.machinery.ModuleSpec("digital_drn", loader=None, is_package=True)
    spec.submodule_search_locations = [str(repo_root)]
    package.__spec__ = spec
    package.__package__ = "digital_drn"
    sys.modules["digital_drn"] = package


_ensure_local_digital_drn_package()

from .model import OPTDRNDecoderLayer, OPTMLPDRNForCausalLM, load_opt_causal_lm, parse_layer_indices

__all__ = [
    "OPTDRNDecoderLayer",
    "OPTMLPDRNForCausalLM",
    "load_opt_causal_lm",
    "parse_layer_indices",
]
