"""OPT MLP-to-DRN replacement and distillation experiments."""

from .model import OPTDRNDecoderLayer, OPTMLPDRNForCausalLM, load_opt_causal_lm, parse_layer_indices

__all__ = [
    "OPTDRNDecoderLayer",
    "OPTMLPDRNForCausalLM",
    "load_opt_causal_lm",
    "parse_layer_indices",
]
