"""Compatibility exports for model loading and generation helpers."""

from ...utils.llm.model_utils import generate_completions, load_hf_lm_and_tokenizer

__all__ = ["generate_completions", "load_hf_lm_and_tokenizer"]
