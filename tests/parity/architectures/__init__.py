"""Registry of the architectures covered by the parity harness.

Adding an architecture means adding a module here that exports ``CASES``.
"""
from __future__ import annotations

from . import gemma2, gpt2, llama, olmo, phi3, qwen3_moe, qwen_mistral

ALL_CASES = [
    *gpt2.CASES,
    *llama.CASES,
    *qwen_mistral.CASES,
    *phi3.CASES,
    *gemma2.CASES,
    *olmo.CASES,
    *qwen3_moe.CASES,
]
CASES_BY_NAME = {c.name: c for c in ALL_CASES}
