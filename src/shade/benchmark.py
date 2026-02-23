"""Benchmark utilities for capability-aware model evaluation."""

from __future__ import annotations

import re


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response."""
    if not text or not text.strip():
        return None

    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    match = re.search(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()

    match = re.search(r"the answer is\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None
