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


def compute_gsm8k_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy by comparing extracted numeric answers."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if (pa := extract_gsm8k_answer(p)) is not None
        and pa == extract_gsm8k_answer(r)
    )
    return correct / len(predictions)


def load_gsm8k_problems(limit: int = 50) -> tuple[list[str], list[str]]:
    """Load GSM8K math problems from HuggingFace."""
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]")
    return [row["question"] for row in dataset], [row["answer"] for row in dataset]
