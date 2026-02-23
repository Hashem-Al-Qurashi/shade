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


def compute_perplexity(
    model: object,
    tokenizer: object,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity on a list of texts."""
    import torch
    import torch.nn.functional as F

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return float(torch.exp(torch.tensor(avg_loss)).item())


def load_wikitext2_samples(limit: int = 50) -> list[str]:
    """Load Wikitext-2 test samples for perplexity evaluation."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{limit}]")
    return [row["text"] for row in dataset if row["text"].strip()]
