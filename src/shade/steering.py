"""Activation steering for LLM behavior modification."""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import Prompt


def build_contrastive_pairs(
    harmless_prompts: list[Prompt],
    harmful_prompts: list[Prompt],
) -> list[tuple[str, str]]:
    """Build (positive, negative) pairs for steering training."""
    n = min(len(harmless_prompts), len(harmful_prompts))
    return [(harmless_prompts[i].user, harmful_prompts[i].user) for i in range(n)]


def save_steering_vector(vector: object, path: Path) -> None:
    """Save a steering vector to disk."""
    torch.save(vector, path)


def load_steering_vector(path: Path) -> object:
    """Load a steering vector from disk."""
    return torch.load(path, weights_only=False)


def train_steering_vector_from_prompts(
    model: object,
    tokenizer: object,
    harmless_prompts: list[Prompt],
    harmful_prompts: list[Prompt],
    layers: list[int] | None = None,
    batch_size: int = 4,
    show_progress: bool = True,
) -> object:
    """Train a steering vector from contrastive prompt pairs."""
    from steering_vectors import train_steering_vector

    pairs = build_contrastive_pairs(harmless_prompts, harmful_prompts)
    return train_steering_vector(
        model, tokenizer, pairs,
        layers=layers, batch_size=batch_size, show_progress=show_progress,
    )
