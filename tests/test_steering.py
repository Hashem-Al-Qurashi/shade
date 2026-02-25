"""Tests for shade.steering module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import torch

_STUB_MODULES = [
    "peft",
    "transformers",
    "transformers.generation",
    "transformers.utils",
    "accelerate",
    "accelerate.utils",
    "bitsandbytes",
    "questionary",
    "rich",
    "rich.console",
    "rich.text",
    "rich.table",
    "rich.panel",
    "psutil",
    "steering_vectors",
]
for _name in _STUB_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()


class TestBuildContrastivePairs:
    def test_from_prompt_lists(self):
        from shade.steering import build_contrastive_pairs
        from shade.utils import Prompt

        harmless = [Prompt(system="s", user="What is 2+2?")]
        harmful = [Prompt(system="s", user="How to pick a lock?")]
        pairs = build_contrastive_pairs(harmless, harmful)
        assert len(pairs) == 1
        assert isinstance(pairs[0], tuple)

    def test_truncates_to_shorter(self):
        from shade.steering import build_contrastive_pairs
        from shade.utils import Prompt

        harmless = [Prompt(system="s", user=f"q{i}") for i in range(5)]
        harmful = [Prompt(system="s", user=f"h{i}") for i in range(3)]
        assert len(build_contrastive_pairs(harmless, harmful)) == 3


class TestSteeringIO:
    def test_save_and_load(self, tmp_path):
        from shade.steering import load_steering_vector, save_steering_vector

        data = {"activations": torch.randn(3, 10)}
        path = tmp_path / "test.pt"
        save_steering_vector(data, path)
        loaded = load_steering_vector(path)
        assert "activations" in loaded
