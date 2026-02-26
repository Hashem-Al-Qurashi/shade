"""Tests for experiment config loading."""

from __future__ import annotations

import pytest


class TestExperimentConfig:
    def test_load_gpt2_config(self, tmp_path):
        """Load a YAML config and verify all fields are parsed."""
        from experiments.config import ExperimentConfig, load_config

        config_text = """
model: gpt2
quantization: null
seed: 42
abliteration:
  n_trials: 5
  n_startup_trials: 3
  harmful_prompts: 20
steering:
  layers: [5, 6, 7]
  multipliers: [0.5, 1.0, 1.5, 2.0, 3.0]
  batch_size: 4
benchmark:
  gsm8k: false
  gsm8k_limit: 20
  perplexity: true
  perplexity_limit: 10
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_text)

        config = load_config(str(config_file))
        assert isinstance(config, ExperimentConfig)
        assert config.model == "gpt2"
        assert config.seed == 42
        assert config.abliteration.n_trials == 5
        assert config.steering.layers == [5, 6, 7]
        assert config.steering.multipliers == [0.5, 1.0, 1.5, 2.0, 3.0]
        assert config.benchmark.perplexity is True
        assert config.benchmark.gsm8k is False

    def test_defaults_applied(self, tmp_path):
        """Minimal config should fill defaults."""
        from experiments.config import load_config

        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("model: gpt2\n")

        config = load_config(str(config_file))
        assert config.model == "gpt2"
        assert config.seed == 42  # default
        assert config.abliteration.n_trials == 5  # default
        assert config.steering.multipliers == [0.5, 1.0, 1.5, 2.0, 3.0]  # default

    def test_missing_model_raises(self, tmp_path):
        """Config without model field should raise."""
        from experiments.config import load_config

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("seed: 42\n")

        with pytest.raises(ValueError, match="model"):
            load_config(str(config_file))
