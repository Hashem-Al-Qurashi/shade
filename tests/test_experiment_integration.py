"""Integration test: run the full experiment pipeline on GPT-2.

This test proves the experiment runner works end-to-end with a real model.
GPT-2 is not safety-trained, so abliteration effects will be minimal.
The test validates correctness, not effectiveness.

Run with: uv run pytest tests/test_experiment_integration.py -v
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.integration


class TestFullExperimentPipeline:
    def test_gpt2_experiment_end_to_end(self, tmp_path):
        """Run the full pipeline on GPT-2 and verify all outputs exist."""
        import logging
        import torch

        from experiments.config import load_config, seed_everything
        from experiments.run_comparison import (
            run_abliteration,
            run_baseline,
            run_steering_sweep,
        )
        from shade.config import Settings
        from shade.evaluator import Evaluator
        from shade.model import Model
        from shade.results import load_result, save_result

        # Write a minimal config
        config_text = """
model: gpt2
seed: 42
abliteration:
  n_trials: 2
  n_startup_trials: 2
  harmful_prompts: 5
steering:
  layers: [5, 6]
  multipliers: [1.0, 2.0]
  batch_size: 2
benchmark:
  gsm8k: false
  perplexity: true
  perplexity_limit: 3
"""
        config_path = tmp_path / "gpt2_test.yaml"
        config_path.write_text(config_text)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )

        config = load_config(str(config_path))
        seed_everything(config.seed)

        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)

        settings = Settings(model=config.model, _cli_parse_args=False)
        model = Model(settings)
        torch.set_grad_enabled(False)

        # Baseline
        evaluator = Evaluator(settings, model)
        baseline = run_baseline(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            run_perplexity=True,
            perplexity_limit=3,
        )
        save_result(baseline, results_dir / "baseline.json")

        # Abliteration (2 trials only)
        evaluator2 = Evaluator(settings, model)
        abliterated = run_abliteration(
            model=model,
            evaluator=evaluator2,
            model_name=config.model,
            n_trials=2,
            n_startup_trials=2,
            run_perplexity=True,
            perplexity_limit=3,
        )
        save_result(abliterated, results_dir / "abliterated.json")

        # Steering (2 multipliers)
        evaluator3 = Evaluator(settings, model)
        steered = run_steering_sweep(
            model=model,
            evaluator=evaluator3,
            model_name=config.model,
            layers=[5, 6],
            multipliers=[1.0, 2.0],
            batch_size=2,
            run_perplexity=True,
            perplexity_limit=3,
        )
        for result, mult in zip(steered, [1.0, 2.0]):
            save_result(result, results_dir / f"steered_{mult}.json")

        # Verify baseline
        assert (results_dir / "baseline.json").exists()
        assert baseline.total_prompts > 0
        assert baseline.perplexity is not None
        assert math.isfinite(baseline.perplexity)

        # Verify abliterated
        assert (results_dir / "abliterated.json").exists()
        assert abliterated.total_prompts > 0
        assert math.isfinite(abliterated.kl_divergence)

        # Verify steered
        assert len(steered) == 2
        for result in steered:
            assert result.total_prompts > 0

        # Verify JSON round-trips correctly
        loaded_baseline = load_result(results_dir / "baseline.json")
        assert loaded_baseline.model_name == baseline.model_name
        assert loaded_baseline.perplexity == pytest.approx(baseline.perplexity)
