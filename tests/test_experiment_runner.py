"""Tests for the experiment runner pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRunBaseline:
    def test_run_baseline_returns_benchmark_result(self, tmp_path):
        """run_baseline should return a BenchmarkResult with correct model name."""
        from experiments.run_comparison import run_baseline
        from shade.benchmark import BenchmarkResult

        # Mock the heavy objects
        mock_model = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.bad_prompts = [MagicMock()] * 10
        mock_evaluator.good_prompts = [MagicMock()] * 10
        mock_evaluator.count_refusals.return_value = 8
        mock_evaluator.base_logprobs = MagicMock()

        # Mock run_benchmark to return a known result
        expected = BenchmarkResult(
            model_name="gpt2 (baseline)",
            refusals=8,
            total_prompts=10,
            refusal_rate=0.8,
            kl_divergence=0.0,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=67.0,
        )

        with patch("experiments.run_comparison.run_benchmark", return_value=expected):
            result = run_baseline(
                model=mock_model,
                evaluator=mock_evaluator,
                model_name="gpt2",
                run_gsm8k=False,
                gsm8k_limit=20,
                run_perplexity=True,
                perplexity_limit=10,
            )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "gpt2 (baseline)"
        assert result.perplexity == 67.0


class TestRunAbliteration:
    def test_returns_best_trial_result(self):
        """run_abliteration should return a BenchmarkResult for the best trial."""
        import torch

        from experiments.run_comparison import run_abliteration
        from shade.benchmark import BenchmarkResult

        mock_model = MagicMock()
        mock_model.get_layers.return_value = list(range(12))
        mock_model.get_abliterable_components.return_value = [
            "attn.o_proj",
            "mlp.down_proj",
        ]
        # Return real tensors so F.normalize works
        mock_model.get_residuals_batched.return_value = torch.randn(10, 12, 64)
        mock_evaluator = MagicMock()
        mock_evaluator.get_score.return_value = ((0.1, 0.2), 0.05, 3)
        mock_evaluator.bad_prompts = [MagicMock()] * 10

        expected = BenchmarkResult(
            model_name="gpt2 (abliterated)",
            refusals=3,
            total_prompts=10,
            refusal_rate=0.3,
            kl_divergence=0.05,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=70.0,
        )

        # Build a mock trial that returns valid suggestions
        def make_mock_trial():
            trial = MagicMock()
            trial.number = 0
            trial.suggest_categorical.return_value = "global"
            trial.suggest_float.return_value = 5.0
            return trial

        # Mock load_prompts (lazy import inside run_abliteration) and run_benchmark
        with (
            patch("experiments.run_comparison.run_benchmark", return_value=expected),
            patch("shade.utils.load_prompts", return_value=[MagicMock()] * 5),
            patch("experiments.run_comparison.optuna") as mock_optuna,
        ):
            # Make the mock study's optimize actually call the objective once
            mock_study = MagicMock()

            def fake_optimize(objective_fn, n_trials=1):
                objective_fn(make_mock_trial())

            mock_study.optimize.side_effect = fake_optimize
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.logging = MagicMock()
            mock_optuna.samplers.TPESampler.return_value = MagicMock()

            result = run_abliteration(
                model=mock_model,
                evaluator=mock_evaluator,
                model_name="gpt2",
                n_trials=2,
                n_startup_trials=1,
                run_gsm8k=False,
                gsm8k_limit=20,
                run_perplexity=True,
                perplexity_limit=10,
            )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "gpt2 (abliterated)"
        assert result.refusals == 3
