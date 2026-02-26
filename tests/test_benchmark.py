"""Tests for shade.benchmark module."""

from __future__ import annotations

import pytest

from shade.benchmark import extract_gsm8k_answer


class TestExtractGsm8kAnswer:
    def test_standard_format(self):
        assert extract_gsm8k_answer("Steps.\n#### 42") == "42"

    def test_negative_number(self):
        assert extract_gsm8k_answer("Negative.\n#### -7") == "-7"

    def test_decimal(self):
        assert extract_gsm8k_answer("Dividing.\n#### 3.5") == "3.5"

    def test_comma_separated(self):
        assert extract_gsm8k_answer("Cost.\n#### 1,200") == "1200"

    def test_model_the_answer_is(self):
        assert extract_gsm8k_answer("Work: 5*4=20. The answer is 20.") == "20"

    def test_model_equals(self):
        assert extract_gsm8k_answer("5 + 3 = 8") == "8"

    def test_boxed_latex(self):
        assert extract_gsm8k_answer("Therefore \\boxed{15}") == "15"

    def test_no_answer(self):
        assert extract_gsm8k_answer("I don't know.") is None

    def test_empty(self):
        assert extract_gsm8k_answer("") is None

    def test_whitespace(self):
        assert extract_gsm8k_answer("Steps.\n####   72  ") == "72"

    def test_multiple_numbers_takes_last(self):
        assert extract_gsm8k_answer("Got 10, then 20, answer is 30.") == "30"


class TestComputeGsm8kAccuracy:
    def test_all_correct(self):
        from shade.benchmark import compute_gsm8k_accuracy

        assert compute_gsm8k_accuracy(
            ["42", "The answer is 10", "#### 7"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(1.0)

    def test_all_wrong(self):
        from shade.benchmark import compute_gsm8k_accuracy

        assert compute_gsm8k_accuracy(
            ["99", "The answer is 0", "I don't know"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(0.0)

    def test_partial(self):
        from shade.benchmark import compute_gsm8k_accuracy

        assert compute_gsm8k_accuracy(
            ["42", "wrong", "#### 7"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(2 / 3)

    def test_empty(self):
        from shade.benchmark import compute_gsm8k_accuracy

        assert compute_gsm8k_accuracy([], []) == pytest.approx(0.0)


class TestComputePerplexity:
    def test_with_mock_model(self):
        from unittest.mock import MagicMock

        import torch

        from shade.benchmark import compute_perplexity

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Simulate tokenizer output
        mock_tokenizer.return_value = {"input_ids": torch.randint(0, 1000, (1, 50))}
        mock_tokenizer.model_max_length = 512

        # Simulate model output with logits
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 50, 1000)
        mock_model.return_value = mock_output

        # Mock model.parameters() to return something with a .device attribute
        mock_param = torch.nn.Parameter(torch.empty(0))
        mock_model.parameters.return_value = iter([mock_param])

        ppl = compute_perplexity(mock_model, mock_tokenizer, ["Hello world test."])
        assert isinstance(ppl, float)
        assert ppl > 0


from dataclasses import asdict  # noqa: E402


class TestBenchmarkResult:
    def test_fields(self):
        from shade.benchmark import BenchmarkResult

        r = BenchmarkResult(
            model_name="test",
            refusals=12,
            total_prompts=100,
            refusal_rate=0.12,
            kl_divergence=0.043,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=None,
        )
        assert r.refusal_rate == pytest.approx(0.12)
        assert r.perplexity is None

    def test_to_dict(self):
        from shade.benchmark import BenchmarkResult

        r = BenchmarkResult(
            model_name="m",
            refusals=5,
            total_prompts=50,
            refusal_rate=0.1,
            kl_divergence=0.02,
            gsm8k_accuracy=0.45,
            gsm8k_total=50,
            perplexity=12.3,
        )
        d = asdict(r)
        assert d["perplexity"] == pytest.approx(12.3)


class TestFormatBenchmarkTable:
    def test_without_gsm8k(self):
        from shade.benchmark import BenchmarkResult, format_benchmark_table

        result = BenchmarkResult("test/model", 12, 100, 0.12, 0.043, None, None, None)
        table_str = format_benchmark_table(result)
        assert "test/model" in table_str
        assert "12" in table_str
        assert "0.043" in table_str

    def test_with_all_metrics(self):
        from shade.benchmark import BenchmarkResult, format_benchmark_table

        result = BenchmarkResult("test/model", 12, 100, 0.12, 0.043, 0.452, 50, 15.7)
        table_str = format_benchmark_table(result)
        assert "GSM8K" in table_str or "45.2" in table_str
        assert "15.7" in table_str or "Perplexity" in table_str


class TestFormatBenchmarkJson:
    def test_valid_json(self):
        import json

        from shade.benchmark import BenchmarkResult, format_benchmark_json

        result = BenchmarkResult("test/model", 12, 100, 0.12, 0.043, 0.452, 50, 15.7)
        parsed = json.loads(format_benchmark_json(result))
        assert parsed["model_name"] == "test/model"
        assert parsed["kl_divergence"] == pytest.approx(0.043)


import json  # noqa: E402


class TestFormatComparisonTable:
    def test_formats_multiple_results(self):
        from shade.benchmark import BenchmarkResult
        from shade.compare import format_comparison_table

        results = [
            BenchmarkResult("Base", 87, 100, 0.87, 0.0, 0.452, 50, 12.3),
            BenchmarkResult("Abliterated", 12, 100, 0.12, 0.043, 0.438, 50, 13.1),
            BenchmarkResult("Steered", 23, 100, 0.23, 0.018, 0.449, 50, 12.5),
        ]
        table = format_comparison_table(results)
        assert "Base" in table
        assert "Abliterated" in table
        assert "Steered" in table


class TestFormatComparisonJson:
    def test_valid_json(self):
        from shade.benchmark import BenchmarkResult
        from shade.compare import format_comparison_json

        results = [
            BenchmarkResult("Base", 87, 100, 0.87, 0.0, 0.452, 50, 12.3),
        ]
        parsed = json.loads(format_comparison_json(results))
        assert len(parsed["results"]) == 1


import numpy as np  # noqa: E402


class TestComputeParetoFrontier:
    def test_simple_frontier(self):
        from shade.pareto import compute_pareto_frontier_2d

        # 4 points: (refusals, kl). Minimizing both.
        points = np.array(
            [
                [10, 0.5],  # Pareto (low refusals, high kl)
                [50, 0.1],  # Pareto (high refusals, low kl)
                [30, 0.3],  # Dominated by a mix
                [10, 0.1],  # Pareto (best on both!)
            ]
        )
        mask = compute_pareto_frontier_2d(points)
        assert mask[3] == True  # dominates everything  # noqa: E712
        # Point 2 (30, 0.3) is dominated by point 3 (10, 0.1)
        assert mask[2] == False  # noqa: E712

    def test_all_pareto(self):
        from shade.pareto import compute_pareto_frontier_2d

        points = np.array([[1, 10.0], [5, 5.0], [10, 1.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask.all()


class TestDecomposeDirection:
    def test_returns_top_features(self):
        from unittest.mock import MagicMock

        import torch

        from shade.sae_analysis import decompose_direction_into_features

        # Mock: refusal direction is a random vector
        direction = torch.randn(2304)  # Gemma 2B hidden dim
        # Mock SAE with encode method
        mock_sae = MagicMock()
        mock_sae.encode.return_value = torch.randn(1, 1, 16384)  # 16K features

        features = decompose_direction_into_features(direction, mock_sae, top_k=10)
        assert len(features) == 10
        assert all("index" in f and "activation" in f for f in features)
