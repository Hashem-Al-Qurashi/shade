"""Pure logic tests — real tensor math, no models, no mocks.

Proves the mathematical correctness of benchmark, Pareto, comparison,
and SAE decomposition functions using deterministic inputs with known
expected outputs.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from shade.benchmark import (
    BenchmarkResult,
    compute_perplexity,
    extract_gsm8k_answer,
    format_benchmark_json,
    format_benchmark_table,
)
from shade.compare import format_comparison_json, format_comparison_table
from shade.pareto import compute_pareto_frontier_2d
from shade.sae_analysis import decompose_direction_into_features


# ---------------------------------------------------------------------------
# Perplexity: mathematical correctness
# ---------------------------------------------------------------------------


class TestPerplexityMath:
    """Verify perplexity computation using tensors with known expected values."""

    def test_uniform_distribution_gives_vocab_size(self):
        """If model predicts uniform over V tokens, perplexity should be V."""
        V = 10
        seq_len = 20

        # A "model" that returns uniform logits (all zeros → softmax = 1/V)
        class UniformModel:
            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                logits = torch.zeros(batch, seq, V)
                return type("Out", (), {"logits": logits})()

            def parameters(self):
                return iter([torch.nn.Parameter(torch.empty(0))])

        # A tokenizer that returns seq_len tokens
        class FakeTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                return {"input_ids": torch.randint(0, V, (1, seq_len))}

        ppl = compute_perplexity(UniformModel(), FakeTokenizer(), ["anything"])
        # Perplexity of uniform over 10 tokens = 10.0
        assert ppl == pytest.approx(V, rel=0.01), f"Expected ~{V}, got {ppl}"

    def test_perfect_prediction_gives_one(self):
        """If model always predicts the correct next token, perplexity ~ 1.0."""
        V = 50
        seq_len = 10
        # Fixed sequence
        token_ids = torch.tensor([[3, 7, 1, 4, 9, 2, 8, 0, 5, 6]])

        class PerfectModel:
            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                # Put very high logit on the correct next token
                logits = torch.full((batch, seq, V), -100.0)
                for t in range(seq - 1):
                    logits[0, t, input_ids[0, t + 1]] = 100.0
                # Last position doesn't matter (shifted out)
                logits[0, -1, 0] = 100.0
                return type("Out", (), {"logits": logits})()

            def parameters(self):
                return iter([torch.nn.Parameter(torch.empty(0))])

        class FakeTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                return {"input_ids": token_ids}

        ppl = compute_perplexity(PerfectModel(), FakeTokenizer(), ["anything"])
        assert ppl == pytest.approx(1.0, abs=0.01), f"Expected ~1.0, got {ppl}"

    def test_high_entropy_gives_high_perplexity(self):
        """Random logits should give perplexity >> 1."""
        V = 1000
        seq_len = 30
        torch.manual_seed(42)

        class RandomModel:
            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                logits = torch.randn(batch, seq, V)
                return type("Out", (), {"logits": logits})()

            def parameters(self):
                return iter([torch.nn.Parameter(torch.empty(0))])

        class FakeTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                return {"input_ids": torch.randint(0, V, (1, seq_len))}

        ppl = compute_perplexity(RandomModel(), FakeTokenizer(), ["anything"])
        # With V=1000, random logits → perplexity should be near V
        assert ppl > 100, f"Expected high perplexity, got {ppl}"
        assert math.isfinite(ppl), f"Perplexity should be finite, got {ppl}"

    def test_multiple_texts_averaged(self):
        """Perplexity over multiple texts averages the loss correctly."""
        V = 10
        token_ids_1 = torch.tensor([[1, 2, 3, 4, 5]])
        token_ids_2 = torch.tensor([[6, 7, 8, 9, 0]])

        class UniformModel:
            def __call__(self, input_ids):
                batch, seq = input_ids.shape
                logits = torch.zeros(batch, seq, V)
                return type("Out", (), {"logits": logits})()

            def parameters(self):
                return iter([torch.nn.Parameter(torch.empty(0))])

        call_count = 0

        class CountingTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                nonlocal call_count
                call_count += 1
                return {"input_ids": token_ids_1 if call_count == 1 else token_ids_2}

        ppl = compute_perplexity(UniformModel(), CountingTokenizer(), ["text1", "text2"])
        assert ppl == pytest.approx(V, rel=0.01)
        assert call_count == 2


# ---------------------------------------------------------------------------
# Pareto frontier: edge cases
# ---------------------------------------------------------------------------


class TestParetoEdgeCases:
    """Comprehensive edge cases for Pareto frontier computation."""

    def test_single_point_is_always_pareto(self):
        points = np.array([[5.0, 3.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask[0]

    def test_two_points_neither_dominates(self):
        """(1, 10) and (10, 1) — neither dominates, both Pareto."""
        points = np.array([[1.0, 10.0], [10.0, 1.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask.all()

    def test_two_points_one_dominates(self):
        """(1, 1) dominates (5, 5)."""
        points = np.array([[1.0, 1.0], [5.0, 5.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask[0]
        assert not mask[1]

    def test_all_identical_points(self):
        """Identical points — at least one should be Pareto."""
        points = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
        mask = compute_pareto_frontier_2d(points)
        # With identical second values, the sweep marks the first encountered
        assert mask.any()

    def test_diagonal_tradeoff_all_pareto(self):
        """Points along y = -x + c: perfect tradeoff, all Pareto."""
        points = np.array([[1.0, 9.0], [3.0, 7.0], [5.0, 5.0], [7.0, 3.0], [9.0, 1.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask.all()

    def test_one_dominates_all(self):
        """One point at (0, 0) dominates everything."""
        points = np.array([[5.0, 5.0], [3.0, 8.0], [0.0, 0.0], [10.0, 2.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask[2]  # (0,0) is Pareto
        # All others dominated
        assert not mask[0]
        assert not mask[1]
        assert not mask[3]

    def test_large_input_performance(self):
        """1000 random points — should complete quickly and return valid mask."""
        rng = np.random.default_rng(42)
        points = rng.random((1000, 2))
        mask = compute_pareto_frontier_2d(points)
        assert mask.shape == (1000,)
        assert mask.dtype == bool
        assert mask.any()
        assert not mask.all()  # unlikely all 1000 are Pareto


# ---------------------------------------------------------------------------
# Comparison formatting: robustness
# ---------------------------------------------------------------------------


class TestComparisonFormattingRobust:
    """Edge cases for comparison table and JSON formatting."""

    def test_single_result(self):
        results = [BenchmarkResult("Only", 10, 100, 0.1, 0.05, None, None, None)]
        table = format_comparison_table(results)
        assert "Only" in table

    def test_mixed_none_fields(self):
        """Some results have GSM8K, others don't."""
        results = [
            BenchmarkResult("A", 10, 100, 0.1, 0.05, 0.8, 50, None),
            BenchmarkResult("B", 20, 100, 0.2, 0.1, None, None, 15.0),
        ]
        table = format_comparison_table(results)
        assert "A" in table
        assert "B" in table
        # GSM8K column should appear (A has it)
        assert "80.0%" in table or "0.8" in table

    def test_all_none_optional_fields(self):
        """No GSM8K, no perplexity — just refusals and KL."""
        results = [
            BenchmarkResult("X", 50, 100, 0.5, 0.2, None, None, None),
            BenchmarkResult("Y", 30, 100, 0.3, 0.1, None, None, None),
        ]
        table = format_comparison_table(results)
        assert "X" in table
        assert "Y" in table

    def test_json_roundtrip(self):
        """JSON output can be parsed back and contains all data."""
        results = [
            BenchmarkResult("M1", 10, 100, 0.1, 0.05, 0.8, 50, 12.3),
            BenchmarkResult("M2", 20, 100, 0.2, 0.1, 0.7, 50, 14.5),
        ]
        parsed = json.loads(format_comparison_json(results))
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["model_name"] == "M1"
        assert parsed["results"][1]["perplexity"] == pytest.approx(14.5)


# ---------------------------------------------------------------------------
# SAE decomposition: deterministic tensor math
# ---------------------------------------------------------------------------


class TestSAEDecompositionMath:
    """Test SAE decomposition with deterministic tensors (no mocks)."""

    def test_top_k_sorted_by_absolute_activation(self):
        """Verify returned features are the ones with highest absolute activation."""
        torch.manual_seed(42)
        direction = torch.randn(256)

        # Build a real (simple) SAE-like encoder: just a linear projection
        class SimpleSAE:
            def __init__(self):
                torch.manual_seed(123)
                self.W = torch.randn(256, 512)  # d_model=256 → 512 features

            def encode(self, x):
                # x shape: (1, 1, 256) → output: (1, 1, 512)
                return (x @ self.W).relu()

        sae = SimpleSAE()
        features = decompose_direction_into_features(direction, sae, top_k=5)

        assert len(features) == 5
        # Verify all features have required keys
        assert all("index" in f and "activation" in f for f in features)
        # Verify indices are valid
        assert all(0 <= f["index"] < 512 for f in features)

        # Verify these are actually the top-5 by recomputing
        acts = sae.encode(direction.unsqueeze(0).unsqueeze(0)).squeeze()
        expected_top5 = acts.abs().topk(5).indices.tolist()
        actual_top5 = [f["index"] for f in features]
        assert set(actual_top5) == set(expected_top5)

    def test_zero_direction_gives_zero_activations(self):
        """Zero input direction should give zero (or near-zero) activations."""
        direction = torch.zeros(128)

        class ZeroSAE:
            def encode(self, x):
                return x @ torch.randn(128, 64)

        features = decompose_direction_into_features(direction, ZeroSAE(), top_k=3)
        assert len(features) == 3
        for f in features:
            assert abs(f["activation"]) < 1e-6
