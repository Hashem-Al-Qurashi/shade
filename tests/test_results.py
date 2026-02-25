"""Tests for shade.results module — result persistence and regression detection."""

from __future__ import annotations

import json

import pytest

from shade.benchmark import BenchmarkResult
from shade.results import compare_baseline, load_result, save_result


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        result = BenchmarkResult(
            model_name="gpt2", refusals=5, total_prompts=100,
            refusal_rate=0.05, kl_divergence=0.02,
            gsm8k_accuracy=0.45, gsm8k_total=50, perplexity=89.0,
        )
        path = tmp_path / "result.json"
        save_result(result, path)
        loaded = load_result(path)

        assert loaded.model_name == "gpt2"
        assert loaded.refusals == 5
        assert loaded.perplexity == pytest.approx(89.0)
        assert loaded.gsm8k_accuracy == pytest.approx(0.45)

    def test_roundtrip_with_none_fields(self, tmp_path):
        result = BenchmarkResult(
            model_name="test", refusals=10, total_prompts=50,
            refusal_rate=0.2, kl_divergence=0.1,
            gsm8k_accuracy=None, gsm8k_total=None, perplexity=None,
        )
        path = tmp_path / "result.json"
        save_result(result, path)
        loaded = load_result(path)

        assert loaded.gsm8k_accuracy is None
        assert loaded.perplexity is None

    def test_saved_file_is_valid_json(self, tmp_path):
        result = BenchmarkResult("m", 1, 10, 0.1, 0.0, None, None, None)
        path = tmp_path / "result.json"
        save_result(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["model_name"] == "m"

    def test_creates_parent_directories(self, tmp_path):
        result = BenchmarkResult("m", 1, 10, 0.1, 0.0, None, None, None)
        path = tmp_path / "deep" / "nested" / "result.json"
        save_result(result, path)
        assert path.exists()


class TestCompareBaseline:
    def test_within_tolerance_passes(self):
        current = BenchmarkResult("m", 5, 100, 0.05, 0.02, None, None, 89.0)
        baseline = BenchmarkResult("m", 5, 100, 0.05, 0.02, None, None, 90.0)
        issues = compare_baseline(current, baseline, tolerance=0.1)
        assert len(issues) == 0

    def test_outside_tolerance_reports(self):
        current = BenchmarkResult("m", 5, 100, 0.05, 0.02, None, None, 200.0)
        baseline = BenchmarkResult("m", 5, 100, 0.05, 0.02, None, None, 89.0)
        issues = compare_baseline(current, baseline, tolerance=0.1)
        assert len(issues) > 0
        assert any("perplexity" in issue.lower() for issue in issues)

    def test_skips_none_fields(self):
        current = BenchmarkResult("m", 5, 100, 0.05, 0.02, None, None, None)
        baseline = BenchmarkResult("m", 5, 100, 0.05, 0.02, 0.8, 50, 89.0)
        issues = compare_baseline(current, baseline, tolerance=0.1)
        # None in current means we didn't run that metric — skip, don't flag
        assert len(issues) == 0

    def test_zero_baseline_uses_absolute_check(self):
        current = BenchmarkResult("m", 5, 100, 0.05, 0.5, None, None, None)
        baseline = BenchmarkResult("m", 5, 100, 0.05, 0.0, None, None, None)
        issues = compare_baseline(current, baseline, tolerance=0.1)
        # KL divergence went from 0 to 0.5 — should flag
        assert len(issues) > 0
