"""Result persistence and regression detection for benchmark data."""

from __future__ import annotations

import json as _json
from dataclasses import asdict
from pathlib import Path

from .benchmark import BenchmarkResult


def save_result(result: BenchmarkResult, path: str | Path) -> None:
    """Save a BenchmarkResult to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(asdict(result), f, indent=2)


def load_result(path: str | Path) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file."""
    with open(path) as f:
        data = _json.load(f)
    return BenchmarkResult(**data)


def compare_baseline(
    current: BenchmarkResult,
    baseline: BenchmarkResult,
    tolerance: float = 0.1,
) -> list[str]:
    """Compare current results against a baseline, returning a list of issues.

    Checks numeric fields. If a field is None in current, it's skipped
    (means that metric wasn't run). Tolerance is relative (10% = 0.1)
    for non-zero baselines, absolute for zero baselines.

    Returns:
        List of human-readable issue strings. Empty = no regressions.
    """
    issues: list[str] = []
    fields = [
        ("refusal_rate", "Refusal rate"),
        ("kl_divergence", "KL divergence"),
        ("perplexity", "Perplexity"),
        ("gsm8k_accuracy", "GSM8K accuracy"),
    ]

    for attr, label in fields:
        current_val = getattr(current, attr)
        baseline_val = getattr(baseline, attr)

        if current_val is None:
            continue

        if baseline_val is None:
            continue

        if baseline_val == 0:
            # Absolute check: flag if current is > tolerance
            if abs(current_val) > tolerance:
                issues.append(
                    f"{label}: was {baseline_val}, now {current_val} "
                    f"(exceeds absolute tolerance {tolerance})"
                )
        else:
            # Relative check
            change = abs(current_val - baseline_val) / abs(baseline_val)
            if change > tolerance:
                issues.append(
                    f"{label}: was {baseline_val}, now {current_val} "
                    f"(changed {change:.1%}, tolerance {tolerance:.0%})"
                )

    return issues
