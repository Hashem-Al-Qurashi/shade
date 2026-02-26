"""Experiment configuration loading from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class AbliterationConfig:
    n_trials: int = 5
    n_startup_trials: int = 3
    harmful_prompts: int = 20


@dataclass
class SteeringConfig:
    layers: list[int] = field(default_factory=lambda: [5, 6, 7])
    multipliers: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0])
    batch_size: int = 4


@dataclass
class BenchmarkConfig:
    gsm8k: bool = False
    gsm8k_limit: int = 20
    perplexity: bool = True
    perplexity_limit: int = 10


@dataclass
class ExperimentConfig:
    model: str
    quantization: str | None = None
    seed: int = 42
    abliteration: AbliterationConfig = field(default_factory=AbliterationConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError("Config must contain a 'model' field")

    abl_raw = raw.pop("abliteration", {}) or {}
    steer_raw = raw.pop("steering", {}) or {}
    bench_raw = raw.pop("benchmark", {}) or {}

    return ExperimentConfig(
        model=raw["model"],
        quantization=raw.get("quantization"),
        seed=raw.get("seed", 42),
        abliteration=AbliterationConfig(**abl_raw),
        steering=SteeringConfig(**steer_raw),
        benchmark=BenchmarkConfig(**bench_raw),
    )
