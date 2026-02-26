# Experiment Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `experiments/run_comparison.py` — a single script that runs baseline → abliterate → steer → benchmark → Pareto on any model, saving reproducible JSON results.

**Architecture:** The script loads a YAML config, instantiates `shade.model.Model` and `shade.evaluator.Evaluator`, then runs the 9-step pipeline from the design doc. Each step checks if results already exist (idempotent). All existing Shade modules are used as-is — no new ML code.

**Tech Stack:** Python 3.10+, PyYAML (via transformers dependency), all existing shade.* modules.

**Important context for the implementer:**

- `Settings(model="gpt2", _cli_parse_args=False)` — the `_cli_parse_args=False` is critical to avoid CLI argument parsing in scripts.
- `Evaluator.__init__` downloads HuggingFace datasets and computes base logprobs/refusals — it takes 10-30s.
- `run_benchmark` takes a `Model` wrapper (not raw model) and an `Evaluator`.
- The abliteration pipeline in `main.py` (lines 564-684) is the reference: load prompts → get residuals → compute refusal directions → Optuna optimize → abliterate.
- `Model.abliterate()` expects `AbliterationParameters` per component, and `direction_index` as a float (layer position) or None (per-layer).
- `shade compare` (cli.py:880-974) has a NOTE at line 921: "abliterate_model() does not exist as a standalone function yet" — abliteration is Optuna-driven. For the experiment runner, we'll run a simplified Optuna loop.
- `run_steering_sweep` in pareto.py returns `{"refusals": [...], "kl": [...], "multipliers": [...]}` — this is what `plot_pareto_frontier` expects.
- `pyyaml` is available transitively via transformers (confirmed in pyproject.toml dependencies).

---

### Task 1: Create directory structure and GPT-2 config

**Files:**
- Create: `experiments/__init__.py` (empty)
- Create: `experiments/configs/gpt2.yaml`

**Step 1: Create the directories and config file**

```bash
mkdir -p experiments/configs
touch experiments/__init__.py
```

Create `experiments/configs/gpt2.yaml`:

```yaml
# GPT-2 (124M) — pipeline validation only.
# GPT-2 is NOT safety-trained, so abliteration effects will be minimal.
# This config validates the pipeline runs end-to-end.

model: gpt2
quantization: null

seed: 42

abliteration:
  n_trials: 5            # Minimal — just prove the loop works
  n_startup_trials: 3    # Most trials are random exploration
  harmful_prompts: 20    # Small prompt set for speed

steering:
  layers: [5, 6, 7]      # Middle layers of GPT-2's 12
  multipliers: [0.5, 1.0, 1.5, 2.0, 3.0]
  batch_size: 4

benchmark:
  gsm8k: false            # GPT-2 can't do math
  gsm8k_limit: 20
  perplexity: true
  perplexity_limit: 10
```

**Step 2: Commit**

```bash
git add experiments/ experiments/configs/gpt2.yaml
git commit -m "chore: add experiments directory and GPT-2 config"
```

---

### Task 2: Write the config loader and dataclass

**Files:**
- Create: `experiments/config.py`
- Create: `tests/test_experiment_config.py`

**Step 1: Write the failing test**

Create `tests/test_experiment_config.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Write minimal implementation**

Create `experiments/config.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add experiments/config.py tests/test_experiment_config.py
git commit -m "feat: add experiment config loader with YAML parsing"
```

---

### Task 3: Write the seeding utility

**Files:**
- Modify: `experiments/config.py` (add `seed_everything` function)
- Modify: `tests/test_experiment_config.py` (add seed test)

**Step 1: Write the failing test**

Add to `tests/test_experiment_config.py`:

```python
class TestSeedEverything:
    def test_seeds_are_deterministic(self):
        """After seeding, random values should be reproducible."""
        import random

        import numpy as np
        import torch

        from experiments.config import seed_everything

        seed_everything(42)
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        seed_everything(42)
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_config.py::TestSeedEverything -v`
Expected: FAIL (ImportError — `seed_everything` doesn't exist)

**Step 3: Write minimal implementation**

Add to `experiments/config.py`:

```python
def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add experiments/config.py tests/test_experiment_config.py
git commit -m "feat: add seed_everything for reproducible experiments"
```

---

### Task 4: Write the experiment runner — steps 1-2 (load model + baseline benchmark)

**Files:**
- Create: `experiments/run_comparison.py`
- Create: `tests/test_experiment_runner.py`

This is the core script. We build it incrementally — this task covers loading the model and running the baseline benchmark (steps 1-2 of the pipeline).

**Step 1: Write the failing test**

Create `tests/test_experiment_runner.py`:

```python
"""Tests for the experiment runner pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
        from experiments.run_comparison import run_abliteration
        from shade.benchmark import BenchmarkResult

        mock_model = MagicMock()
        mock_model.get_layers.return_value = list(range(12))
        mock_model.get_abliterable_components.return_value = ["attn.o_proj", "mlp.down_proj"]
        mock_model.get_residuals_batched.return_value = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.get_score.return_value = ((0.1, 0.2), 0.05, 3)
        mock_evaluator.bad_prompts = [MagicMock()] * 10

        # We mock the internal Optuna study
        with patch("experiments.run_comparison.run_benchmark") as mock_bench:
            mock_bench.return_value = BenchmarkResult(
                model_name="gpt2 (abliterated)",
                refusals=3,
                total_prompts=10,
                refusal_rate=0.3,
                kl_divergence=0.05,
                gsm8k_accuracy=None,
                gsm8k_total=None,
                perplexity=70.0,
            )
            with patch("experiments.run_comparison.optuna"):
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_runner.py::TestRunBaseline -v`
Expected: FAIL (ImportError — module doesn't exist)

**Step 3: Write minimal implementation**

Create `experiments/run_comparison.py`:

```python
"""Experiment runner: baseline → abliterate → steer → benchmark → Pareto.

Usage:
    python -m experiments.run_comparison experiments/configs/gpt2.yaml
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F

from shade.benchmark import BenchmarkResult, run_benchmark
from shade.results import save_result

logger = logging.getLogger("shade.experiment")


def run_baseline(
    model: object,
    evaluator: object,
    model_name: str,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> BenchmarkResult:
    """Step 2: Run baseline benchmark on the unmodified model."""
    logger.info("Running baseline benchmark...")
    start = time.time()

    result = run_benchmark(
        model=model,
        evaluator=evaluator,
        model_name=f"{model_name} (baseline)",
        run_gsm8k=run_gsm8k,
        gsm8k_limit=gsm8k_limit,
        run_perplexity=run_perplexity,
        perplexity_limit=perplexity_limit,
    )

    elapsed = time.time() - start
    logger.info(
        "Baseline: refusals=%d/%d, KL=%.4f, perplexity=%s (%.1fs)",
        result.refusals,
        result.total_prompts,
        result.kl_divergence,
        f"{result.perplexity:.2f}" if result.perplexity else "N/A",
        elapsed,
    )
    return result


def run_abliteration(
    model: object,
    evaluator: object,
    model_name: str,
    n_trials: int = 5,
    n_startup_trials: int = 3,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> BenchmarkResult:
    """Step 3-4: Run Optuna abliteration search, then benchmark the best trial."""
    from shade.model import AbliterationParameters
    from shade.utils import load_prompts

    logger.info("Running abliteration optimization (%d trials)...", n_trials)

    # Compute refusal directions
    settings = model.settings
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0), p=2, dim=1
    )

    del good_residuals, bad_residuals

    last_layer_index = len(model.get_layers()) - 1
    best_trial_data = {"refusals": float("inf"), "kl": float("inf"), "params": None, "direction_index": None}

    def objective(trial):
        direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
        direction_index = trial.suggest_float(
            "direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index
        )
        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position", 0.6 * last_layer_index, 1.0 * last_layer_index
            )
            min_weight_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance", 1.0, 0.6 * last_layer_index
            )
            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight_frac * max_weight),
                min_weight_distance=min_weight_distance,
            )

        model.reset_model()
        model.abliterate(refusal_directions, direction_index, parameters)
        score, kl_divergence, refusals = evaluator.get_score()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        logger.info(
            "  Trial %d: refusals=%d, KL=%.4f",
            trial.number + 1, refusals, kl_divergence,
        )

        # Track best by lowest refusals, then lowest KL
        if (refusals, kl_divergence) < (best_trial_data["refusals"], best_trial_data["kl"]):
            from dataclasses import asdict
            best_trial_data["refusals"] = refusals
            best_trial_data["kl"] = kl_divergence
            best_trial_data["params"] = {k: asdict(v) for k, v in parameters.items()}
            best_trial_data["direction_index"] = direction_index

        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=True,
        ),
        directions=["minimize", "minimize"],
    )
    study.optimize(objective, n_trials=n_trials)

    # Restore best trial and benchmark it
    logger.info("Restoring best trial (refusals=%d, KL=%.4f)...",
                best_trial_data["refusals"], best_trial_data["kl"])
    model.reset_model()
    if best_trial_data["params"] is not None:
        params = {
            k: AbliterationParameters(**v) for k, v in best_trial_data["params"].items()
        }
        model.abliterate(refusal_directions, best_trial_data["direction_index"], params)

    result = run_benchmark(
        model=model,
        evaluator=evaluator,
        model_name=f"{model_name} (abliterated)",
        run_gsm8k=run_gsm8k,
        gsm8k_limit=gsm8k_limit,
        run_perplexity=run_perplexity,
        perplexity_limit=perplexity_limit,
    )

    logger.info(
        "Abliterated: refusals=%d/%d, KL=%.4f, perplexity=%s",
        result.refusals, result.total_prompts, result.kl_divergence,
        f"{result.perplexity:.2f}" if result.perplexity else "N/A",
    )
    return result


def run_steering_sweep(
    model: object,
    evaluator: object,
    model_name: str,
    layers: list[int],
    multipliers: list[float],
    batch_size: int = 4,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> list[BenchmarkResult]:
    """Step 5-6: Reset model, train steering vector, sweep multipliers."""
    from shade.steering import train_steering_vector_from_prompts
    from shade.utils import load_prompts

    logger.info("Training steering vector (layers=%s)...", layers)
    model.reset_model()

    settings = model.settings
    harmless = load_prompts(settings, settings.good_evaluation_prompts)
    harmful = load_prompts(settings, settings.bad_evaluation_prompts)

    base_model = model.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    sv = train_steering_vector_from_prompts(
        model=base_model,
        tokenizer=model.tokenizer,
        harmless_prompts=harmless,
        harmful_prompts=harmful,
        layers=layers,
        batch_size=batch_size,
        show_progress=False,
    )

    results = []
    for mult in multipliers:
        logger.info("Benchmarking steering multiplier x%.1f...", mult)
        with sv.apply(base_model, multiplier=mult):
            # Reinitialize evaluator inside the steering context
            from shade.evaluator import Evaluator
            steered_eval = Evaluator(settings, model)
            result = run_benchmark(
                model=model,
                evaluator=steered_eval,
                model_name=f"{model_name} (steered x{mult})",
                run_gsm8k=run_gsm8k,
                gsm8k_limit=gsm8k_limit,
                run_perplexity=run_perplexity,
                perplexity_limit=perplexity_limit,
            )
        results.append(result)
        logger.info(
            "  Steered x%.1f: refusals=%d/%d, KL=%.4f",
            mult, result.refusals, result.total_prompts, result.kl_divergence,
        )

    return results


def run_experiment(config_path: str) -> None:
    """Run the full experiment pipeline from a YAML config."""
    from experiments.config import ExperimentConfig, load_config, seed_everything
    from shade.compare import format_comparison_table
    from shade.config import Settings
    from shade.evaluator import Evaluator
    from shade.model import Model

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(config_path)
    logger.info("Loaded config for model: %s", config.model)

    # Create results directory
    model_slug = config.model.replace("/", "--")
    results_dir = Path("results") / "experiments" / model_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    seed_everything(config.seed)
    logger.info("Seeded with %d", config.seed)

    # Step 1: Load model
    logger.info("Step 1/9: Loading model %s...", config.model)
    settings_kwargs = {"model": config.model, "_cli_parse_args": False}
    if config.quantization == "bnb_4bit":
        settings_kwargs["quantization"] = "bnb_4bit"
    settings = Settings(**settings_kwargs)
    model = Model(settings)
    torch.set_grad_enabled(False)

    # Step 2: Baseline benchmark
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        logger.info("Step 2/9: Baseline already exists, skipping.")
        from shade.results import load_result
        baseline = load_result(baseline_path)
    else:
        logger.info("Step 2/9: Running baseline benchmark...")
        evaluator = Evaluator(settings, model)
        baseline = run_baseline(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        save_result(baseline, baseline_path)
        logger.info("Saved baseline to %s", baseline_path)

    # Step 3-4: Abliterate + benchmark
    abliterated_path = results_dir / "abliterated.json"
    if abliterated_path.exists():
        logger.info("Steps 3-4/9: Abliterated result already exists, skipping.")
        from shade.results import load_result
        abliterated = load_result(abliterated_path)
    else:
        logger.info("Steps 3-4/9: Running abliteration optimization...")
        evaluator = Evaluator(settings, model)
        abliterated = run_abliteration(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            n_trials=config.abliteration.n_trials,
            n_startup_trials=config.abliteration.n_startup_trials,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        save_result(abliterated, abliterated_path)
        logger.info("Saved abliterated result to %s", abliterated_path)

    # Step 5-6: Steering sweep
    steered_results = []
    all_steered_exist = all(
        (results_dir / f"steered_{mult}.json").exists()
        for mult in config.steering.multipliers
    )
    if all_steered_exist:
        logger.info("Steps 5-6/9: Steered results already exist, skipping.")
        from shade.results import load_result
        for mult in config.steering.multipliers:
            steered_results.append(load_result(results_dir / f"steered_{mult}.json"))
    else:
        logger.info("Steps 5-6/9: Running steering sweep...")
        evaluator = Evaluator(settings, model)
        steered_results = run_steering_sweep(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            layers=config.steering.layers,
            multipliers=config.steering.multipliers,
            batch_size=config.steering.batch_size,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        for result, mult in zip(steered_results, config.steering.multipliers):
            path = results_dir / f"steered_{mult}.json"
            save_result(result, path)
            logger.info("Saved steered x%.1f result to %s", mult, path)

    # Step 7: Save comparison JSON
    logger.info("Step 7/9: Saving comparison...")
    all_results = [baseline, abliterated] + steered_results
    from shade.compare import format_comparison_json
    comparison_path = results_dir / "comparison.json"
    comparison_path.write_text(format_comparison_json(all_results))

    # Step 8: Generate Pareto frontier plot
    logger.info("Step 8/9: Generating Pareto frontier plot...")
    try:
        from shade.pareto import plot_pareto_frontier

        # Build data dicts that plot_pareto_frontier expects
        abliteration_data = {
            "refusals": [abliterated.refusals],
            "kl": [abliterated.kl_divergence],
        }
        steering_data = {
            "refusals": [r.refusals for r in steered_results],
            "kl": [r.kl_divergence for r in steered_results],
            "multipliers": config.steering.multipliers,
        }
        plot_pareto_frontier(
            abliteration_data=abliteration_data,
            steering_data=steering_data,
            total_prompts=baseline.total_prompts,
            model_name=config.model,
            save_path=str(results_dir / "pareto.png"),
        )
        logger.info("Saved Pareto plot to %s", results_dir / "pareto.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping Pareto plot")

    # Step 9: Print comparison table
    logger.info("Step 9/9: Results summary")
    print("\n" + format_comparison_table(all_results) + "\n")
    logger.info("All results saved to %s", results_dir)


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: python -m experiments.run_comparison <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_experiment_runner.py::TestRunBaseline -v`
Expected: PASS

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: All PASS (no regressions)

**Step 5: Commit**

```bash
git add experiments/run_comparison.py tests/test_experiment_runner.py
git commit -m "feat: add experiment runner with baseline, abliteration, and steering steps"
```

---

### Task 5: Write integration test — full pipeline with GPT-2

**Files:**
- Create: `tests/test_experiment_integration.py`

This is the critical test. It runs the full pipeline on GPT-2 end-to-end (no mocks). It proves the experiment runner works before running on real models.

**Important:** GPT-2 is NOT safety-trained, so abliteration effects will be minimal. The test validates the pipeline runs without errors, not that abliteration is effective.

**Step 1: Write the integration test**

Create `tests/test_experiment_integration.py`:

```python
"""Integration test: run the full experiment pipeline on GPT-2.

This test proves the experiment runner works end-to-end with a real model.
GPT-2 is not safety-trained, so abliteration effects will be minimal.
The test validates correctness, not effectiveness.

Run with: uv run pytest tests/test_experiment_integration.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestFullExperimentPipeline:
    def test_gpt2_experiment_end_to_end(self, tmp_path):
        """Run the full pipeline on GPT-2 and verify all outputs exist."""
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

        # Monkey-patch results dir to use tmp_path
        import experiments.run_comparison as runner

        original_run = runner.run_experiment

        def patched_run(cfg_path):
            """Redirect results to tmp_path."""
            from experiments.config import load_config, seed_everything
            from shade.config import Settings
            from shade.evaluator import Evaluator
            from shade.model import Model

            import logging
            import torch

            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

            config = load_config(cfg_path)
            seed_everything(config.seed)

            results_dir = tmp_path / "results"
            results_dir.mkdir(exist_ok=True)

            settings = Settings(model=config.model, _cli_parse_args=False)
            model = Model(settings)
            torch.set_grad_enabled(False)

            # Baseline
            evaluator = Evaluator(settings, model)
            baseline = runner.run_baseline(
                model=model, evaluator=evaluator, model_name=config.model,
                run_perplexity=True, perplexity_limit=3,
            )
            from shade.results import save_result
            save_result(baseline, results_dir / "baseline.json")

            # Abliteration (2 trials only)
            evaluator2 = Evaluator(settings, model)
            abliterated = runner.run_abliteration(
                model=model, evaluator=evaluator2, model_name=config.model,
                n_trials=2, n_startup_trials=2,
                run_perplexity=True, perplexity_limit=3,
            )
            save_result(abliterated, results_dir / "abliterated.json")

            # Steering (2 multipliers)
            evaluator3 = Evaluator(settings, model)
            steered = runner.run_steering_sweep(
                model=model, evaluator=evaluator3, model_name=config.model,
                layers=[5, 6], multipliers=[1.0, 2.0], batch_size=2,
                run_perplexity=True, perplexity_limit=3,
            )
            for result, mult in zip(steered, [1.0, 2.0]):
                save_result(result, results_dir / f"steered_{mult}.json")

            return baseline, abliterated, steered

        baseline, abliterated, steered = patched_run(str(config_path))

        # Verify baseline
        assert (tmp_path / "results" / "baseline.json").exists()
        assert baseline.total_prompts > 0
        assert baseline.perplexity is not None
        assert math.isfinite(baseline.perplexity)

        # Verify abliterated
        assert (tmp_path / "results" / "abliterated.json").exists()
        assert abliterated.total_prompts > 0
        assert math.isfinite(abliterated.kl_divergence)

        # Verify steered
        assert len(steered) == 2
        for result in steered:
            assert result.total_prompts > 0

        # Verify JSON round-trips correctly
        from shade.results import load_result
        loaded_baseline = load_result(tmp_path / "results" / "baseline.json")
        assert loaded_baseline.model_name == baseline.model_name
        assert loaded_baseline.perplexity == pytest.approx(baseline.perplexity)
```

**Step 2: Run the integration test**

Run: `uv run pytest tests/test_experiment_integration.py -v`

**Important:** This will take 2-5 minutes because it runs real Optuna trials on GPT-2. If it passes, the pipeline is proven to work.

Expected: PASS (may take a while)

**Step 3: Commit**

```bash
git add tests/test_experiment_integration.py
git commit -m "test: add full pipeline integration test for experiment runner"
```

---

### Task 6: Add TinyLlama config

**Files:**
- Create: `experiments/configs/tinyllama-1.1b.yaml`

**Step 1: Create the config**

Create `experiments/configs/tinyllama-1.1b.yaml`:

```yaml
# TinyLlama-1.1B (4-bit) — first real abliteration test.
# LLaMA-architecture model with basic instruction following.
# Runs on GTX 1660 Ti (6GB) with 4-bit quantization.

model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
quantization: bnb_4bit

seed: 42

abliteration:
  n_trials: 20
  n_startup_trials: 10
  harmful_prompts: 50

steering:
  layers: [8, 9, 10, 11, 12]    # Later layers of TinyLlama's 22
  multipliers: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  batch_size: 4

benchmark:
  gsm8k: false            # TinyLlama has limited math ability
  gsm8k_limit: 20
  perplexity: true
  perplexity_limit: 20
```

**Step 2: Commit**

```bash
git add experiments/configs/tinyllama-1.1b.yaml
git commit -m "chore: add TinyLlama-1.1B experiment config"
```

---

### Task 7: Add Colab-tier model configs

**Files:**
- Create: `experiments/configs/llama-3.2-3b.yaml`
- Create: `experiments/configs/phi-3-mini.yaml`
- Create: `experiments/configs/gemma-2-2b.yaml`

**Step 1: Create configs**

Create `experiments/configs/llama-3.2-3b.yaml`:

```yaml
# Llama 3.2-3B Instruct — real safety-trained model with strong refusal behavior.
# Requires Colab T4 (16GB) or better.

model: meta-llama/Llama-3.2-3B-Instruct
quantization: bnb_4bit

seed: 42

abliteration:
  n_trials: 30
  n_startup_trials: 15
  harmful_prompts: 100

steering:
  layers: [12, 13, 14, 15, 16, 17]  # Middle-to-late layers of 28 total
  multipliers: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
  batch_size: 4

benchmark:
  gsm8k: true
  gsm8k_limit: 50
  perplexity: true
  perplexity_limit: 30
```

Create `experiments/configs/phi-3-mini.yaml`:

```yaml
# Phi-3-mini-4k-instruct — Microsoft's small but capable model.
# Good reasoning ability for its size. Requires Colab T4.

model: microsoft/Phi-3-mini-4k-instruct
quantization: bnb_4bit

seed: 42

abliteration:
  n_trials: 30
  n_startup_trials: 15
  harmful_prompts: 100

steering:
  layers: [14, 15, 16, 17, 18, 19]  # Middle-to-late layers of 32 total
  multipliers: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
  batch_size: 4

benchmark:
  gsm8k: true
  gsm8k_limit: 50
  perplexity: true
  perplexity_limit: 30
```

Create `experiments/configs/gemma-2-2b.yaml`:

```yaml
# Gemma-2-2B-it — Google's instruction-tuned model.
# SAE support available via SAELens for interpretability analysis.

model: google/gemma-2-2b-it
quantization: bnb_4bit

seed: 42

abliteration:
  n_trials: 30
  n_startup_trials: 15
  harmful_prompts: 100

steering:
  layers: [10, 11, 12, 13, 14, 15]  # Middle-to-late layers of 26 total
  multipliers: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
  batch_size: 4

benchmark:
  gsm8k: true
  gsm8k_limit: 50
  perplexity: true
  perplexity_limit: 30
```

**Step 2: Commit**

```bash
git add experiments/configs/
git commit -m "chore: add Colab-tier experiment configs (Llama-3.2-3B, Phi-3-mini, Gemma-2-2B)"
```

---

### Task 8: Add `__main__.py` for `python -m experiments.run_comparison`

**Files:**
- Create: `experiments/__main__.py`

**Step 1: Create the entry point**

Create `experiments/__main__.py`:

```python
"""Allow running as: python -m experiments.run_comparison <config.yaml>"""

from experiments.run_comparison import main

main()
```

Wait — actually this should be `experiments/__main__.py` to support `python -m experiments <config.yaml>`. But the design says `python -m experiments.run_comparison`. Let me do both.

Create `experiments/__main__.py`:

```python
"""Allow running as: python -m experiments <config.yaml>"""

from experiments.run_comparison import main

main()
```

**Step 2: Add results/ to .gitignore**

Append to `.gitignore`:

```
# Experiment results (generated, potentially large)
results/experiments/
```

**Step 3: Commit**

```bash
git add experiments/__main__.py .gitignore
git commit -m "chore: add experiments __main__.py entry point and gitignore results"
```

---

### Task 9: Run full pipeline on GPT-2 and verify results

This is the final validation task. No new code — just run the experiment and verify it produces real results.

**Step 1: Run the experiment**

```bash
uv run python -m experiments experiments/configs/gpt2.yaml
```

This will:
1. Load GPT-2
2. Run baseline benchmark
3. Run 5 Optuna abliteration trials
4. Train steering vector and sweep 5 multipliers
5. Save all results to `results/experiments/gpt2/`
6. Generate Pareto frontier plot
7. Print comparison table

**Step 2: Verify output files exist**

```bash
ls -la results/experiments/gpt2/
```

Expected files:
- `baseline.json`
- `abliterated.json`
- `steered_0.5.json`, `steered_1.0.json`, `steered_1.5.json`, `steered_2.0.json`, `steered_3.0.json`
- `comparison.json`
- `pareto.png`

**Step 3: Verify JSON content is sane**

```bash
python -c "
import json
from pathlib import Path

results_dir = Path('results/experiments/gpt2')
for f in sorted(results_dir.glob('*.json')):
    data = json.loads(f.read_text())
    if 'results' in data:
        # comparison.json
        print(f'{f.name}: {len(data[\"results\"])} results')
    else:
        print(f'{f.name}: refusals={data[\"refusals\"]}/{data[\"total_prompts\"]}, '
              f'KL={data[\"kl_divergence\"]:.4f}, '
              f'perplexity={data.get(\"perplexity\", \"N/A\")}')
"
```

**Step 4: Run all tests to verify no regressions**

```bash
uv run pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_experiment_integration.py -v
uv run ruff check .
uv run ruff format --check .
```

Expected: All pass, lint clean.

**Step 5: Commit results directory structure and final state**

```bash
git add experiments/ tests/
git commit -m "feat: complete experiment runner pipeline — validated on GPT-2"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Directory structure + GPT-2 config | `experiments/`, `experiments/configs/gpt2.yaml` |
| 2 | Config loader + dataclass | `experiments/config.py`, `tests/test_experiment_config.py` |
| 3 | Seeding utility | `experiments/config.py` (modify) |
| 4 | Experiment runner (all 9 steps) | `experiments/run_comparison.py`, `tests/test_experiment_runner.py` |
| 5 | Integration test with real GPT-2 | `tests/test_experiment_integration.py` |
| 6 | TinyLlama config | `experiments/configs/tinyllama-1.1b.yaml` |
| 7 | Colab model configs | 3 YAML files |
| 8 | Entry point + gitignore | `experiments/__main__.py`, `.gitignore` |
| 9 | Run GPT-2 experiment + verify | No new code — validation only |

**Total: 9 tasks, ~6 new files, ~3 test files.**

After Task 9, the experiment runner is proven to work. The next step is running it on TinyLlama locally, then on Colab for Llama-3.2-3B/Phi-3/Gemma-2.
