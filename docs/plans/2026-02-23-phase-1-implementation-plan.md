# Phase 1: Multi-Technique Pioneer — Full Research Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first tool combining abliteration + activation steering, run experiments on 3-5 models, produce novel Pareto frontier findings, and publish an arXiv paper.

**Architecture:** Three phases — (A) engineering the tool, (B) running experiments and producing findings, (C) writing and publishing the paper. New modules: `benchmark.py`, `steering.py`, `compare.py`, `pareto.py`, `sae_analysis.py`.

**Tech Stack:** Python 3.10+, Click CLI, steering-vectors>=0.12.2, sae-lens>=6.37.3, matplotlib, datasets, torch

**Design doc:** `docs/plans/2026-02-23-phase-1-multi-technique-pioneer-design.md`
**Research:** `docs/plans/2026-02-23-phase-1-research-findings.md`

---

# Phase 1A: Engineering the Tool (Tasks 1-13)

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:24-42` (core deps), `pyproject.toml:44-52` (research extras)

**Step 1: Add dependencies**

In `pyproject.toml`, add to `[project.dependencies]`:
```toml
steering-vectors>=0.12.2
```

Add to `[project.optional-dependencies]` research group:
```toml
research = [
    "geom-median>=0.1",
    "imageio>=2.3",
    "matplotlib>=3.10",
    "numpy>=2.0",
    "pacmap>=0.7",
    "scikit-learn>=1.6",
    "sae-lens>=6.37",
]
```

**Step 2: Install and verify**

Run: `uv sync --all-extras`
Run: `uv run python3 -c "from steering_vectors import train_steering_vector; print('steering OK')"`
Run: `uv run python3 -c "from sae_lens import SAE; print('sae-lens OK')"`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add steering-vectors and sae-lens dependencies"
```

---

## Task 2: GSM8K Answer Parsing

**Files:**
- Create: `src/shade/benchmark.py`
- Create: `tests/test_benchmark.py`

**Step 1: Write failing tests**

Create `tests/test_benchmark.py`:

```python
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
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

Create `src/shade/benchmark.py`:

```python
"""Benchmark utilities for capability-aware model evaluation."""

from __future__ import annotations

import re


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style response."""
    if not text or not text.strip():
        return None

    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    match = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()

    match = re.search(r"the answer is\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/test_benchmark.py -v`

**Step 5: Commit**

```bash
git add src/shade/benchmark.py tests/test_benchmark.py
git commit -m "feat: add GSM8K answer extraction"
```

---

## Task 3: GSM8K Accuracy Computation + Dataset Loading

**Files:**
- Modify: `src/shade/benchmark.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
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


class TestLoadGsm8kProblems:
    def test_returns_questions_and_answers(self):
        from shade.benchmark import load_gsm8k_problems
        questions, answers = load_gsm8k_problems(limit=5)
        assert len(questions) == 5
        assert all("####" in a for a in answers)

    def test_limit(self):
        from shade.benchmark import load_gsm8k_problems
        questions, answers = load_gsm8k_problems(limit=3)
        assert len(questions) == 3
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_benchmark.py::TestComputeGsm8kAccuracy -v`

**Step 3: Implement**

Append to `src/shade/benchmark.py`:

```python
def compute_gsm8k_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy by comparing extracted numeric answers."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if (pa := extract_gsm8k_answer(p)) is not None
        and pa == extract_gsm8k_answer(r)
    )
    return correct / len(predictions)


def load_gsm8k_problems(limit: int = 50) -> tuple[list[str], list[str]]:
    """Load GSM8K math problems from HuggingFace."""
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]")
    return [row["question"] for row in dataset], [row["answer"] for row in dataset]
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/test_benchmark.py -v`

**Step 5: Commit**

```bash
git add src/shade/benchmark.py tests/test_benchmark.py
git commit -m "feat: add GSM8K accuracy computation and dataset loading"
```

---

## Task 4: Perplexity Metric (Wikitext-2)

**Files:**
- Modify: `src/shade/benchmark.py`
- Modify: `tests/test_benchmark.py`

Perplexity is a standard metric the paper needs. It measures general language modeling capability.

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
class TestComputePerplexity:
    def test_with_mock_model(self):
        import torch
        from unittest.mock import MagicMock
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

        ppl = compute_perplexity(mock_model, mock_tokenizer, ["Hello world test."])
        assert isinstance(ppl, float)
        assert ppl > 0
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_benchmark.py::TestComputePerplexity -v`

**Step 3: Implement**

Append to `src/shade/benchmark.py`:

```python
def compute_perplexity(
    model: object,
    tokenizer: object,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute perplexity on a list of texts.

    Uses sliding window approach for texts longer than max_length.
    """
    import torch
    import torch.nn.functional as F

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Shift: predict token i+1 from position i
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return float(torch.exp(torch.tensor(avg_loss)).item())


def load_wikitext2_samples(limit: int = 50) -> list[str]:
    """Load Wikitext-2 test samples for perplexity evaluation."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{limit}]")
    return [row["text"] for row in dataset if row["text"].strip()]
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/test_benchmark.py -v`

**Step 5: Commit**

```bash
git add src/shade/benchmark.py tests/test_benchmark.py
git commit -m "feat: add Wikitext-2 perplexity metric"
```

---

## Task 5: BenchmarkResult Dataclass + `run_benchmark`

**Files:**
- Modify: `src/shade/benchmark.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
from dataclasses import asdict


class TestBenchmarkResult:
    def test_fields(self):
        from shade.benchmark import BenchmarkResult
        r = BenchmarkResult(
            model_name="test", refusals=12, total_prompts=100,
            refusal_rate=0.12, kl_divergence=0.043,
            gsm8k_accuracy=None, gsm8k_total=None,
            perplexity=None,
        )
        assert r.refusal_rate == pytest.approx(0.12)
        assert r.perplexity is None

    def test_to_dict(self):
        from shade.benchmark import BenchmarkResult
        r = BenchmarkResult(
            model_name="m", refusals=5, total_prompts=50,
            refusal_rate=0.1, kl_divergence=0.02,
            gsm8k_accuracy=0.45, gsm8k_total=50, perplexity=12.3,
        )
        d = asdict(r)
        assert d["perplexity"] == pytest.approx(12.3)
```

**Step 2: Implement**

Append to `src/shade/benchmark.py`:

```python
from dataclasses import dataclass
import json as _json


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    refusals: int
    total_prompts: int
    refusal_rate: float
    kl_divergence: float
    gsm8k_accuracy: float | None
    gsm8k_total: int | None
    perplexity: float | None


def run_benchmark(
    model: object,
    evaluator: object,
    model_name: str,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> BenchmarkResult:
    """Run a complete benchmark on a model."""
    import torch.nn.functional as F

    refusals = evaluator.count_refusals()
    total_prompts = len(evaluator.bad_prompts)
    refusal_rate = refusals / total_prompts if total_prompts > 0 else 0.0

    logprobs = model.get_logprobs_batched(evaluator.good_prompts)
    kl_divergence = F.kl_div(
        logprobs, evaluator.base_logprobs,
        reduction="batchmean", log_target=True,
    ).item()

    gsm8k_accuracy = None
    gsm8k_total = None
    if run_gsm8k:
        from shade.utils import Prompt
        questions, answers = load_gsm8k_problems(limit=gsm8k_limit)
        gsm8k_total = len(questions)
        prompts = [Prompt(system="Solve this math problem. Show your work.", user=q) for q in questions]
        responses = model.get_responses_batched(prompts)
        gsm8k_accuracy = compute_gsm8k_accuracy(responses, answers)

    perplexity = None
    if run_perplexity:
        texts = load_wikitext2_samples(limit=perplexity_limit)
        perplexity = compute_perplexity(model.model, model.tokenizer, texts)

    return BenchmarkResult(
        model_name=model_name, refusals=refusals, total_prompts=total_prompts,
        refusal_rate=refusal_rate, kl_divergence=kl_divergence,
        gsm8k_accuracy=gsm8k_accuracy, gsm8k_total=gsm8k_total,
        perplexity=perplexity,
    )


def format_benchmark_table(result: BenchmarkResult) -> str:
    """Format a BenchmarkResult as a human-readable text table."""
    lines = [
        f"Benchmark Results: {result.model_name}",
        "=" * 50,
        f"  Refusals:       {result.refusals}/{result.total_prompts} ({result.refusal_rate:.1%})",
        f"  KL Divergence:  {result.kl_divergence:.4f}",
    ]
    if result.perplexity is not None:
        lines.append(f"  Perplexity:     {result.perplexity:.2f}")
    if result.gsm8k_accuracy is not None:
        lines.append(f"  GSM8K Accuracy: {result.gsm8k_accuracy:.1%} ({result.gsm8k_total} problems)")
    lines.append("=" * 50)
    return "\n".join(lines)


def format_benchmark_json(result: BenchmarkResult) -> str:
    """Format a BenchmarkResult as JSON."""
    from dataclasses import asdict
    return _json.dumps(asdict(result), indent=2)
```

**Step 3: Run to verify pass**

Run: `uv run pytest tests/test_benchmark.py -v`

**Step 4: Commit**

```bash
git add src/shade/benchmark.py tests/test_benchmark.py
git commit -m "feat: add BenchmarkResult, run_benchmark, and formatting"
```

---

## Task 6: Wire `shade benchmark` CLI

**Files:**
- Modify: `src/shade/cli.py:646-679` (replace fake benchmark)

**Step 1: Replace benchmark command**

Replace the benchmark command (cli.py lines 646-679) with:

```python
@cli.command()
@click.argument("model_id", required=False)
@click.option("--gsm8k", is_flag=True, help="Run GSM8K math reasoning evaluation.")
@click.option("--gsm8k-limit", default=50, type=int, help="Number of GSM8K problems.")
@click.option("--perplexity", is_flag=True, help="Run Wikitext-2 perplexity evaluation.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table", help="Output format.")
def benchmark(model_id, gsm8k, gsm8k_limit, perplexity, output):
    """Run a capability-aware benchmark on a model."""
    from .model import Model
    from .config import Settings
    from .evaluator import Evaluator
    from .benchmark import run_benchmark, format_benchmark_table, format_benchmark_json

    if not model_id:
        print("[red]Please specify a model ID or path.[/red]")
        raise SystemExit(1)

    print(f"[bold]Loading model: {model_id}[/bold]")
    settings = Settings(model=model_id, _cli_parse_args=False)
    model = Model(settings)

    print("[bold]Running evaluation...[/bold]")
    evaluator = Evaluator(settings, model)
    result = run_benchmark(
        model=model, evaluator=evaluator, model_name=model_id,
        run_gsm8k=gsm8k, gsm8k_limit=gsm8k_limit,
        run_perplexity=perplexity,
    )

    from .utils import print as rprint
    rprint(format_benchmark_json(result) if output == "json" else format_benchmark_table(result))
```

**Step 2: Verify**

Run: `uv run shade benchmark --help`
Expected: shows `--gsm8k`, `--perplexity`, `--output` options

**Step 3: Commit**

```bash
git add src/shade/cli.py
git commit -m "feat: replace fake benchmark with capability-aware metrics"
```

---

## Task 7: Steering Module — Training + I/O

**Files:**
- Create: `src/shade/steering.py`
- Create: `tests/test_steering.py`

**Step 1: Write failing tests**

Create `tests/test_steering.py`:

```python
"""Tests for shade.steering module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_STUB_MODULES = [
    "peft", "transformers", "transformers.generation", "transformers.utils",
    "accelerate", "accelerate.utils", "bitsandbytes",
    "questionary", "rich", "rich.console", "rich.text", "rich.table",
    "rich.panel", "psutil", "steering_vectors",
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
        from shade.steering import save_steering_vector, load_steering_vector

        data = {"activations": torch.randn(3, 10)}
        path = tmp_path / "test.pt"
        save_steering_vector(data, path)
        loaded = load_steering_vector(path)
        assert "activations" in loaded
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_steering.py -v`

**Step 3: Implement**

Create `src/shade/steering.py`:

```python
"""Activation steering for LLM behavior modification."""

from __future__ import annotations

from pathlib import Path

import torch

from .utils import Prompt


def build_contrastive_pairs(
    harmless_prompts: list[Prompt],
    harmful_prompts: list[Prompt],
) -> list[tuple[str, str]]:
    """Build (positive, negative) pairs for steering training."""
    n = min(len(harmless_prompts), len(harmful_prompts))
    return [(harmless_prompts[i].user, harmful_prompts[i].user) for i in range(n)]


def save_steering_vector(vector: object, path: Path) -> None:
    """Save a steering vector to disk."""
    torch.save(vector, path)


def load_steering_vector(path: Path) -> object:
    """Load a steering vector from disk."""
    return torch.load(path, weights_only=False)


def train_steering_vector_from_prompts(
    model: object,
    tokenizer: object,
    harmless_prompts: list[Prompt],
    harmful_prompts: list[Prompt],
    layers: list[int] | None = None,
    batch_size: int = 4,
    show_progress: bool = True,
) -> object:
    """Train a steering vector from contrastive prompt pairs."""
    from steering_vectors import train_steering_vector

    pairs = build_contrastive_pairs(harmless_prompts, harmful_prompts)
    return train_steering_vector(
        model, tokenizer, pairs,
        layers=layers, batch_size=batch_size, show_progress=show_progress,
    )
```

**Step 4: Run to verify pass**

Run: `uv run pytest tests/test_steering.py -v`

**Step 5: Commit**

```bash
git add src/shade/steering.py tests/test_steering.py
git commit -m "feat: add steering vector training and I/O"
```

---

## Task 8: Wire `shade steer` CLI (train / apply / evaluate)

**Files:**
- Modify: `src/shade/cli.py`

**Step 1: Add the steer group**

Add after the `cuda` group in cli.py:

```python
@cli.group()
def steer():
    """Activation steering — train, apply, and evaluate steering vectors."""
    pass


@steer.command(name="train")
@click.argument("model_id")
@click.option("--output", "-o", default="steering_vector.pt", help="Output path.")
@click.option("--layers", "-l", default=None, help="Comma-separated layer indices.")
@click.option("--batch-size", default=4, type=int, help="Training batch size.")
def steer_train(model_id, output, layers, batch_size):
    """Train a steering vector from contrastive prompt pairs."""
    from .model import Model
    from .config import Settings
    from .steering import train_steering_vector_from_prompts, save_steering_vector
    from .utils import load_prompts

    settings = Settings(model=model_id, _cli_parse_args=False)
    model_wrapper = Model(settings)
    harmless = load_prompts(settings.good_evaluation_prompts)
    harmful = load_prompts(settings.bad_evaluation_prompts)

    parsed_layers = [int(x) for x in layers.split(",")] if layers else None
    base_model = model_wrapper.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    sv = train_steering_vector_from_prompts(
        model=base_model, tokenizer=model_wrapper.tokenizer,
        harmless_prompts=harmless, harmful_prompts=harmful,
        layers=parsed_layers, batch_size=batch_size,
    )
    from pathlib import Path
    save_steering_vector(sv, Path(output))
    print(f"[green]Saved to {output}[/green]")


@steer.command(name="apply")
@click.argument("model_id")
@click.option("--vector", "-v", required=True, help="Path to steering vector (.pt).")
@click.option("--multiplier", "-m", default=1.0, type=float, help="Steering strength.")
def steer_apply(model_id, vector, multiplier):
    """Chat with a model using an active steering vector."""
    from .model import Model
    from .config import Settings
    from .steering import load_steering_vector

    settings = Settings(model=model_id, _cli_parse_args=False)
    model_wrapper = Model(settings)
    from pathlib import Path
    sv = load_steering_vector(Path(vector))

    base_model = model_wrapper.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    print(f"[bold]Steering active (x{multiplier}). Type 'exit' to quit.[/bold]")
    with sv.apply(base_model, multiplier=multiplier):
        while True:
            try:
                user_input = input("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in ("exit", "quit"):
                break
            response = model_wrapper.stream_chat_response(
                [{"role": "user", "content": user_input}]
            )
            print(f"Model: {response}")


@steer.command(name="evaluate")
@click.argument("model_id")
@click.option("--vector", "-v", required=True, help="Path to steering vector (.pt).")
@click.option("--multiplier", "-m", default=1.0, type=float, help="Steering strength.")
@click.option("--gsm8k", is_flag=True, help="Run GSM8K evaluation.")
@click.option("--gsm8k-limit", default=50, type=int)
@click.option("--perplexity", is_flag=True, help="Run perplexity evaluation.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
def steer_evaluate(model_id, vector, multiplier, gsm8k, gsm8k_limit, perplexity, output):
    """Benchmark a model with a steering vector active."""
    from .model import Model
    from .config import Settings
    from .evaluator import Evaluator
    from .steering import load_steering_vector
    from .benchmark import run_benchmark, format_benchmark_table, format_benchmark_json

    settings = Settings(model=model_id, _cli_parse_args=False)
    model_wrapper = Model(settings)
    from pathlib import Path
    sv = load_steering_vector(Path(vector))

    base_model = model_wrapper.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    with sv.apply(base_model, multiplier=multiplier):
        evaluator = Evaluator(settings, model_wrapper)
        result = run_benchmark(
            model=model_wrapper, evaluator=evaluator,
            model_name=f"{model_id} (steered x{multiplier})",
            run_gsm8k=gsm8k, gsm8k_limit=gsm8k_limit,
            run_perplexity=perplexity,
        )

    from .utils import print as rprint
    rprint(format_benchmark_json(result) if output == "json" else format_benchmark_table(result))
```

**Step 2: Verify**

Run: `uv run shade steer --help`
Run: `uv run shade steer train --help`

**Step 3: Commit**

```bash
git add src/shade/cli.py
git commit -m "feat: add shade steer train/apply/evaluate CLI"
```

---

## Task 9: Comparison Module + Formatting

**Files:**
- Create: `src/shade/compare.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
import json


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
```

**Step 2: Implement**

Create `src/shade/compare.py`:

```python
"""Side-by-side comparison of abliteration vs activation steering."""

from __future__ import annotations

import json as _json
from dataclasses import asdict

from .benchmark import BenchmarkResult


def format_comparison_table(results: list[BenchmarkResult]) -> str:
    """Format multiple BenchmarkResults as a comparison table."""
    has_gsm8k = any(r.gsm8k_accuracy is not None for r in results)
    has_ppl = any(r.perplexity is not None for r in results)
    name_w = max(len(r.model_name) for r in results)
    name_w = max(name_w, len("Technique"))

    header = f"{'Technique':<{name_w}}  {'Refusals':>10}  {'KL Div':>10}"
    if has_ppl:
        header += f"  {'PPL':>8}"
    if has_gsm8k:
        header += f"  {'GSM8K':>8}"

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        line = f"{r.model_name:<{name_w}}  {r.refusals}/{r.total_prompts}:>{10 - len(str(r.refusals)) - 1}  {r.kl_divergence:>10.4f}" if r.kl_divergence > 0 else f"{r.model_name:<{name_w}}  {r.refusals}/{r.total_prompts}:>10  {'-':>10}"
        refusal_str = f"{r.refusals}/{r.total_prompts}"
        kl_str = f"{r.kl_divergence:.4f}" if r.kl_divergence > 0 else "-"
        line = f"{r.model_name:<{name_w}}  {refusal_str:>10}  {kl_str:>10}"
        if has_ppl:
            ppl_str = f"{r.perplexity:.2f}" if r.perplexity is not None else "-"
            line += f"  {ppl_str:>8}"
        if has_gsm8k:
            gsm_str = f"{r.gsm8k_accuracy:.1%}" if r.gsm8k_accuracy is not None else "-"
            line += f"  {gsm_str:>8}"
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def format_comparison_json(results: list[BenchmarkResult]) -> str:
    """Format multiple BenchmarkResults as JSON."""
    return _json.dumps({"results": [asdict(r) for r in results]}, indent=2)
```

**Step 3: Verify**

Run: `uv run pytest tests/test_benchmark.py::TestFormatComparisonTable tests/test_benchmark.py::TestFormatComparisonJson -v`

**Step 4: Commit**

```bash
git add src/shade/compare.py tests/test_benchmark.py
git commit -m "feat: add comparison table and JSON formatting"
```

---

## Task 10: Wire `shade compare` CLI

**Files:**
- Modify: `src/shade/cli.py`

**Step 1: Add compare command to cli.py**

Add the compare command (see design doc Section 2.3 for the full pipeline). This command runs all three benchmarks: base → abliterated → steered.

The implementation is identical to what was in the previous plan version (Task 10). Add it to cli.py.

**Step 2: Verify**

Run: `uv run shade compare --help`

**Step 3: Commit**

```bash
git add src/shade/cli.py
git commit -m "feat: add shade compare command"
```

---

## Task 11: Pareto Frontier Plotting Module

**Files:**
- Create: `src/shade/pareto.py`
- Modify: `tests/test_benchmark.py`

This is new — not in the old plan. Produces the Pareto frontier plots for the paper.

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
import numpy as np


class TestComputeParetoFrontier:
    def test_simple_frontier(self):
        from shade.pareto import compute_pareto_frontier_2d

        # 4 points: (refusals, kl). Minimizing both.
        points = np.array([
            [10, 0.5],   # Pareto (low refusals, high kl)
            [50, 0.1],   # Pareto (high refusals, low kl)
            [30, 0.3],   # Dominated by a mix
            [10, 0.1],   # Pareto (best on both!)
        ])
        mask = compute_pareto_frontier_2d(points)
        assert mask[3] is True   # dominates everything
        # Point 2 (30, 0.3) is dominated by point 3 (10, 0.1)
        assert mask[2] is False

    def test_all_pareto(self):
        from shade.pareto import compute_pareto_frontier_2d
        points = np.array([[1, 10.0], [5, 5.0], [10, 1.0]])
        mask = compute_pareto_frontier_2d(points)
        assert mask.all()
```

**Step 2: Implement**

Create `src/shade/pareto.py`:

```python
"""Pareto frontier computation and visualization."""

from __future__ import annotations

import numpy as np


def compute_pareto_frontier_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D Pareto frontier for minimization of both objectives.

    O(n log n) sort-and-sweep algorithm.

    Args:
        points: Shape (n, 2) array. Both columns minimized.

    Returns:
        Boolean mask of shape (n,) where True = Pareto-optimal.
    """
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    pareto_mask = np.zeros(len(points), dtype=bool)
    min_second = np.inf

    for i in range(len(sorted_points)):
        if sorted_points[i, 1] < min_second:
            min_second = sorted_points[i, 1]
            pareto_mask[sorted_idx[i]] = True

    return pareto_mask


def extract_optuna_trial_data(checkpoint_path: str) -> dict:
    """Extract all completed trial data from a Shade Optuna checkpoint.

    Returns dict with keys: refusals, kl, trial_numbers.
    """
    from optuna.storages.journal import (
        JournalFileBackend, JournalFileOpenLock, JournalStorage,
    )
    from optuna.trial import TrialState
    import optuna

    lock = JournalFileOpenLock(checkpoint_path)
    backend = JournalFileBackend(checkpoint_path, lock_obj=lock)
    storage = JournalStorage(backend)
    study = optuna.load_study(study_name="shade", storage=storage)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    return {
        "refusals": [t.user_attrs["refusals"] for t in completed],
        "kl": [t.user_attrs["kl_divergence"] for t in completed],
        "trial_numbers": [t.number for t in completed],
    }


def run_steering_sweep(
    model: object,
    evaluator: object,
    steering_vector: object,
    multipliers: list[float] | None = None,
) -> dict:
    """Sweep steering multiplier values and collect metrics.

    Returns dict with keys: refusals, kl, multipliers.
    """
    import torch.nn.functional as F

    if multipliers is None:
        multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results: dict = {"refusals": [], "kl": [], "multipliers": multipliers}

    base_model = model.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    for mult in multipliers:
        if mult == 0.0:
            refusals = evaluator.count_refusals()
            logprobs = model.get_logprobs_batched(evaluator.good_prompts)
        else:
            with steering_vector.apply(base_model, multiplier=mult):
                refusals = evaluator.count_refusals()
                logprobs = model.get_logprobs_batched(evaluator.good_prompts)

        kl = F.kl_div(
            logprobs, evaluator.base_logprobs,
            reduction="batchmean", log_target=True,
        ).item()
        results["refusals"].append(refusals)
        results["kl"].append(kl)

    return results


def plot_pareto_frontier(
    abliteration_data: dict,
    steering_data: dict,
    total_prompts: int,
    model_name: str = "",
    save_path: str | None = None,
) -> object:
    """Plot abliteration vs steering Pareto frontier.

    Two panels: raw metrics (arXiv style) and normalized (0-1).
    """
    import matplotlib.pyplot as plt

    abl_kl = np.array(abliteration_data["kl"])
    abl_ref = np.array(abliteration_data["refusals"], dtype=float)
    steer_kl = np.array(steering_data["kl"])
    steer_ref = np.array(steering_data["refusals"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: raw metrics
    ax1.scatter(abl_kl, abl_ref, c="royalblue", alpha=0.25, s=30)
    pts = np.column_stack([abl_kl, abl_ref])
    mask = compute_pareto_frontier_2d(pts)
    ax1.scatter(abl_kl[mask], abl_ref[mask], c="royalblue", s=80,
                edgecolors="navy", lw=1.5, label="Abliteration Pareto")
    pidx = np.where(mask)[0][np.argsort(abl_kl[mask])]
    ax1.plot(abl_kl[pidx], abl_ref[pidx], c="royalblue", lw=2)

    ax1.scatter(steer_kl, steer_ref, c="darkorange", s=80,
                edgecolors="saddlebrown", lw=1.5, label="Steering")
    sidx = np.argsort(steer_kl)
    ax1.plot(steer_kl[sidx], steer_ref[sidx], c="darkorange", lw=2, ls="--")
    for i, m in enumerate(steering_data["multipliers"]):
        ax1.annotate(f"x{m}", (steer_kl[i], steer_ref[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax1.set_xlabel("KL Divergence")
    ax1.set_ylabel(f"Refusals (out of {total_prompts})")
    ax1.set_title("Raw Metrics (lower-left = optimal)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Right: normalized
    abl_x = 1.0 - abl_ref / total_prompts
    abl_y = 1.0 / (1.0 + abl_kl)
    steer_x = 1.0 - steer_ref / total_prompts
    steer_y = 1.0 / (1.0 + steer_kl)

    ax2.scatter(abl_x, abl_y, c="royalblue", alpha=0.25, s=30)
    neg_pts = np.column_stack([-abl_x, -abl_y])
    mask2 = compute_pareto_frontier_2d(neg_pts)
    ax2.scatter(abl_x[mask2], abl_y[mask2], c="royalblue", s=80,
                edgecolors="navy", lw=1.5, label="Abliteration Pareto")
    ax2.scatter(steer_x, steer_y, c="darkorange", s=80,
                edgecolors="saddlebrown", lw=1.5, label="Steering")
    ax2.set_xlabel("Refusal Removal Rate")
    ax2.set_ylabel("Capability Preservation")
    ax2.set_title("Normalized (upper-right = optimal)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Abliteration vs Steering{' — ' + model_name if model_name else ''}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
```

**Step 3: Verify**

Run: `uv run pytest tests/test_benchmark.py::TestComputeParetoFrontier -v`

**Step 4: Commit**

```bash
git add src/shade/pareto.py tests/test_benchmark.py
git commit -m "feat: add Pareto frontier computation and plotting"
```

---

## Task 12: SAE Decomposition Module

**Files:**
- Create: `src/shade/sae_analysis.py`

This module decomposes the refusal direction into SAE features to show what it overlaps with. Uses SAELens + Gemma 2 2B (best pre-trained SAE ecosystem) or any model with available SAEs.

**Step 1: Write failing tests**

Append to `tests/test_benchmark.py`:

```python
class TestDecomposeDirection:
    def test_returns_top_features(self):
        from shade.sae_analysis import decompose_direction_into_features

        # Mock: refusal direction is a random vector
        direction = torch.randn(2304)  # Gemma 2B hidden dim
        # Mock SAE with encode method
        mock_sae = MagicMock()
        mock_sae.encode.return_value = torch.randn(1, 1, 16384)  # 16K features

        features = decompose_direction_into_features(direction, mock_sae, top_k=10)
        assert len(features) == 10
        assert all("index" in f and "activation" in f for f in features)
```

**Step 2: Implement**

Create `src/shade/sae_analysis.py`:

```python
"""SAE-based decomposition of refusal directions.

Decomposes abliteration refusal directions into interpretable sparse
autoencoder features using SAELens. Shows what the refusal direction
overlaps with (math, code, facts, etc.).
"""

from __future__ import annotations

import torch


def decompose_direction_into_features(
    direction: torch.Tensor,
    sae: object,
    top_k: int = 20,
) -> list[dict]:
    """Decompose a direction vector into SAE features.

    Args:
        direction: The refusal direction vector, shape (d_model,).
        sae: An SAELens SAE instance with .encode() method.
        top_k: Number of top-activating features to return.

    Returns:
        List of dicts with 'index' and 'activation' keys, sorted by activation.
    """
    # SAE expects (batch, seq, d_model)
    acts = sae.encode(direction.unsqueeze(0).unsqueeze(0))
    acts_flat = acts.squeeze()

    topk = acts_flat.abs().topk(top_k)
    return [
        {"index": idx.item(), "activation": acts_flat[idx].item()}
        for idx, val in zip(topk.indices, topk.values)
    ]


def compute_refusal_direction(
    model: object,
    tokenizer: object,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layer_index: int,
) -> torch.Tensor:
    """Compute the mean-difference refusal direction at a specific layer.

    Args:
        model: HuggingFace model.
        tokenizer: Model tokenizer.
        harmful_prompts: Prompts that elicit refusal.
        harmless_prompts: Prompts that elicit compliance.
        layer_index: Which transformer layer to extract from.

    Returns:
        Normalized refusal direction vector, shape (d_model,).
    """
    def get_activations(prompts: list[str]) -> torch.Tensor:
        acts = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # Take last token's hidden state at target layer
            hidden = outputs.hidden_states[layer_index + 1]  # +1 for embedding layer
            acts.append(hidden[0, -1, :])  # last token
        return torch.stack(acts)

    harmful_acts = get_activations(harmful_prompts)
    harmless_acts = get_activations(harmless_prompts)

    direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
    return direction / direction.norm()


def load_sae_for_model(
    model_name: str,
    layer_index: int,
    width: str = "16k",
) -> object:
    """Load a pre-trained SAE from SAELens registry.

    Supported models:
    - gemma-2-2b: gemma-scope-2b-pt-res-canonical
    - gpt2: gpt2-small-res-jb

    Args:
        model_name: HuggingFace model name.
        layer_index: Target layer.
        width: SAE width (e.g., "16k", "65k").

    Returns:
        SAELens SAE instance.
    """
    from sae_lens import SAE

    # Map model names to SAELens release IDs
    sae_registry = {
        "google/gemma-2-2b": {
            "release": "gemma-scope-2b-pt-res-canonical",
            "sae_id": f"layer_{layer_index}/width_{width}/canonical",
        },
        "openai-community/gpt2": {
            "release": "gpt2-small-res-jb",
            "sae_id": f"blocks.{layer_index}.hook_resid_post",
        },
    }

    if model_name not in sae_registry:
        raise ValueError(
            f"No pre-trained SAE available for {model_name}. "
            f"Supported: {list(sae_registry.keys())}"
        )

    config = sae_registry[model_name]
    return SAE.from_pretrained(
        release=config["release"],
        sae_id=config["sae_id"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def run_sae_analysis(
    model: object,
    tokenizer: object,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    model_name: str,
    layer_index: int,
    top_k: int = 20,
) -> dict:
    """Full SAE analysis pipeline.

    1. Compute refusal direction at target layer
    2. Load pre-trained SAE
    3. Decompose into features
    4. Return results dict

    Returns:
        Dict with direction, features, and metadata.
    """
    direction = compute_refusal_direction(
        model, tokenizer, harmful_prompts, harmless_prompts, layer_index,
    )

    sae = load_sae_for_model(model_name, layer_index)
    features = decompose_direction_into_features(direction, sae, top_k=top_k)

    return {
        "model_name": model_name,
        "layer_index": layer_index,
        "direction_norm": direction.norm().item(),
        "top_features": features,
        "n_harmful_prompts": len(harmful_prompts),
        "n_harmless_prompts": len(harmless_prompts),
    }
```

**Step 3: Verify**

Run: `uv run pytest tests/test_benchmark.py::TestDecomposeDirection -v`

**Step 4: Commit**

```bash
git add src/shade/sae_analysis.py tests/test_benchmark.py
git commit -m "feat: add SAE decomposition of refusal directions"
```

---

## Task 13: Full Test Suite + Lint

**Step 1:** Run: `uv run pytest tests/ -v --tb=short`
**Step 2:** Run: `uv run ruff check src/shade/benchmark.py src/shade/steering.py src/shade/compare.py src/shade/pareto.py src/shade/sae_analysis.py`
**Step 3:** Fix any issues.
**Step 4:** Commit: `git commit -m "chore: lint and fix Phase 1A test suite"`

---

# Phase 1B: Running Experiments (Tasks 14-18)

## Task 14: Create Experiment Runner Script

**Files:**
- Create: `experiments/run_comparison.py`

This script automates running the full comparison pipeline on a model. It will be run locally on small models and on Colab for 7B+.

**Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Run full abliteration vs steering comparison experiment for a model.

Usage:
    python experiments/run_comparison.py --model <model_id> --output-dir results/

Produces:
    results/<model_name>/benchmark_base.json
    results/<model_name>/benchmark_abliterated.json
    results/<model_name>/benchmark_steered_x{mult}.json  (multiple)
    results/<model_name>/pareto_data.json
    results/<model_name>/pareto_plot.png
    results/<model_name>/sae_analysis.json  (if SAE available)
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--gsm8k-limit", type=int, default=50)
    parser.add_argument("--perplexity-limit", type=int, default=50)
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials for abliteration")
    parser.add_argument("--steering-multipliers", default="0.0,0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--skip-abliteration", action="store_true")
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_slug = args.model.replace("/", "--")
    out = Path(args.output_dir) / model_slug
    out.mkdir(parents=True, exist_ok=True)

    from shade.config import Settings
    from shade.model import Model
    from shade.evaluator import Evaluator
    from shade.benchmark import run_benchmark, format_benchmark_json
    from shade.steering import train_steering_vector_from_prompts, save_steering_vector
    from shade.pareto import run_steering_sweep, plot_pareto_frontier
    from shade.utils import load_prompts

    # --- Step 1: Load model + benchmark base ---
    print(f"\n{'='*60}")
    print(f"MODEL: {args.model}")
    print(f"{'='*60}")

    settings = Settings(model=args.model, _cli_parse_args=False)
    model = Model(settings)
    evaluator = Evaluator(settings, model)

    print("\n[1/4] Benchmarking base model...")
    base_result = run_benchmark(
        model=model, evaluator=evaluator, model_name=f"{args.model} (base)",
        run_gsm8k=True, gsm8k_limit=args.gsm8k_limit,
        run_perplexity=True, perplexity_limit=args.perplexity_limit,
    )
    (out / "benchmark_base.json").write_text(format_benchmark_json(base_result))
    print(f"  Refusals: {base_result.refusals}/{base_result.total_prompts}")
    print(f"  KL: {base_result.kl_divergence:.4f}")
    print(f"  PPL: {base_result.perplexity:.2f}")
    print(f"  GSM8K: {base_result.gsm8k_accuracy:.1%}")

    # --- Step 2: Abliterate + benchmark ---
    if not args.skip_abliteration:
        print("\n[2/4] Running abliteration optimization...")
        settings.n_trials = args.n_trials
        settings.n_startup_trials = min(10, args.n_trials // 3)
        from shade.main import run as shade_run
        shade_run(settings)

        evaluator_abl = Evaluator(settings, model)
        abl_result = run_benchmark(
            model=model, evaluator=evaluator_abl,
            model_name=f"{args.model} (abliterated)",
            run_gsm8k=True, gsm8k_limit=args.gsm8k_limit,
            run_perplexity=True, perplexity_limit=args.perplexity_limit,
        )
        (out / "benchmark_abliterated.json").write_text(format_benchmark_json(abl_result))
        print(f"  Refusals: {abl_result.refusals}/{abl_result.total_prompts}")
        print(f"  GSM8K: {abl_result.gsm8k_accuracy:.1%}")

    # --- Step 3: Train steering vector + sweep ---
    if not args.skip_steering:
        print("\n[3/4] Training steering vector + multiplier sweep...")
        model.reset_model()
        harmless = load_prompts(settings.good_evaluation_prompts)
        harmful = load_prompts(settings.bad_evaluation_prompts)

        base_model = model.model
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model

        sv = train_steering_vector_from_prompts(
            model=base_model, tokenizer=model.tokenizer,
            harmless_prompts=harmless, harmful_prompts=harmful,
        )
        save_steering_vector(sv, out / "steering_vector.pt")

        multipliers = [float(x) for x in args.steering_multipliers.split(",")]
        evaluator_steer = Evaluator(settings, model)
        sweep_data = run_steering_sweep(model, evaluator_steer, sv, multipliers)

        # Benchmark at best multiplier
        best_idx = min(range(len(sweep_data["refusals"])),
                       key=lambda i: sweep_data["refusals"][i])
        best_mult = multipliers[best_idx]
        with sv.apply(base_model, multiplier=best_mult):
            steer_result = run_benchmark(
                model=model, evaluator=evaluator_steer,
                model_name=f"{args.model} (steered x{best_mult})",
                run_gsm8k=True, gsm8k_limit=args.gsm8k_limit,
                run_perplexity=True, perplexity_limit=args.perplexity_limit,
            )
        (out / f"benchmark_steered_x{best_mult}.json").write_text(
            format_benchmark_json(steer_result)
        )

        # Save sweep data + plot
        (out / "steering_sweep.json").write_text(json.dumps(sweep_data, indent=2))

    # --- Step 4: SAE analysis (if available) ---
    if not args.skip_sae:
        print("\n[4/4] SAE decomposition...")
        try:
            from shade.sae_analysis import run_sae_analysis
            # Determine best layer (middle of network)
            n_layers = len(model.get_layers())
            target_layer = n_layers // 2

            sae_results = run_sae_analysis(
                model=base_model, tokenizer=model.tokenizer,
                harmful_prompts=[p.user for p in harmful[:50]],
                harmless_prompts=[p.user for p in harmless[:50]],
                model_name=args.model, layer_index=target_layer,
            )
            (out / "sae_analysis.json").write_text(json.dumps(sae_results, indent=2, default=str))
            print(f"  Top features: {[f['index'] for f in sae_results['top_features'][:5]]}")
        except (ValueError, Exception) as e:
            print(f"  SAE analysis skipped: {e}")

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
mkdir -p experiments
git add experiments/run_comparison.py
git commit -m "feat: add experiment runner script"
```

---

## Task 15: Run Experiments on Model 1 (Local — TinyLlama or Qwen2.5-1.5B)

**Step 1:** Run locally for validation:
```bash
uv run python3 experiments/run_comparison.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gsm8k-limit 20 --perplexity-limit 20 \
    --n-trials 10 --skip-sae
```

**Step 2:** Verify all JSON files are produced in `results/Qwen--Qwen2.5-1.5B-Instruct/`

**Step 3:** Fix any issues found during the real run.

**Step 4:** Commit fixes.

---

## Task 16: Run Experiments on Models 2-4 (Colab)

Run on Colab Pro (T4 16GB) for each model:

1. **Llama-3.2-3B-Instruct**
2. **microsoft/Phi-3-mini-4k-instruct** (3.8B)
3. **google/gemma-2-2b-it** (for SAE analysis — has Gemma Scope SAEs)

For each:
```bash
python experiments/run_comparison.py \
    --model <model_id> \
    --gsm8k-limit 50 --perplexity-limit 100 \
    --n-trials 30 --seed 42
```

Run 3 seeds (42, 123, 456) for steering to get error bars.

Save all results locally. Commit results JSONs to `results/` directory.

---

## Task 17: Generate Figures

**Step 1:** Create `experiments/generate_figures.py`:

```python
"""Generate all paper figures from experiment results."""

import json
from pathlib import Path
from shade.pareto import plot_pareto_frontier
# ... load all result JSONs, generate:
# - Figure 1: Pareto frontier per model (2x2 grid)
# - Figure 2: Comparison bar chart (refusals, KL, PPL, GSM8K)
# - Figure 3: SAE feature overlap analysis
# - Table 1: Full metrics table (LaTeX format)
```

**Step 2:** Run to produce `figures/` directory with all PNGs.

**Step 3:** Commit.

---

## Task 18: Generate LaTeX Table

**Step 1:** Create `experiments/generate_tables.py`:

```python
"""Generate LaTeX tables from experiment results."""

def generate_main_results_table(results_dir: str) -> str:
    """Generate Table 1: Head-to-head comparison.

    Columns: Model | Technique | Refusals | KL Div | PPL | GSM8K
    """
    # Load all benchmark JSONs
    # Format as LaTeX tabular
    # Include mean +/- std across seeds for steering
    pass
```

---

# Phase 1C: Writing the Paper (Tasks 19-22)

## Task 19: Set Up LaTeX Paper

**Step 1:** Create `paper/` directory with arxiv-style template:

```bash
mkdir -p paper
# Download arxiv-style template
cd paper && git clone https://github.com/kourgeorge/arxiv-style.git template
```

**Step 2:** Create `paper/main.tex` with the following structure:

```latex
\documentclass{article}
\usepackage{arxiv}
\usepackage{booktabs, graphicx, hyperref, amsmath}

\title{Decomposing Refusal: An Empirical Comparison of Abliteration\\
and Activation Steering for Capability-Preserving Refusal Removal}

\author{Author Name}

\begin{document}
\maketitle

\begin{abstract}
% 150-200 words
\end{abstract}

\section{Introduction}
\section{Background \& Related Work}
\section{Method}
\section{Experimental Setup}
\section{Results}
\section{Discussion}
\section{Conclusion}

\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
```

**Step 3:** Create `paper/references.bib` with all citations from the research doc.

---

## Task 20: Write Paper Sections

Write each section based on experiment results. Key sections:

- **Abstract:** Novel comparison, N models, 4 metrics, key finding
- **Introduction:** Gap — no tool compares abliteration vs steering with capability metrics
- **Method:** Shade architecture, how metrics are computed
- **Results:** Table 1 (comparison), Figure 1 (Pareto), Figure 2 (SAE)
- **Discussion:** When to use which technique, limitations

---

## Task 21: Polish and Submit to arXiv

**Step 1:** Proofread all sections
**Step 2:** Verify all figures render correctly
**Step 3:** Upload to arXiv (cs.CL + cs.AI categories)
**Step 4:** Publish models to HuggingFace with comparative model cards

---

## Task 22: Publish HuggingFace Model Cards

For each model tested, push:
- Abliterated variant (merged LoRA)
- Steering vector (.pt file)
- Model card with comparison table and Pareto plot
- Reproducibility command: `shade compare --model X`

---

# Summary

| Phase | Tasks | What |
|-------|-------|------|
| **1A: Engineering** | 1-13 | Build benchmark + steering + compare + pareto + SAE modules |
| **1B: Experiments** | 14-18 | Run on 3-5 models, generate figures and tables |
| **1C: Paper** | 19-22 | Write arXiv paper, publish models to HuggingFace |

**Total: 22 tasks.** Phase 1A is the foundation. Phase 1B produces the novel findings. Phase 1C turns findings into a citable paper — which is what actually gets you noticed.
