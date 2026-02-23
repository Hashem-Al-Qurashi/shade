# Testing Strategy: Proving Phase 1A Actually Works

## Problem

182 tests pass but most are mocks. The code has never been run against a real model. If someone clones the repo and runs `shade benchmark gpt2`, we have zero evidence it won't crash.

## Architecture: Three Layers

### Layer 1: Pure Logic Tests (`tests/test_logic.py`)
Test the math and edge cases with real tensors, no models.
- Perplexity: craft logits where expected perplexity = exactly 10.0
- Pareto: empty, single, all-dominated, ties, diagonal tradeoff
- Comparison formatting: None fields, single result
- SAE decomposition: deterministic tensor, verify sorted by activation

### Layer 2: Integration Tests (`tests/test_integration.py`)
Test with real GPT-2 (124M params, CPU, ~5-8s total).
- `@pytest.fixture(scope="module")` loads GPT-2 once
- Real perplexity computation → verify in [30, 500] range
- Steering vector training → save → load roundtrip
- Steering changes model output (base vs steered differ)
- Full BenchmarkResult assembly with real values
- Pareto frontier on realistic sweep data

### Layer 3: CLI Smoke Tests (`tests/test_cli.py`)
Click's CliRunner, no model loading.
- `--help` for benchmark, steer, compare → verify options present
- Missing model_id → exit code 1

## Test Markers

```ini
markers = ["integration: tests requiring real model downloads (GPT-2)"]
```

Default: `pytest tests/` runs everything.
Integration only: `pytest -m integration`

## Known Tradeoffs

1. Don't test through Shade's Model/Evaluator wrappers (depend on HF dataset downloads)
2. No real SAE testing (multi-GB weights) — test tensor math only
3. CPU only — GPU testing in Phase 1B on Colab

## Verification Protocol

Every test is run and output verified before commit. No "it should work."
