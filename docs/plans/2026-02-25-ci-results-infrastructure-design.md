# CI + Results Infrastructure Design

## Problem

CI exists but doesn't run tests. 214 tests pass locally but are never verified in CI. No results persistence, no regression detection, no experiment data structure.

## CI Changes

Fix `.github/workflows/ci.yml`:
- Branch target: `master` → `main`
- Add unit test job: `pytest -m "not integration" --cov`
- Add integration test job: `pytest -m integration` with HF cache
- Coverage gate: 80%
- Upload coverage + results as artifacts

## Results Infrastructure

### ResultsManager (`src/shade/results.py`)
- `save_result(result, path)` — BenchmarkResult to JSON
- `load_result(path)` — JSON to BenchmarkResult
- `compare_baseline(current, baseline, tolerance)` — regression check

### Directory Structure
```
results/
├── baselines/          # Integration test reference values (git-tracked)
├── experiments/        # Phase 1B results (git-tracked)
│   └── <model>/
│       ├── base.json
│       ├── abliterated.json
│       └── steered_x1.0.json
└── figures/            # Paper plots (git-tracked)
```

### Integration Test Saving
Tests save real values to `results/baselines/gpt2_baseline.json`.
Future runs compare against baseline with 10% tolerance.

## Scope
- Fix CI (branch + tests + coverage)
- Create ResultsManager
- Wire integration tests to save baselines
- Create results/ directory structure
