# Experiment Runner Design

## Problem

Phase 1A engineering is complete — abliteration, steering, benchmarking, Pareto analysis all work and are tested. But there are zero experiment results. The tool exists with no data proving it works on real safety-trained models.

## Goal

Build a single experiment runner script that executes the full comparison pipeline (baseline → abliterate → steer → benchmark → Pareto) on any model, saving reproducible results.

## Architecture

```
experiments/run_comparison.py
    ├── Step 1: Load model via shade.model.Model
    ├── Step 2: Baseline benchmark (refusal rate, KL, GSM8K, perplexity)
    ├── Step 3: Abliterate (Optuna search for best parameters)
    ├── Step 4: Benchmark abliterated model
    ├── Step 5: Reset model, train steering vector
    ├── Step 6: Sweep steering multipliers, benchmark each
    ├── Step 7: Save all BenchmarkResults as JSON
    ├── Step 8: Generate Pareto frontier plot
    └── Step 9: Print comparison table
```

## Config per Model

YAML configs in `experiments/configs/` with model-specific settings:

```yaml
model: gpt2
quantization: null
abliteration:
  n_trials: 10
  harmful_prompts: 20
steering:
  layers: [5, 6, 7]
  multipliers: [0.5, 1.0, 1.5, 2.0, 3.0]
benchmark:
  gsm8k_limit: 20
  perplexity_samples: 10
seed: 42
```

## Results Structure

```
results/experiments/
├── gpt2/
│   ├── baseline.json
│   ├── abliterated.json
│   ├── steered_1.0.json
│   ├── steered_2.0.json
│   ├── comparison.json
│   └── pareto.png
├── tinyllama-1.1b/
└── ...
```

## Model Progression

| Model | Hardware | Purpose |
|-------|----------|---------|
| GPT-2 (124M) | Local CPU | Validate pipeline runs end-to-end |
| TinyLlama-1.1B | Local GPU (4-bit) | First real abliteration/steering comparison |
| Llama-3.2-3B | Colab T4 | Real safety-trained model with refusal behavior |
| Phi-3-mini-4k | Colab T4 | Different architecture, good reasoning |
| Gemma-2-2B | Colab T4 | Google's model, SAE support available |

## Key Decisions

- **No new ML code** — calls existing modules (Model, run_benchmark, train_steering_vector_from_prompts, compute_pareto_frontier_2d)
- **YAML configs** — each model gets hardware-appropriate settings
- **Deterministic** — seed torch, random, numpy for reproducibility
- **Results as JSON** — every BenchmarkResult saved via results.save_result()
- **Idempotent** — skip steps if results already exist (resume after crash)
- **Logging** — structured logging (model name, step, elapsed time, key metrics)

## Dependencies

No new dependencies. Uses:
- `shade.model.Model` — model loading
- `shade.benchmark.run_benchmark` — metrics computation
- `shade.steering.train_steering_vector_from_prompts` — steering vector training
- `shade.pareto.plot_pareto_frontier` — visualization
- `shade.results.save_result` / `load_result` — persistence
- `shade.compare.format_comparison_table` — comparison output
- `pyyaml` — config loading (already available via transformers dependency)

## Constraints

- GPT-2 is not safety-trained, so abliteration effects will be minimal — this is expected and serves only as a pipeline validation
- Real refusal reduction data comes from Llama-3.2-3B+ on Colab
- GSM8K evaluation is slow (1-2 min per model on T4) — configs control sample size
- Steering vector training requires contrastive pairs — use Shade's built-in harmful/harmless prompt sets
