# Phase 1 Design: Multi-Technique Pioneer

**Date:** 2026-02-23
**Approach:** B — Multi-Technique Pioneer
**Research:** See `2026-02-23-phase-1-research-findings.md` for full competitive analysis

---

## 1. Problem Statement

Shade has a working abliteration engine, but:
- Its `shade benchmark` command is fake — 5 hardcoded prompts, no metrics, just prints responses
- It only does weight modification (abliteration). No runtime steering.
- It has no way to measure whether abliteration preserved model capabilities

The entire abliteration ecosystem is bifurcated:
- **Weight modification tools:** Heretic, ErisForge, DECCP, FailSpy, Shade
- **Runtime steering tools:** repeng, steering-vectors, wisent, AxBench

No tool bridges both. No tool benchmarks capability preservation automatically.

---

## 2. What We're Building

Three new CLI commands:

### 2.1 `shade benchmark` — Real Metrics

**Replaces** the current fake benchmark (cli.py:648-678).

**Metrics:**
| Metric | Source | New work needed |
|--------|--------|-----------------|
| Refusal rate | evaluator.py already computes this | Wire to CLI, expose results |
| KL divergence | evaluator.py already computes this | Wire to CLI, expose results |
| GSM8K accuracy | New — load 50-100 problems from `datasets` | Parse model answers, compute accuracy |

**Interface:**
```
shade benchmark --model <path-or-hf-id> [--gsm8k] [--gsm8k-limit 50] [--output json]
```

**Output:** Rich table showing all metrics. Optional JSON export for reproducibility.

**Why refusal + KL isn't enough:** The arXiv paper (2512.13655) proved GSM8K is the most vulnerable capability — drops from +1.51pp to -18.81pp depending on tool. KL divergence alone does NOT predict capability loss. This is the metric gap.

### 2.2 `shade steer` — Activation Steering

**New capability.** Uses `steering-vectors` library (v0.12.2, PyPI).

**Three subcommands:**

```
shade steer train --model <model> --dataset <harmful/harmless pairs>
```
- Extract activations at specified layer(s)
- Compute steering vector via mean-difference of contrastive pairs
- Save vector to disk (.pt file)

```
shade steer apply --model <model> --vector <path> --multiplier <float>
```
- Load model + steering vector
- Apply vector via forward hooks during inference
- Open chat interface (reuse existing webchat)

```
shade steer evaluate --model <model> --vector <path>
```
- Run benchmark suite with steering active
- Report refusal rate + KL + GSM8K with steering applied

**Library:** `steering-vectors` v0.12.2
- Supports: LLaMA, Gemma, Mistral, Qwen, GPT, Pythia (any HF decoder-only)
- Based on: Rimsky et al. 2023, Zou et al. 2023
- API: Train from contrastive pairs, apply via forward hooks
- Fallback: `repeng` (689 stars) if integration issues arise

### 2.3 `shade compare` — Side-by-Side Analysis

**Orchestrates both techniques on the same model.**

```
shade compare --model <model> --dataset <prompts> [--gsm8k] [--output json]
```

**Pipeline:**
1. Benchmark base model (refusal rate, KL baseline, GSM8K)
2. Run abliteration → benchmark abliterated model
3. Train steering vector → benchmark steered model
4. Output comparison table

**Output example:**
```
┌─────────────┬──────────┬───────────┬────────────┐
│ Technique   │ Refusals │ KL Div    │ GSM8K Acc  │
├─────────────┼──────────┼───────────┼────────────┤
│ Base        │ 87/100   │ -         │ 45.2%      │
│ Abliterated │ 12/100   │ 0.043     │ 43.8%      │
│ Steered     │ 23/100   │ 0.018     │ 44.9%      │
└─────────────┴──────────┴───────────┴────────────┘
```

**This table doesn't exist anywhere** — no tool, paper, or blog has published an automated abliteration-vs-steering comparison with capability metrics.

---

## 3. Architecture

### 3.1 New Dependencies

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `steering-vectors` | >=0.12.2 | Steering vector training + application | Lightweight |
| `datasets` | (already a dependency) | Load GSM8K benchmark | Already installed |

### 3.2 New Modules

```
src/shade/
├── benchmark.py          # Benchmark logic (refusal + KL + GSM8K)
├── steering.py           # Steering vector training + application
├── compare.py            # Orchestrates abliteration vs steering comparison
```

### 3.3 CLI Integration

All commands integrate into existing `cli.py` Click group:
- `shade benchmark` — replaces current fake benchmark
- `shade steer train|apply|evaluate` — new subcommand group
- `shade compare` — new top-level command

### 3.4 Reuse of Existing Code

| Existing code | Reused by |
|--------------|-----------|
| `evaluator.py` refusal detection (27 markers) | `benchmark`, `compare` |
| `evaluator.py` KL divergence computation | `benchmark`, `compare` |
| `model.py` abliteration engine | `compare` |
| `server.py` webchat | `steer apply` (chat with steered model) |
| `config.py` Settings | All new commands |

---

## 4. Hardware Strategy

### Local Development (GTX 1660 Ti, 6 GB)
- All unit tests (mocked, no GPU)
- Integration tests on TinyLlama-1.1B (2-3 GB at 4-bit)
- Quick iteration on steering vector training (3B models)

### Colab Pro (T4 16GB / A100 40GB)
- Full benchmark runs on 7B models
- Published results and model cards
- GSM8K evaluation at full scale

### Development Workflow
1. Write code + unit tests locally
2. Test on TinyLlama locally for fast iteration
3. Run on 7B+ via Colab for real results
4. Publish findings from Colab runs

---

## 5. Testing Strategy

| Layer | What | How |
|-------|------|-----|
| Unit | Benchmark metric computation, GSM8K answer parsing, steering vector math | pytest, mocked models (same pattern as existing 153 tests) |
| Integration | Full benchmark run, steering train + evaluate | TinyLlama-1.1B locally, real model |
| End-to-end | `shade compare` full pipeline | Colab Pro, 7B models |

---

## 6. Publishable Artifacts

1. **Blog post:** "Abliteration vs Steering: Which Preserves More Capability?"
   - Novel comparison nobody has published
   - Include the comparison table, methodology, reproducibility instructions

2. **HuggingFace model cards (3-4 models):**
   - Each card includes: abliterated variant, steering vector, comparison metrics
   - Reproducibility: `shade compare --model X` reproduces results

3. **Code itself:**
   - First open-source tool combining both techniques
   - Proper benchmark with capability metrics

---

## 7. Implementation Order

1. **`shade benchmark`** first — mostly wiring existing evaluator internals to CLI + adding GSM8K
2. **`shade steer`** second — new functionality, depends on steering-vectors library
3. **`shade compare`** third — orchestrates the first two
4. **Publishing** — blog post + model cards after validation on 7B models

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `steering-vectors` doesn't integrate cleanly with Shade's model loading | Medium | Fallback to `repeng` (similar API, 689 stars) |
| GSM8K evaluation too slow on 6 GB | Low | Use `--limit 50`, Colab for full runs |
| Steering results are unimpressive compared to abliteration | Medium | This IS the finding — honest reporting is what impresses researchers |
| KL divergence doesn't correlate with GSM8K loss | Medium | This IS the finding — proves why capability benchmarks are needed |

---

## 9. Success Criteria

- [ ] `shade benchmark` outputs refusal rate + KL + GSM8K in a clean table
- [ ] `shade steer train` produces a steering vector from contrastive pairs
- [ ] `shade steer apply` opens chat with steering active
- [ ] `shade steer evaluate` benchmarks a steered model
- [ ] `shade compare` produces the side-by-side table
- [ ] All commands work on TinyLlama locally and 7B on Colab
- [ ] Blog post written with reproducible findings
- [ ] 3+ model cards published on HuggingFace
