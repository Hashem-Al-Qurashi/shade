# Shade Dominance Strategy: From Abliteration Tool to Model Modification Workbench

**Date:** 2026-02-21
**Author:** Brainstorming session — strategic design document
**Status:** Draft — awaiting approval

---

## Executive Summary

Shade cannot win by being "a better Heretic." Heretic has 8.8k stars, 1000+ published models, and community momentum. Competing on the same axis (CLI abliteration tool) is a losing game.

The winning strategy: **evolve Shade from an abliteration tool into the first multi-technique model modification workbench.** Make abliteration one feature among many. When Shade has abliteration + activation steering + benchmarking + recipes + a visual workbench, the "clone" origin becomes irrelevant — Heretic becomes a single-purpose subset of what Shade offers.

Execute in 4 phases. Each phase independently creates value. Even stopping after Phase 1 improves Shade's position.

---

## Current State (Honest Assessment)

### What Shade Has
- Working abliteration engine (Optuna-optimized, LoRA-based)
- Web chat UI (ahead of Heretic, which has no UI)
- Ollama integration (Heretic doesn't have this)
- Side-by-side comparison mode
- Model library management
- Professional error handling
- 95% type hint coverage
- Published on PyPI as `shade-ai`

### What Shade Lacks (Critical Gaps)
| Gap | Severity |
|-----|----------|
| Zero test suite | CRITICAL |
| No published benchmarks | CRITICAL |
| No CI integration tests | CRITICAL |
| `main.py` run() is 916 lines | HIGH |
| 4 FIXMEs in model.py | HIGH |
| No CONTRIBUTING guide | HIGH |
| No Docker support | HIGH |
| No changelog | MEDIUM |
| Web UI is prototype (147 lines vanilla JS) | MEDIUM |
| No plugin/extension system | MEDIUM |
| `benchmark` command runs 5 hardcoded prompts | HIGH |

---

## Phase 0: Fix the Foundation (Weeks 1-2)

### Goal
Make Shade the only **tested** abliteration tool. Create the engineering credibility that everything else builds on.

### 0.1 Test Suite Architecture

Three-layer testing pyramid:

**Layer 1: Pure Math Tests (~50 tests, no GPU, fast)**
- Refusal direction calculation with synthetic tensors
- Weight kernel computation (given layer count, position, weights — verify per-layer lambda values)
- LoRA delta computation (given direction vector and weight matrix — verify delta_W = -lambda * v * v^T @ W)
- Row normalization variants (NONE, PRE, FULL)
- Refusal detection patterns (all 31+ markers, edge cases, false positives)
- KL divergence computation with known distributions
- Batch utilities (batchify, format_duration, get_trial_parameters)
- Config validation (invalid values, type coercion, TOML parsing)

**Layer 2: Integration Tests (~20 tests, tiny model, CPU)**
- Use a tiny transformer (e.g., `sshleifer/tiny-gpt2` or create a 1M param test fixture)
- Full pipeline: load model -> compute residuals -> compute directions -> abliterate -> verify weights changed
- LoRA merge: abliterate -> merge -> verify model generates text
- Generation: verify batch and streaming output
- Server API: test /api/chat, /api/info, /compare endpoints with test client
- Config loading: TOML file -> Settings object -> verify all fields

**Layer 3: Smoke Tests (manual, pre-release)**
- Full optimization on Qwen 2.5 0.5B (smallest supported model)
- Verify Ollama export works end-to-end
- Web UI serves and streams correctly

### 0.2 Refactor main.py

Break the 916-line `run()` into a pipeline package:

```
src/shade/pipeline/
    __init__.py          # Pipeline orchestrator (~30 lines)
    setup.py             # Banner, deps, GPU detection, HF login
    selection.py         # Model selection menu, suggested models
    study.py             # Optuna study management, checkpoints
    preprocessing.py     # Dataset loading, batch size, prefix detection
    directions.py        # Residual extraction, direction computation
    optimization.py      # Optuna objective, trial loop
    export.py            # Model saving, GGUF, EXL2, Ollama
```

The new `run()` becomes:
```python
def run(settings: Settings) -> None:
    env = setup.initialize(settings)
    model_id = selection.select_model(settings, env)
    study = study_manager.manage_study(settings, model_id)
    data = preprocessing.prepare(settings, model_id)
    directions = direction_computer.compute(model, data)
    best_trial = optimizer.run(model, directions, study, settings)
    export.save_and_deploy(model, best_trial, settings)
```

Each module takes explicit dependencies (no global state). This enables unit testing with mocks and future plugin system.

### 0.3 Fix FIXMEs

- `model.py:432` — Add explicit dtype checking before tensor casts, add assertions
- `model.py:451` — Validate assumptions about tensor shapes, add tests for edge cases
- `model.py:567` — Keep type: ignore but add wrapper with proper signature, document why
- `model.py:723` — Same approach as 567

### 0.4 CI Improvements

Add to `.github/workflows/ci.yml`:
- `pytest --cov=shade --cov-report=xml` with 50% coverage threshold
- Integration tests with tiny model (separate job, longer timeout)
- Coverage badge generation
- Keep existing ruff/ty checks

### 0.5 Deliverables
- ~70 tests (50 unit + 20 integration)
- Refactored pipeline package
- Fixed or documented FIXMEs
- CI with test execution and coverage
- Coverage badge in README

---

## Phase 1: Prove Quality with Benchmarks (Weeks 3-4)

### Goal
Publish transparent, reproducible benchmark results. Make Shade's benchmark suite a product feature that even Heretic users want.

### 1.1 Benchmark Suite (Product Feature)

Build a proper benchmark command:

```bash
shade benchmark <model_path_or_id> --suite refusal,kl,gsm8k,mmlu,humaneval
```

**Metrics:**

| Metric | What It Measures | Tool |
|--------|-----------------|------|
| Refusal Rate | % of harmful prompts refused (lower = more effective) | Custom (existing evaluator.py) |
| KL Divergence | Distribution shift from original (lower = less damage) | Custom (existing) |
| GSM8K | Math reasoning preservation | lm-evaluation-harness |
| MMLU | General knowledge preservation | lm-evaluation-harness |
| HumanEval | Code generation preservation | lm-evaluation-harness |

**Implementation:**
- Integrate [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as optional dependency
- Wrap in Shade CLI with unified output format
- Output: terminal table (rich), markdown report, JSON, HuggingFace model card

**The Trojan Horse:** If Shade's benchmark suite is the best way to evaluate ANY abliterated model, even Heretic users install Shade to benchmark. That's an acquisition channel.

### 1.2 Comparative Benchmarks

Run head-to-head against Heretic using identical conditions:
- Same models: Qwen 2.5 1.5B, Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B v0.3
- Same defaults: 200 trials, default datasets
- Same hardware: Document exact GPU, driver version, CUDA version
- Same metrics: refusal rate, KL divergence, GSM8K, MMLU

**Honesty policy:** Publish ALL results, even where Heretic wins. Analyze WHY differences exist. Cherry-picking destroys credibility.

### 1.3 Published Models

Upload 4-6 abliterated models to HuggingFace with:
- Full benchmark results in model card
- "Made with Shade" branding
- Reproduction instructions
- Download counts become social proof

### 1.4 Deliverables
- `shade benchmark` command with 5 metric suites
- Head-to-head results table (4+ models)
- 4-6 models published to HuggingFace
- Blog post: "Shade vs Heretic: Honest Benchmarks"
- Benchmark results in README

---

## Phase 2: First Multi-Technique Tool (Weeks 5-8)

### Goal
Add activation steering. The moment this ships, Shade is no longer "an abliteration tool" — it's "a model modification tool." This is the narrative-changing phase.

### 2.1 What Is Activation Steering?

Unlike abliteration (permanent weight modification), activation steering adds direction vectors during inference to shift behavior in real-time. It's complementary — you can abliterate AND steer.

**Research basis:**
- [Turner et al. 2024 — "Activation Addition"](https://proceedings.iclr.cc/paper_files/paper/2025/file/c4d26a95fd83f8e590f81c54ae670b5d-Paper-Conference.pdf) (ICLR 2025)
- [steering-vectors PyPI library](https://pypi.org/project/steering-vectors/0.8.0/) — existing Python implementation to build on
- [Dialz toolkit](https://www.arxiv.org/pdf/2505.06262v2) — reference implementation from Cardiff University

### 2.2 Implementation Plan

**New module: `src/shade/steering.py` (~400 lines)**

```python
class SteeringVector:
    """Compute and apply behavioral steering vectors."""

    def compute(self, model, positive_prompts, negative_prompts, layer):
        """Compute steering direction from contrastive prompt pairs."""
        # Extract residuals at specified layer
        # Return normalized difference of means

    def apply(self, model, strength=1.0):
        """Register forward hooks to inject steering during inference."""
        # PyTorch forward hooks on specified layers

    def remove(self):
        """Remove steering hooks."""

class SteeringLibrary:
    """Pre-computed steering vectors for common behaviors."""
    # creativity, honesty, verbosity, formality, helpfulness
```

**Technical approach:**
- Use PyTorch forward hooks (same mechanism Shade already uses for residual extraction)
- Steering vector = normalized(mean(positive_residuals) - mean(negative_residuals))
- During inference: residual_out = residual_out + strength * steering_vector
- Leverage existing `model.get_residuals_batched()` infrastructure

**CLI commands:**
```bash
shade steer compute <model> --positive creative_prompts.txt --negative boring_prompts.txt --output creativity.pt
shade steer apply <model> --vector creativity.pt --strength 1.2
shade steer list  # Show pre-computed vectors
```

**Pre-built steering vectors (shipped with Shade):**
- `creativity` — more creative/imaginative responses
- `honesty` — more direct/truthful responses
- `verbosity` — control response length
- `formality` — formal vs casual tone
- `technical` — more technical/detailed explanations

### 2.3 Combination: Abliterate + Steer

The unique value prop no other tool has:
```bash
# First abliterate (remove refusal)
shade Qwen/Qwen2.5-7B-Instruct

# Then steer the abliterated model (add creativity)
shade steer apply models/Qwen2.5-7B-shade --vector creativity --strength 1.2

# Benchmark the combined result
shade benchmark models/Qwen2.5-7B-shade-steered --suite refusal,kl,gsm8k
```

### 2.4 Web UI Enhancement

Add to chat interface:
- Behavior sliders (creativity, honesty, verbosity) that adjust steering in real-time
- Toggle steering on/off to see the difference
- This is HIGHLY demoable and visually compelling

### 2.5 Published Models

Upload 4-6 steered models to HuggingFace:
- "Qwen-7B-Shade-Creative" (abliterated + creativity steering)
- "Llama-3B-Shade-Honest" (abliterated + honesty steering)
- These are UNIQUE — nobody else publishes steered models

### 2.6 Deliverables
- `shade steer` command suite
- 5 pre-built steering vectors
- Steering sliders in web UI
- 4-6 combined (abliterated + steered) models on HuggingFace
- Blog post: "Beyond Abliteration: Activation Steering in Shade"
- Tests for steering module (unit + integration)

---

## Phase 3: The Workbench (Weeks 9-12)

### Goal
Build the visual workbench that makes Shade the most accessible AND most powerful model modification tool.

### 3.1 Web UI Rebuild

**Framework decision: Gradio (not SvelteKit)**

Rationale:
- Python-native (no JS build chain for solo Python dev)
- Standard in HuggingFace ecosystem
- Can be hosted on HuggingFace Spaces for free live demo
- Handles streaming, file upload, sliders natively
- Fast to build (~1 week for full UI)

**Tabs:**
1. **Abliterate** — Model selector, parameter config, progress tracking, results
2. **Steer** — Behavior sliders, real-time preview, vector management
3. **Chat** — Test the modified model, A/B comparison with original
4. **Benchmark** — Run evaluations, view results, compare models
5. **Library** — Saved models with version history, export options
6. **Recipes** — Pipeline builder, community recipes, one-click apply

### 3.2 Recipe/Pipeline System

YAML-based modification pipelines:

```yaml
name: "creative-uncensored-writer"
description: "Abliterated model with enhanced creativity"
base_model: "Qwen/Qwen2.5-7B-Instruct"
steps:
  - technique: abliteration
    params:
      n_trials: 200
      kl_divergence_target: 0.01
  - technique: activation_steering
    vector: creativity
    strength: 1.2
  - technique: activation_steering
    vector: verbosity
    strength: -0.5
benchmark:
  suites: [refusal, kl, gsm8k]
export:
  formats: [safetensors, gguf-q4_k_m]
  ollama: true
  huggingface: true
```

**CLI:**
```bash
shade recipe run creative-writer.yaml
shade recipe list          # Show built-in recipes
shade recipe validate my-recipe.yaml
```

**Pre-built recipes (shipped with Shade):**
- `uncensored-base` — Standard abliteration
- `creative-writer` — Abliterate + creativity steering
- `honest-assistant` — Abliterate + honesty steering
- `code-helper` — Abliterate + technical steering
- `roleplay-model` — Abliterate + creativity + verbosity steering

### 3.3 Model Versioning

Track every modification applied to a model:

```
qwen-7b-v1: base model (downloaded 2026-02-21)
qwen-7b-v2: + abliteration (KL: 0.14, refusal: 2/100)
qwen-7b-v3: + creativity steering (strength: 1.2)
qwen-7b-v4: + verbosity steering (strength: -0.5)
```

Stored as JSON metadata alongside the model files. Enables:
- Undo/rollback to any version
- Compare versions side-by-side
- Reproduce any modification from scratch

### 3.4 HuggingFace Spaces Demo

Host a live demo on HuggingFace Spaces:
- Users try Shade without installing anything
- Live steering sliders on a small model
- Benchmark comparison dashboard
- This becomes the primary marketing asset

### 3.5 Deliverables
- Gradio-based workbench (6 tabs)
- Recipe YAML system with 5 pre-built recipes
- Model versioning with metadata tracking
- HuggingFace Spaces live demo
- 10+ pre-modified models published to HuggingFace
- Blog post: "Introducing the Shade Workbench"
- Docker support (Dockerfile + docker-compose)
- CONTRIBUTING.md with dev setup guide

---

## Parallel Track: Community & Marketing (Continuous)

Runs alongside all phases. Not a separate effort — integrated into every phase.

### Model Zoo Strategy
Every phase publishes models to HuggingFace. Models are the #1 discovery mechanism for ML tools.
- Phase 0: 3-5 abliterated models (establish presence)
- Phase 1: 4-6 benchmarked models (with full report cards)
- Phase 2: 4-6 steered models (unique, nobody else has these)
- Phase 3: Recipe-made combination models

### Content Strategy
- Phase 0 blog: "Building a Tested Abliteration Tool" (engineering credibility)
- Phase 1 blog: "Shade vs Heretic: Honest Benchmarks" (attention + credibility)
- Phase 2 blog: "Beyond Abliteration: Activation Steering" (thought leadership)
- Phase 3 blog: "The Model Modification Workbench" (product launch)

### Community Infrastructure
- Discord server (launch at Phase 1)
- GitHub issue templates
- CONTRIBUTING.md (Phase 3)
- Community recipe sharing (Phase 3)

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 0 takes too long | HIGH | Delays everything | Strict 2-week timebox. Ship what's ready. |
| Benchmarks show Shade is worse | MEDIUM | Credibility hit | Don't publish until you understand why. Fix first. |
| Activation steering quality is low | LOW | Phase 2 underwhelms | Fallback to representation engineering. |
| Heretic ships multi-technique first | LOW | Loses first-mover | Speed. Ship Phase 2 fast. |
| Scope creep in Phase 3 | HIGH | Never ships | MVP workbench first. Iterate after launch. |

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 0 | Test coverage | 50%+ |
| 0 | CI passing with tests | Yes |
| 1 | Published benchmark models | 4+ on HuggingFace |
| 1 | Benchmark blog views | 1000+ |
| 2 | Activation steering working | Yes, 5 pre-built vectors |
| 2 | Steered models published | 4+ on HuggingFace |
| 3 | HuggingFace Space demo live | Yes |
| 3 | GitHub stars | 500+ (from ~current) |
| 3 | Community recipes submitted | 5+ |

---

## Narrative Arc

```
Today:        "Shade is an abliteration tool" (same category as Heretic)
After Ph 0:   "Shade is the only TESTED abliteration tool"
After Ph 1:   "Here are Shade's actual numbers vs Heretic"
After Ph 2:   "Shade goes beyond abliteration" (NEW category)
After Ph 3:   "Shade is the workbench for model modification" (OWNS the category)
```

Each phase shifts perception. By Phase 2, the origin story is irrelevant.
