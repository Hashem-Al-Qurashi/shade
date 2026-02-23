# Phase 1 Research Findings: Competitive Landscape & Strategy

**Date:** 2026-02-23
**Purpose:** Complete research context for Phase 1 design decisions

---

## 1. Current State of Shade (Post Phase 0)

### What We Built
- 153 unit tests (config, evaluator, model math, server, utils) — all passing
- CI pipeline with pytest + coverage on every push
- 4 FIXMEs in model.py resolved with runtime type checks
- Forked to https://github.com/Hashem-Al-Qurashi/shade

### What Shade Already Has
- Working abliteration engine (Optuna TPE-optimized, LoRA-based)
- Web chat UI (Heretic has no UI)
- Ollama integration (Heretic doesn't have this)
- Side-by-side model comparison mode
- Model library management
- Professional error handling with rich output
- 95% type hint coverage
- Published on PyPI as `shade-ai`

### What Shade's Benchmark Command Actually Does (cli.py:648-678)
**It's a joke.** Five hardcoded prompts, no metrics, no scoring:
```python
test_prompts = [
    "Explain quantum physics in one sentence.",
    "What is 25 * 4 + 10?",
    "How do I bake a chocolate cake?",
    "Who was Albert Einstein?",
    "What is the capital of France?"
]
for q in test_prompts:
    response = model.stream_chat_response([{"role": "user", "content": q}])
    print(f"A: {response}")
```
No refusal rate. No KL divergence. No capability measurement. Just prints responses.

### What Shade's Evaluator Already Computes (evaluator.py)
Internally during optimization, Shade already computes:
- **Refusal rate**: Count of responses matching 27+ refusal markers (e.g., "i'm sorry", "i cannot", "as an ai")
- **KL divergence**: First-token probability distribution shift from base model
- **Combined score**: `(kl_divergence / kl_scale, refusals / base_refusals)` — Optuna minimizes both

These metrics exist but are NOT exposed through the benchmark command. They only run during the optimization loop.

---

## 2. Competitor Analysis

### Heretic (Dominant Player)
- **GitHub:** https://github.com/p-e-w/heretic — 9,300+ stars
- **Last commit:** February 14, 2026 (v1.2.0, actively maintained)
- **HuggingFace:** heretic-org, 1000+ published models
- **Metrics:** Refusal rate (out of 100 prompts) + KL divergence, co-minimized via Bayesian optimization
- **Research mode:** `--research` flag generates PaCMAP scatter plots of harmful vs harmless residuals per layer, animated GIF showing separation across depth, per-layer geometry table (cosine similarity, L2 norms, silhouette coefficients)
- **Published results:** Per-model HuggingFace cards with both metrics. Notes results are platform-dependent (compiled on RTX 5090)
- **Reproducibility:** `--evaluate-model` flag lets users reproduce evaluation
- **Key weakness:** PaCMAP shows WHERE separation happens but NOT what else the refusal direction overlaps with. No capability benchmarks. Highly variable outcomes (KL from 0.043 to 1.646 across models per arXiv paper)

### ErisForge
- **GitHub:** https://github.com/Tsadoq/ErisForge — 250 stars
- **Focus:** Capability preservation during abliteration
- **Metrics:** Built-in ExpressionRefusalScorer (keyword matching)
- **Performance:** Most consistent capability preservation (avg GSM8K change: -0.28pp per arXiv)
- **Weakness:** Less aggressive refusal removal than Heretic

### DECCP
- **GitHub:** https://github.com/AUGMXNT/deccp
- **Focus:** Simple, single-pass abliteration
- **Performance:** Best capability preservation overall (avg GSM8K change: -0.13pp per arXiv)
- **Weakness:** Less effective at refusal removal, fewer model architectures supported

### FailSpy/abliterator
- **GitHub:** https://github.com/FailSpy/abliterator — 581 stars
- **Focus:** Interactive research workflows via TransformerLens
- **Strength:** Best tool for exploratory analysis (IPython-based, cache activations across resid_pre/post, attn_out, mlp_out)
- **Weakness:** "A glorified IPython notebook" — not a production tool. Fewest architecture successes.

### NousResearch/llm-abliteration
- **GitHub:** https://github.com/NousResearch/llm-abliteration
- **Has:** `analyze.py` with `-c` flag for per-layer refusal direction magnitude charts, `compare.py` for model comparison
- **Weakness:** Lightweight charts only, not a visualization tool

### andyrdt/refusal_direction (Original Research)
- **GitHub:** https://github.com/andyrdt/refusal_direction — 350 stars
- **Paper:** "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024, arXiv:2406.11717)
- **What it is:** Canonical research code. Reproducibility artifact, not a tool.

---

## 3. Benchmark & Evaluation Landscape

### What Heretic Users See
- Refusal count (e.g., "21/100") + KL divergence (e.g., "0.43")
- Base model benchmark table COPIED (not re-run on abliterated version)
- Assumption: low KL ≈ preserved capabilities

### What the arXiv Paper Found (2512.13655, Dec 2025)
Cross-tool comparison of Heretic, DECCP, ErisForge, FailSpy across 16 models (7B-14B):
- **GSM8K is the most vulnerable capability** — math reasoning drops from +1.51pp to -18.81pp (-26.5% relative) depending on tool
- Heretic's Bayesian optimization produces **highly variable** outcomes (KL 0.043 to 1.646)
- Simpler tools (DECCP, ErisForge) are MORE consistent
- No tool was universally best across all models

### UncensorBench (Only Standalone Benchmark)
- **GitHub:** https://github.com/wisent-ai/uncensorbench — 17 stars
- **HuggingFace:** https://huggingface.co/spaces/wisent-ai/UncensorBench (live leaderboard)
- **What it does:** 150 prompts across 15 categories, 4 evaluation methods (LLM-judge F1=0.888, semantic similarity F1=0.640, keyword F1=0.449, log-likelihood F1=0.271)
- **Limitation:** Measures refusal rate ONLY. No capability preservation metrics.

### lm-evaluation-harness (EleutherAI)
- **GitHub:** https://github.com/EleutherAI/lm-evaluation-harness
- **Can run on 6 GB GPU:** Yes, with 4-bit quantization and `--batch_size auto`
- **Practical tip:** `--limit 100` runs partial evaluations (how solo researchers work on consumer hardware)
- **GGUF support:** `--model hf --model_args pretrained=/path/,gguf_file=model.gguf,tokenizer=/path/`
- **Warning:** Omitting separate tokenizer for GGUF causes hours-long hangs
- **Time constraint:** Full MMLU on 7B takes hours on 24GB GPU, days on 6GB

### LightEval (HuggingFace)
- **GitHub:** https://github.com/huggingface/lighteval
- **Alternative to lm-eval-harness:** Cleaner CLI, better HF integration, 1000+ tasks
- **Same VRAM constraint:** Model size determines GPU requirement

---

## 4. Activation Steering Landscape

### What Steering Is
Unlike abliteration (permanent weight modification), activation steering adds direction vectors during inference to shift behavior in real-time. Complementary to abliteration — you can do both.

### steering-vectors Library
- **GitHub:** https://github.com/steering-vectors/steering-vectors
- **PyPI:** steering-vectors v0.12.2 (released Feb 21, 2025)
- **Stars:** 139 | Active maintenance, CI/CD, codecov
- **Supports:** Any HF decoder-only transformer (GPT, LLaMA, Gemma, Mistral, Pythia)
- **Based on:** Rimsky et al. 2023, Zou et al. 2023 (representation engineering)
- **What it does:** Trains steering vectors from contrastive prompt pairs, applies via forward hooks at inference

### repeng
- **GitHub:** https://github.com/vgel/repeng — 689 stars
- **Focus:** Runtime representation engineering
- **Difference from steering-vectors:** Similar approach, more stars, different API

### AxBench (Stanford NLP)
- **GitHub:** https://github.com/stanfordnlp/axbench — 165 stars
- **Paper:** arXiv:2501.17148 (ICML 2025 spotlight)
- **What it does:** Benchmarks 10+ interpretability/steering methods on Gemma-2-2B/9B
- **Key finding:** "Prompting outperforms all existing methods" for steering; "difference-in-means performs best" for detection
- **Scope:** Steering and concept detection only, not abliteration

### Dialz (Cardiff University)
- **Paper:** arXiv:2505.06262 (ACL 2025 demo)
- **Alternative steering toolkit** — newer, less established than steering-vectors

---

## 5. The Three Gaps Nobody Has Filled

### Gap 1: Capability-Overlap Visualization
**What's missing:** No tool shows whether the refusal direction being removed overlaps with math reasoning, code generation, or factual retrieval circuits.

**Heretic's research mode** shows WHERE harmful/harmless prompts separate (PaCMAP plots) but NOT what else those directions touch.

**Relevant research:**
- SAE features for refusal decomposition (arXiv:2411.11296) — shows sparse autoencoder features CAN decompose refusal into interpretable components, but no tool packages this
- Alignment Forum post on SAE features for refusal and sycophancy steering vectors
- "Norm-Preserving Biprojected Abliteration" (HuggingFace blog by grimjim) — addresses norm preservation but not capability overlap

### Gap 2: Unified Multi-Technique Tool
**What's missing:** Zero tools combine permanent weight abliteration with runtime activation steering.

**The ecosystem is completely bifurcated:**
- Weight modification: Heretic, ErisForge, DECCP, FailSpy, Shade
- Runtime steering: repeng, steering-vectors, wisent, AxBench

**No overlap.** The Surgical Refusal Ablation paper (arXiv:2601.08489) compares abliteration vs steering academically but released no tool.

### Gap 3: Capability-Aware Abliteration Benchmark
**What's missing:** No automated pipeline combines refusal rate + KL divergence + capability benchmarks (GSM8K/MMLU) in one tool that works across different abliteration methods.

**UncensorBench** measures refusal only (17 stars). **Heretic** evaluates internally only. The **arXiv paper** (2512.13655) did the analysis manually but released no code.

---

## 6. Hardware Constraints

### Our Setup
- GTX 1660 Ti — 6 GB VRAM
- Intel i7-9750H — 6 cores / 12 threads
- 32 GB RAM
- 1.3 TB free NVMe
- Google Colab Pro available (T4 16GB / A100 40GB)

### What Fits Locally
| Model | Precision | VRAM | Feasible? |
|-------|-----------|------|-----------|
| 1-3B (TinyLlama, Gemma-2B, Phi-2) | 4-bit | 2-3 GB | Yes, comfortable |
| 7B (Llama 3, Mistral, Qwen 2.5) | 4-bit | 5-6 GB | Tight, batch_size=1 |
| 7B | FP16 | ~14 GB | Colab only |
| 13B+ | Any | 10+ GB | Colab only |

### Practical Workflow
- Develop and test locally (code, unit tests, 3-second runs)
- Run Shade on small models locally (1-3B for quick iteration)
- Use Colab Pro for 7B+ models, benchmarks, and published results
- Kaggle free tier: 16GB GPU, 30 hrs/week (backup)

---

## 7. Key Papers

| Paper | Year | Key Finding |
|-------|------|-------------|
| Refusal in LMs Is Mediated by a Single Direction (arXiv:2406.11717) | NeurIPS 2024 | Foundational abliteration paper — single direction removal works |
| Comparative Analysis of LLM Abliteration Methods (arXiv:2512.13655) | Dec 2025 | Heretic variable, simpler tools more consistent, GSM8K most vulnerable |
| Surgical Refusal Ablation (arXiv:2601.08489) | Jan 2026 | Compares abliteration vs steering, introduces "Concept Atoms" to protect |
| SAE Features for Refusal Steering (arXiv:2411.11296) | Nov 2024 | Sparse autoencoders can decompose refusal into interpretable features |
| AxBench (arXiv:2501.17148) | ICML 2025 | Benchmarks steering methods; prompting outperforms all for steering |
| Activation Addition (Turner et al.) | ICLR 2025 | Foundational activation steering paper |

---

## 8. Strategic Conclusion

### What would NOT impress top AI labs
- More GitHub stars than Heretic
- More HuggingFace model downloads
- A prettier CLI or better UX
- Incremental improvements to existing abliteration

### What WOULD impress
- **Research novelty** — first tool combining abliteration + steering with comparative analysis
- **Technical depth** — understanding model internals, not just calling APIs
- **Rigorous methodology** — proper experiments, honest reporting, reproducibility
- **Published artifacts** — blog post with novel findings, code that others cite

### Recommended Phase 1: Multi-Technique Pioneer + Benchmark
1. Add `shade steer` — activation steering via steering-vectors library
2. Fix `shade benchmark` — refusal rate + KL + optional GSM8K subset
3. Add `shade compare` — run BOTH techniques on same model, metrics side by side
4. Publish blog post: "Abliteration vs Steering: Which Preserves More Capability?"
5. Publish 3-4 models to HuggingFace with comparative analysis in model cards

This fills Gap 2 (multi-technique) and Gap 3 (capability-aware benchmark) simultaneously. Nobody else is doing this.
