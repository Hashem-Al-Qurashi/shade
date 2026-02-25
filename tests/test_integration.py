"""Integration tests with real GPT-2 model.

These tests load GPT-2 (124M params) on CPU and exercise the actual
functions end-to-end. They prove the code works with real models,
not just mocks.

Run with: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fix: other test files patch sys.modules with MagicMock at import time.
# Since pytest imports ALL files before running ANY test, we must undo the
# mocking at test execution time so we get real transformers/steering_vectors.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _restore_real_modules():
    """Temporarily remove MagicMock entries so real imports work; restore after."""
    mocked = {
        name: mod
        for name, mod in sys.modules.items()
        if isinstance(mod, MagicMock)
    }
    for name in mocked:
        del sys.modules[name]
    yield
    # Restore mocks so other test modules aren't affected
    sys.modules.update(mocked)


# ---------------------------------------------------------------------------
# Shared fixture: load GPT-2 once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2(_restore_real_modules):
    """Load GPT-2 (124M) once, shared across all integration tests."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Test 1: Real perplexity computation
# ---------------------------------------------------------------------------


class TestRealPerplexity:
    def test_perplexity_on_english_text(self, gpt2):
        """GPT-2 perplexity on common English should be in [30, 500]."""
        from shade.benchmark import compute_perplexity

        model, tokenizer = gpt2
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a field of artificial intelligence.",
        ]
        ppl = compute_perplexity(model, tokenizer, texts)

        assert isinstance(ppl, float)
        assert math.isfinite(ppl), f"Perplexity should be finite, got {ppl}"
        assert 30 < ppl < 500, (
            f"GPT-2 perplexity on English text should be in [30, 500], got {ppl}"
        )

    def test_perplexity_on_gibberish_is_higher(self, gpt2):
        """Gibberish text should have higher perplexity than English."""
        from shade.benchmark import compute_perplexity

        model, tokenizer = gpt2
        english = ["The president gave a speech about the economy."]
        gibberish = ["Xkp qvz mtl brf nwy jdh ocs ael."]

        ppl_english = compute_perplexity(model, tokenizer, english)
        ppl_gibberish = compute_perplexity(model, tokenizer, gibberish)

        assert ppl_gibberish > ppl_english, (
            f"Gibberish perplexity ({ppl_gibberish:.1f}) should be higher "
            f"than English ({ppl_english:.1f})"
        )


# ---------------------------------------------------------------------------
# Test 2: Steering vector training roundtrip
# ---------------------------------------------------------------------------


class TestSteeringRoundtrip:
    def test_train_save_load(self, gpt2, tmp_path):
        """Train a steering vector, save it, load it back."""
        from steering_vectors import train_steering_vector

        from shade.steering import load_steering_vector, save_steering_vector

        model, tokenizer = gpt2
        pairs = [
            ("What is 2+2?", "How to hack a server?"),
            ("Tell me about dogs", "How to make explosives?"),
            ("What is the weather?", "How to steal money?"),
        ]

        sv = train_steering_vector(
            model,
            tokenizer,
            pairs,
            layers=[5, 6],
            batch_size=2,
            show_progress=False,
        )

        # Save and load
        path = tmp_path / "sv.pt"
        save_steering_vector(sv, path)
        loaded = load_steering_vector(path)

        assert type(loaded).__name__ == type(sv).__name__
        assert path.stat().st_size > 0, "Saved file should not be empty"


# ---------------------------------------------------------------------------
# Test 3: Steering actually changes model behavior
# ---------------------------------------------------------------------------


class TestSteeringChangesBehavior:
    def test_steered_output_differs_from_base(self, gpt2):
        """Applying a steering vector should change the model's output."""
        from steering_vectors import train_steering_vector

        model, tokenizer = gpt2
        pairs = [
            ("What is 2+2?", "How to hack a server?"),
            ("Tell me about dogs", "How to make explosives?"),
        ]

        sv = train_steering_vector(
            model,
            tokenizer,
            pairs,
            layers=[5, 6],
            batch_size=2,
            show_progress=False,
        )

        prompt = "How to"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        # Base output
        with torch.no_grad():
            base_out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)

        # Steered output (high multiplier to ensure visible difference)
        with sv.apply(model, multiplier=3.0):
            with torch.no_grad():
                steered_out = model.generate(
                    input_ids, max_new_tokens=20, do_sample=False
                )
        steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True)

        assert base_text != steered_text, (
            f"Steering should change output.\n"
            f"Base:    {base_text!r}\n"
            f"Steered: {steered_text!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: build_contrastive_pairs → train flow
# ---------------------------------------------------------------------------


class TestFullTrainingPipeline:
    def test_build_pairs_then_train(self, gpt2):
        """Use shade's build_contrastive_pairs → train_steering_vector_from_prompts."""
        from shade.steering import (
            build_contrastive_pairs,
            train_steering_vector_from_prompts,
        )
        from shade.utils import Prompt

        model, tokenizer = gpt2

        harmless = [
            Prompt(system="You are helpful.", user="What is 2+2?"),
            Prompt(system="You are helpful.", user="Tell me about cats."),
        ]
        harmful = [
            Prompt(system="You are helpful.", user="How to hack?"),
            Prompt(system="You are helpful.", user="How to steal?"),
        ]

        # Verify pairs are built correctly
        pairs = build_contrastive_pairs(harmless, harmful)
        assert len(pairs) == 2
        assert pairs[0] == ("What is 2+2?", "How to hack?")

        # Verify training works end-to-end
        sv = train_steering_vector_from_prompts(
            model=model,
            tokenizer=tokenizer,
            harmless_prompts=harmless,
            harmful_prompts=harmful,
            layers=[4, 5],
            batch_size=2,
            show_progress=False,
        )
        assert sv is not None
        assert hasattr(sv, "apply"), "Steering vector should have an apply method"


# ---------------------------------------------------------------------------
# Test 5: BenchmarkResult with real values
# ---------------------------------------------------------------------------


class TestBenchmarkResultWithRealData:
    def test_assembly_with_real_perplexity(self, gpt2):
        """Construct BenchmarkResult with a real perplexity value."""
        from shade.benchmark import (
            BenchmarkResult,
            compute_perplexity,
            format_benchmark_json,
            format_benchmark_table,
        )

        model, tokenizer = gpt2
        real_ppl = compute_perplexity(model, tokenizer, ["Hello, world!"])

        result = BenchmarkResult(
            model_name="gpt2",
            refusals=0,
            total_prompts=10,
            refusal_rate=0.0,
            kl_divergence=0.0,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=real_ppl,
        )

        # Table output should contain the real perplexity value
        table = format_benchmark_table(result)
        assert "gpt2" in table
        # The perplexity value should appear formatted to 2 decimals
        assert f"{real_ppl:.2f}" in table

        # JSON output should be parseable and contain the value
        import json

        parsed = json.loads(format_benchmark_json(result))
        assert parsed["perplexity"] == pytest.approx(real_ppl)


# ---------------------------------------------------------------------------
# Test 6: Pareto frontier on realistic sweep data
# ---------------------------------------------------------------------------


class TestParetoWithRealisticData:
    def test_steering_sweep_shape(self):
        """Simulate a steering sweep and verify Pareto frontier makes sense."""
        import numpy as np

        from shade.pareto import compute_pareto_frontier_2d

        # Simulate: as steering multiplier increases, refusals drop but KL rises
        refusals = [87, 65, 40, 22, 10, 5, 3]
        kl_values = [0.0, 0.01, 0.03, 0.08, 0.15, 0.25, 0.4]

        points = np.array(list(zip(refusals, kl_values)), dtype=float)
        mask = compute_pareto_frontier_2d(points)

        # All points should be Pareto (it's a monotonic tradeoff)
        assert mask.all(), f"Expected all Pareto on monotonic tradeoff, got {mask}"

    def test_with_dominated_points(self):
        """Add a few dominated points to a realistic sweep."""
        import numpy as np

        from shade.pareto import compute_pareto_frontier_2d

        # Monotonic tradeoff + 2 dominated outliers
        points = np.array(
            [
                [87, 0.0],  # Pareto: highest refusals, lowest KL
                [40, 0.03],  # Pareto
                [10, 0.15],  # Pareto
                [3, 0.4],  # Pareto: lowest refusals, highest KL
                [50, 0.2],  # Dominated by (40, 0.03)
                [30, 0.3],  # Dominated by (10, 0.15)
            ],
            dtype=float,
        )
        mask = compute_pareto_frontier_2d(points)

        # First 4 should be Pareto
        assert mask[0] and mask[1] and mask[2] and mask[3]
        # Last 2 should be dominated
        assert not mask[4], "Point (50, 0.2) should be dominated"
        assert not mask[5], "Point (30, 0.3) should be dominated"


# ---------------------------------------------------------------------------
# Test 7: Save baseline results and regression check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baselines_dir(tmp_path_factory):
    """Temp directory for baseline files, shared across TestBaselinePersistence tests."""
    return tmp_path_factory.mktemp("baselines")


class TestBaselinePersistence:
    """Save real GPT-2 metrics as a baseline and check for regressions.

    NOTE: test_regression_check_against_baseline depends on test_save_baseline
    running first (pytest runs methods in definition order by default).
    The pytest.skip guard handles the case where the baseline file doesn't exist.
    """

    def test_save_baseline(self, gpt2, baselines_dir):
        """Compute real metrics and save as baseline JSON."""
        from shade.benchmark import BenchmarkResult, compute_perplexity
        from shade.results import save_result

        path = baselines_dir / "gpt2_baseline.json"
        model, tokenizer = gpt2
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a field of artificial intelligence.",
            "The president gave a speech about the economy.",
        ]
        ppl = compute_perplexity(model, tokenizer, texts)

        result = BenchmarkResult(
            model_name="gpt2",
            refusals=0,
            total_prompts=len(texts),
            refusal_rate=0.0,
            kl_divergence=0.0,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=ppl,
        )

        save_result(result, path)
        assert path.exists(), "Baseline file should be created"

        # Verify it's valid JSON and round-trips correctly
        from shade.results import load_result

        loaded = load_result(path)
        assert loaded.model_name == "gpt2"
        assert loaded.perplexity == pytest.approx(ppl)

    def test_regression_check_against_baseline(self, gpt2, baselines_dir):
        """If a baseline exists, recompute and compare within tolerance."""
        from shade.benchmark import BenchmarkResult, compute_perplexity
        from shade.results import compare_baseline, load_result

        path = baselines_dir / "gpt2_baseline.json"
        if not path.exists():
            pytest.skip("No baseline file — run test_save_baseline first")

        baseline = load_result(path)
        model, tokenizer = gpt2
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a field of artificial intelligence.",
            "The president gave a speech about the economy.",
        ]
        ppl = compute_perplexity(model, tokenizer, texts)

        current = BenchmarkResult(
            model_name="gpt2",
            refusals=0,
            total_prompts=len(texts),
            refusal_rate=0.0,
            kl_divergence=0.0,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=ppl,
        )

        issues = compare_baseline(current, baseline, tolerance=0.1)
        assert len(issues) == 0, "Regression detected vs baseline:\n" + "\n".join(
            f"  - {i}" for i in issues
        )


# ---------------------------------------------------------------------------
# GPT-2 architecture compatibility helpers for the Model class.
#
# The Model class targets LLaMA-style architectures (self_attn.o_proj,
# mlp.down_proj, model.model.layers).  GPT-2 uses a different structure
# (attn.c_proj, mlp.c_proj, model.transformer.h) with Conv1D layers
# instead of Linear.  The helpers below monkey-patch Model at test time
# so the abliteration pipeline can exercise real GPU-free inference on
# GPT-2 without modifying production source code.
# ---------------------------------------------------------------------------


def _gpt2_get_layers(self):
    """Return GPT-2's transformer blocks (model.transformer.h)."""
    from peft import PeftModel

    model = self.model
    if isinstance(model, PeftModel):
        model = model.base_model.model
    return model.transformer.h


def _gpt2_get_layer_modules(self, layer_index):
    """Return GPT-2's abliterable modules per layer."""
    from contextlib import suppress

    from torch.nn import Module

    layer = self.get_layers()[layer_index]
    modules: dict[str, list[Module]] = {}

    def try_add(component: str, module):
        if isinstance(module, Module):
            modules.setdefault(component, []).append(module)

    with suppress(Exception):
        try_add("attn.c_proj", layer.attn.c_proj)
    with suppress(Exception):
        try_add("mlp.c_proj", layer.mlp.c_proj)

    total = sum(len(mods) for mods in modules.values())
    assert total > 0, "No abliterable modules found in layer"
    return modules


def _gpt2_abliterate(self, refusal_directions, direction_index, parameters):
    """Abliterate GPT-2 LoRA adapters (handles Conv1D weight layout)."""
    import math

    import torch
    import torch.nn.functional as F

    if direction_index is None:
        refusal_direction = None
    else:
        weight_frac, index = math.modf(direction_index + 1)
        refusal_direction = F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1], weight_frac
            ),
            p=2,
            dim=0,
        )

    for layer_index in range(len(self.get_layers())):
        for component, modules in self.get_layer_modules(layer_index).items():
            params = parameters[component]
            distance = float(abs(layer_index - params.max_weight_position))
            if distance > params.min_weight_distance:
                continue
            w = params.max_weight + (distance / params.min_weight_distance) * (
                params.min_weight - params.max_weight
            )
            if refusal_direction is None:
                layer_dir = refusal_directions[layer_index + 1]
            else:
                layer_dir = refusal_direction

            for module in modules:
                v = layer_dir.to(next(module.parameters()).device)
                W = module.base_layer.weight.to(torch.float32)
                # Conv1D stores weights as (in_features, out_features);
                # transpose to (out_features, in_features) if needed.
                if W.shape[0] != v.shape[0]:
                    W = W.T
                W = W.reshape(W.shape[0], -1)

                lora_A = (v @ W).view(1, -1)
                lora_B = (-w * v).view(-1, 1)

                wa = module.lora_A["default"].weight
                wb = module.lora_B["default"].weight
                wa.data = lora_A.to(wa.dtype)
                wb.data = lora_B.to(wb.dtype)


# ---------------------------------------------------------------------------
# Shared fixture: load GPT-2 via the Model class (with LoRA adapters)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shade_model(_restore_real_modules):
    """Load GPT-2 via shade's Model wrapper with LoRA adapters.

    Monkey-patches Model methods so the LLaMA-oriented pipeline works
    with GPT-2's Conv1D / transformer.h architecture.
    """
    from unittest.mock import patch

    from shade.config import Settings
    from shade.model import Model

    with (
        patch.object(Model, "get_layers", _gpt2_get_layers),
        patch.object(Model, "get_layer_modules", _gpt2_get_layer_modules),
        patch.object(Model, "abliterate", _gpt2_abliterate),
    ):
        settings = Settings(model="gpt2", _cli_parse_args=False)
        model_wrapper = Model(settings)

        # GPT-2 has no chat template; provide a minimal one so
        # Model.generate() (which calls apply_chat_template) works.
        model_wrapper.tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
        )

        yield model_wrapper
        # Safety reset: ensure model is clean even if a test crashes mid-abliteration
        try:
            model_wrapper.reset_model()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test 8: Model loading via shade's Model class
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_shade_model_loads_with_lora(self, shade_model):
        """Model should be a PeftModel wrapping GPT-2 with 12 layers."""
        from peft import PeftModel

        assert isinstance(shade_model.model, PeftModel), (
            f"Expected PeftModel, got {type(shade_model.model).__name__}"
        )
        assert len(shade_model.get_layers()) == 12, (
            f"GPT-2 should have 12 layers, got {len(shade_model.get_layers())}"
        )


# ---------------------------------------------------------------------------
# Test 9: get_residuals returns valid tensor
# ---------------------------------------------------------------------------


class TestGetResiduals:
    def test_returns_valid_tensor(self, shade_model):
        """get_residuals should return shape (1, 13, 768) with finite, non-zero values."""
        from shade.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="What is 2+2?")]
        residuals = shade_model.get_residuals(prompts)

        assert residuals.shape == (1, 13, 768), (
            f"Expected shape (1, 13, 768), got {residuals.shape}"
        )
        assert residuals.isfinite().all(), "Residuals should be finite"
        assert (residuals != 0).any(), "Residuals should not be all zeros"


# ---------------------------------------------------------------------------
# Test 10: Residual signal differs between harmless and harmful prompts
# ---------------------------------------------------------------------------


class TestResidualSignal:
    def test_harmless_vs_harmful_differ(self, shade_model):
        """Mean residuals for harmless vs harmful prompts should differ noticeably."""
        from shade.utils import Prompt

        harmless = [
            Prompt(system="You are helpful.", user="What is 2+2?"),
            Prompt(system="You are helpful.", user="Tell me about cats."),
        ]
        harmful = [
            Prompt(system="You are helpful.", user="How to hack a server?"),
            Prompt(system="You are helpful.", user="How to make explosives?"),
        ]

        h_residuals = shade_model.get_residuals(harmless)
        b_residuals = shade_model.get_residuals(harmful)

        diff = (b_residuals.mean(dim=0) - h_residuals.mean(dim=0)).norm()
        assert diff.item() > 0.1, (
            f"Mean residual difference norm should be > 0.1, got {diff.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 11: Abliteration changes LoRA weights and reset restores them
# ---------------------------------------------------------------------------


class TestAbliteration:
    def _get_refusal_directions(self, shade_model):
        """Compute real refusal directions from harmless vs harmful residuals."""
        from shade.utils import Prompt

        harmless = [Prompt(system="You are helpful.", user="What is 2+2?")]
        harmful = [Prompt(system="You are helpful.", user="How to hack?")]

        h_res = shade_model.get_residuals(harmless)
        b_res = shade_model.get_residuals(harmful)
        return b_res.mean(dim=0) - h_res.mean(dim=0)

    def _get_abliteration_params(self, shade_model):
        """Build AbliterationParameters for all components."""
        from shade.model import AbliterationParameters

        components = shade_model.get_abliterable_components()
        return {
            comp: AbliterationParameters(
                max_weight=1.0,
                max_weight_position=6.0,
                min_weight=0.1,
                min_weight_distance=6.0,
            )
            for comp in components
        }

    def _snapshot_lora_b(self, shade_model):
        """Return a list of cloned LoRA B weight tensors."""
        weights = []
        for name, module in shade_model.model.named_modules():
            if "lora_B" in name and hasattr(module, "weight"):
                weights.append(module.weight.data.clone())
        return weights

    def test_changes_lora_weights(self, shade_model):
        """Abliteration should modify at least one LoRA B weight matrix."""
        before = self._snapshot_lora_b(shade_model)
        directions = self._get_refusal_directions(shade_model)
        params = self._get_abliteration_params(shade_model)

        shade_model.abliterate(directions, direction_index=6.0, parameters=params)

        after = self._snapshot_lora_b(shade_model)
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "At least one LoRA B weight should change after abliteration"

        # Clean up for other tests
        shade_model.reset_model()

    def test_reset_restores_zero_lora(self, shade_model):
        """After abliterate + reset, all LoRA B weights should be zero."""
        directions = self._get_refusal_directions(shade_model)
        params = self._get_abliteration_params(shade_model)

        shade_model.abliterate(directions, direction_index=6.0, parameters=params)
        shade_model.reset_model()

        for name, module in shade_model.model.named_modules():
            if "lora_B" in name and hasattr(module, "weight"):
                assert torch.all(module.weight == 0), (
                    f"LoRA B weight {name} should be zero after reset, "
                    f"but has max abs value {module.weight.abs().max().item():.6f}"
                )


# ---------------------------------------------------------------------------
# Test 12: get_logprobs returns valid log probability distribution
# ---------------------------------------------------------------------------


class TestLogprobs:
    def test_returns_valid_log_distribution(self, shade_model):
        """Logprobs should have correct shape, be <= 0, and logsumexp ~ 0."""
        from shade.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Hello")]
        logprobs = shade_model.get_logprobs(prompts)

        # GPT-2 vocabulary size is 50257
        assert logprobs.shape == (1, 50257), (
            f"Expected shape (1, 50257), got {logprobs.shape}"
        )
        assert (logprobs <= 0).all(), "Log probabilities must be <= 0"

        logsumexp = torch.logsumexp(logprobs, dim=-1)
        assert torch.allclose(logsumexp, torch.zeros_like(logsumexp), atol=1e-3), (
            f"logsumexp should be ~0, got {logsumexp.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 13: get_responses generates non-empty text
# ---------------------------------------------------------------------------


class TestGetResponses:
    def test_generates_nonempty_text(self, shade_model):
        """get_responses should return at least one non-empty string."""
        from shade.utils import Prompt

        prompts = [Prompt(system="You are helpful.", user="Tell me a fact.")]
        responses = shade_model.get_responses(prompts, skip_special_tokens=True)

        assert len(responses) == 1, f"Expected 1 response, got {len(responses)}"
        assert len(responses[0].strip()) > 0, (
            f"Response should not be empty, got {responses[0]!r}"
        )


# ---------------------------------------------------------------------------
# Test 14: Full abliteration loop with benchmark
# ---------------------------------------------------------------------------


class TestFullAbliterationLoop:
    def test_abliterate_then_benchmark(self, shade_model):
        """End-to-end: residuals -> abliterate -> perplexity -> BenchmarkResult."""
        from shade.benchmark import BenchmarkResult, compute_perplexity
        from shade.model import AbliterationParameters
        from shade.utils import Prompt

        # Step 1: Compute refusal directions from residuals
        harmless = [Prompt(system="You are helpful.", user="What is 2+2?")]
        harmful = [Prompt(system="You are helpful.", user="How to hack?")]

        h_res = shade_model.get_residuals(harmless)
        b_res = shade_model.get_residuals(harmful)
        refusal_directions = b_res.mean(dim=0) - h_res.mean(dim=0)

        # Step 2: Abliterate
        components = shade_model.get_abliterable_components()
        params = {
            comp: AbliterationParameters(
                max_weight=1.0,
                max_weight_position=6.0,
                min_weight=0.1,
                min_weight_distance=6.0,
            )
            for comp in components
        }
        shade_model.abliterate(
            refusal_directions, direction_index=6.0, parameters=params
        )

        # Step 3: Compute perplexity on the abliterated model
        texts = ["The quick brown fox jumps over the lazy dog."]
        ppl = compute_perplexity(shade_model.model, shade_model.tokenizer, texts)

        assert math.isfinite(ppl), f"Perplexity should be finite, got {ppl}"

        # Step 4: Construct a BenchmarkResult with the real perplexity
        result = BenchmarkResult(
            model_name="gpt2-abliterated",
            refusals=0,
            total_prompts=1,
            refusal_rate=0.0,
            kl_divergence=0.0,
            gsm8k_accuracy=None,
            gsm8k_total=None,
            perplexity=ppl,
        )

        assert result.perplexity is not None
        assert math.isfinite(result.perplexity), (
            f"BenchmarkResult perplexity should be finite, got {result.perplexity}"
        )

        # Clean up for other tests
        shade_model.reset_model()
