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
    """Remove MagicMock entries from sys.modules so real imports work."""
    mocked_names = [
        name for name, mod in sys.modules.items()
        if isinstance(mod, MagicMock)
    ]
    for name in mocked_names:
        del sys.modules[name]
    yield


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
        assert 30 < ppl < 500, f"GPT-2 perplexity on English text should be in [30, 500], got {ppl}"

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
            model, tokenizer, pairs,
            layers=[5, 6], batch_size=2, show_progress=False,
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
            model, tokenizer, pairs,
            layers=[5, 6], batch_size=2, show_progress=False,
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
                steered_out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
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
        from shade.steering import build_contrastive_pairs, train_steering_vector_from_prompts
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
            model=model, tokenizer=tokenizer,
            harmless_prompts=harmless, harmful_prompts=harmful,
            layers=[4, 5], batch_size=2, show_progress=False,
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
        multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
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
        points = np.array([
            [87, 0.0],    # Pareto: highest refusals, lowest KL
            [40, 0.03],   # Pareto
            [10, 0.15],   # Pareto
            [3, 0.4],     # Pareto: lowest refusals, highest KL
            [50, 0.2],    # Dominated by (40, 0.03)
            [30, 0.3],    # Dominated by (10, 0.15)
        ], dtype=float)
        mask = compute_pareto_frontier_2d(points)

        # First 4 should be Pareto
        assert mask[0] and mask[1] and mask[2] and mask[3]
        # Last 2 should be dominated
        assert not mask[4], "Point (50, 0.2) should be dominated"
        assert not mask[5], "Point (30, 0.3) should be dominated"
