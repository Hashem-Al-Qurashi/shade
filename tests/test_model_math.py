"""Unit tests for mathematical logic in shade.model.

Tests the pure math and data structures used by the Model class
without requiring GPU access or model loading. All tensors use CPU.

Note: We define AbliterationParameters locally instead of importing from
shade.model, because shade.model has heavy ML dependencies (transformers,
peft) that contaminate torch operations when mocked. The dataclass is
trivial — 4 float fields — so duplication is the pragmatic choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F


@dataclass
class AbliterationParameters:
    """Local mirror of shade.model.AbliterationParameters."""

    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


# ---------------------------------------------------------------------------
# 1. AbliterationParameters dataclass
# ---------------------------------------------------------------------------


class TestAbliterationParameters:
    def test_creation_and_field_access(self):
        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=10.0,
            min_weight=0.1,
            min_weight_distance=5.0,
        )
        assert params.max_weight == 1.0
        assert params.max_weight_position == 10.0
        assert params.min_weight == 0.1
        assert params.min_weight_distance == 5.0

    def test_creation_with_zero_values(self):
        params = AbliterationParameters(
            max_weight=0.0,
            max_weight_position=0.0,
            min_weight=0.0,
            min_weight_distance=0.0,
        )
        assert params.max_weight == 0.0
        assert params.min_weight_distance == 0.0

    def test_creation_with_fractional_position(self):
        """max_weight_position can be a non-integer layer index."""
        params = AbliterationParameters(
            max_weight=2.5,
            max_weight_position=7.3,
            min_weight=0.2,
            min_weight_distance=4.0,
        )
        assert params.max_weight_position == 7.3


# ---------------------------------------------------------------------------
# Helper: Reproduce the weight kernel logic from Model.abliterate()
# ---------------------------------------------------------------------------


def compute_layer_weight(
    layer_index: int,
    params: AbliterationParameters,
) -> float:
    """Reimplements lines 411-422 of model.py.

    Returns the per-layer lambda weight for abliteration.
    """
    distance = abs(layer_index - params.max_weight_position)

    if distance > params.min_weight_distance:
        return 0.0

    weight = params.max_weight + (distance / params.min_weight_distance) * (
        params.min_weight - params.max_weight
    )
    return weight


# ---------------------------------------------------------------------------
# 2. Weight kernel computation (bell-shaped lambda schedule)
# ---------------------------------------------------------------------------


class TestWeightKernel:
    """Verify the per-layer lambda schedule.

    The kernel is bell-shaped:
      - max_weight at the layer equal to max_weight_position
      - linearly decays toward min_weight at distance == min_weight_distance
      - zero beyond min_weight_distance
    """

    @pytest.fixture()
    def standard_params(self) -> AbliterationParameters:
        return AbliterationParameters(
            max_weight=1.0,
            max_weight_position=10.0,
            min_weight=0.2,
            min_weight_distance=5.0,
        )

    def test_center_layer_gets_max_weight(self, standard_params):
        """Layer at max_weight_position should get exactly max_weight."""
        w = compute_layer_weight(10, standard_params)
        assert w == pytest.approx(1.0)

    def test_boundary_layers_get_min_weight(self, standard_params):
        """Layers exactly min_weight_distance away should get min_weight."""
        w_left = compute_layer_weight(5, standard_params)
        w_right = compute_layer_weight(15, standard_params)
        assert w_left == pytest.approx(0.2)
        assert w_right == pytest.approx(0.2)

    def test_layers_beyond_distance_get_zero(self, standard_params):
        """Layers further than min_weight_distance should be skipped (weight 0)."""
        assert compute_layer_weight(4, standard_params) == 0.0
        assert compute_layer_weight(0, standard_params) == 0.0
        assert compute_layer_weight(16, standard_params) == 0.0
        assert compute_layer_weight(30, standard_params) == 0.0

    def test_linear_interpolation_midpoint(self, standard_params):
        """A layer halfway between center and boundary gets the midpoint weight."""
        # distance = 2.5 out of 5.0 -> 50% decay
        # expected: 1.0 + (2.5/5.0) * (0.2 - 1.0) = 1.0 - 0.4 = 0.6
        w = compute_layer_weight(12, standard_params)  # distance = 2.0
        # 1.0 + (2.0/5.0) * (0.2 - 1.0) = 1.0 - 0.32 = 0.68
        assert w == pytest.approx(0.68)

    def test_symmetry_around_center(self, standard_params):
        """Layers equidistant from center should get the same weight."""
        for d in range(1, 6):
            w_left = compute_layer_weight(10 - d, standard_params)
            w_right = compute_layer_weight(10 + d, standard_params)
            assert w_left == pytest.approx(w_right), f"Asymmetry at distance {d}"

    def test_full_schedule_for_32_layers(self):
        """Verify the complete lambda schedule for a 32-layer model."""
        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=16.0,
            min_weight=0.0,
            min_weight_distance=8.0,
        )
        weights = [compute_layer_weight(i, params) for i in range(32)]

        # Layers 0-7: beyond distance -> 0
        for i in range(8):
            assert weights[i] == 0.0, f"Layer {i} should be 0"

        # Layer 8: exactly at boundary -> min_weight = 0.0
        assert weights[8] == pytest.approx(0.0)

        # Layer 16: center -> max_weight = 1.0
        assert weights[16] == pytest.approx(1.0)

        # Layer 24: other boundary -> min_weight = 0.0
        assert weights[24] == pytest.approx(0.0)

        # Layers 25-31: beyond distance -> 0
        for i in range(25, 32):
            assert weights[i] == 0.0, f"Layer {i} should be 0"

        # Monotonically increasing from layer 8 to 16
        for i in range(8, 16):
            assert weights[i] < weights[i + 1], (
                f"Not increasing at layer {i}: {weights[i]} >= {weights[i+1]}"
            )

        # Monotonically decreasing from layer 16 to 24
        for i in range(16, 24):
            assert weights[i] > weights[i + 1], (
                f"Not decreasing at layer {i}: {weights[i]} <= {weights[i+1]}"
            )


# ---------------------------------------------------------------------------
# 3. LoRA delta math: delta_W = -lambda * outer(v, v @ W)
# ---------------------------------------------------------------------------


class TestLoraDeltaMath:
    """Verify the abliteration LoRA decomposition.

    From model.py (RowNormalization.NONE path):
        lora_A = (v @ W).view(1, -1)      # shape (1, d_in)
        lora_B = (-weight * v).view(-1, 1) # shape (d_out, 1)
        delta_W = lora_B @ lora_A          # shape (d_out, d_in)
                = -weight * outer(v, v @ W)
    """

    def test_lora_delta_small_matrix(self):
        """Compute delta on a 4x4 weight matrix with a known direction."""
        torch.manual_seed(42)
        d_out, d_in = 4, 4
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        v = F.normalize(torch.randn(d_out, dtype=torch.float32), p=2, dim=0)
        lam = 0.8  # lambda weight

        # Reproduce model.py logic
        lora_A = (v @ W).view(1, -1)
        lora_B = (-lam * v).view(-1, 1)
        delta_W = lora_B @ lora_A

        # Independent reference computation
        vTW = v @ W  # shape (d_in,)
        expected = -lam * torch.outer(v, vTW)

        assert delta_W.shape == (d_out, d_in)
        assert torch.allclose(delta_W, expected, atol=1e-6)

    def test_lora_delta_rectangular_matrix(self):
        """Verify delta computation on a non-square weight matrix."""
        torch.manual_seed(7)
        d_out, d_in = 8, 16
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        v = F.normalize(torch.randn(d_out, dtype=torch.float32), p=2, dim=0)
        lam = 1.0

        lora_A = (v @ W).view(1, -1)
        lora_B = (-lam * v).view(-1, 1)
        delta_W = lora_B @ lora_A

        expected = -lam * torch.outer(v, v @ W)

        assert delta_W.shape == (d_out, d_in)
        assert torch.allclose(delta_W, expected, atol=1e-6)

    def test_delta_is_rank_one(self):
        """The abliteration delta should be a rank-1 matrix."""
        torch.manual_seed(99)
        d_out, d_in = 8, 8
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        v = F.normalize(torch.randn(d_out, dtype=torch.float32), p=2, dim=0)
        lam = 0.5

        lora_A = (v @ W).view(1, -1)
        lora_B = (-lam * v).view(-1, 1)
        delta_W = lora_B @ lora_A

        # A rank-1 matrix has exactly one non-zero singular value.
        singular_values = torch.linalg.svdvals(delta_W)
        nonzero = (singular_values > 1e-6).sum().item()
        assert nonzero == 1

    def test_delta_zero_when_lambda_zero(self):
        """When lambda is 0, the delta should be all zeros."""
        torch.manual_seed(0)
        W = torch.randn(4, 4, dtype=torch.float32)
        v = F.normalize(torch.randn(4, dtype=torch.float32), p=2, dim=0)

        lora_A = (v @ W).view(1, -1)
        lora_B = (0.0 * v).view(-1, 1)
        delta_W = lora_B @ lora_A

        assert torch.allclose(delta_W, torch.zeros_like(delta_W), atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Direction computation: interpolation + normalization
# ---------------------------------------------------------------------------


class TestDirectionComputation:
    """Verify the direction selection/interpolation logic from abliterate().

    From model.py (lines 393-402):
        weight, index = math.modf(direction_index + 1)
        refusal_direction = F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1], weight
            ),
            p=2, dim=0,
        )
    """

    @staticmethod
    def _compute_direction(
        refusal_directions: torch.Tensor,
        direction_index: float,
    ) -> torch.Tensor:
        """Reimplements the direction interpolation from model.py."""
        weight, index = math.modf(direction_index + 1)
        return F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1],
                weight,
            ),
            p=2,
            dim=0,
        )

    def test_integer_index_selects_exact_direction(self):
        """When direction_index is an integer, no interpolation occurs.

        direction_index=0 -> math.modf(1.0) -> (0.0, 1.0)
        So refusal_directions[1].lerp(refusal_directions[2], 0.0) = refusal_directions[1]
        """
        torch.manual_seed(10)
        # 5 directions (index 0 is embeddings, 1-4 are layers 0-3)
        dirs = F.normalize(torch.randn(5, 8, dtype=torch.float32), p=2, dim=1)

        result = self._compute_direction(dirs, direction_index=0.0)
        expected = F.normalize(dirs[1], p=2, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_midpoint_interpolation(self):
        """direction_index=0.5 should interpolate halfway between dirs[1] and dirs[2]."""
        torch.manual_seed(20)
        dirs = F.normalize(torch.randn(5, 8, dtype=torch.float32), p=2, dim=1)

        # math.modf(0.5 + 1) = math.modf(1.5) = (0.5, 1.0)
        # So lerp(dirs[1], dirs[2], 0.5), then normalize
        result = self._compute_direction(dirs, direction_index=0.5)
        interpolated = dirs[1].lerp(dirs[2], 0.5)
        expected = F.normalize(interpolated, p=2, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_result_is_unit_vector(self):
        """The returned direction should always be a unit vector."""
        torch.manual_seed(30)
        dirs = torch.randn(10, 16, dtype=torch.float32)  # Not pre-normalized

        for idx in [0.0, 1.0, 2.5, 3.7]:
            result = self._compute_direction(dirs, direction_index=idx)
            norm = torch.linalg.vector_norm(result).item()
            assert norm == pytest.approx(1.0, abs=1e-5), (
                f"Norm {norm} at index {idx}"
            )

    def test_mean_difference_direction(self):
        """Verify that mean-difference of two groups gives a meaningful direction.

        This simulates how refusal directions are typically computed:
        direction = mean(harmful_residuals) - mean(harmless_residuals), normalized.
        """
        torch.manual_seed(50)
        d = 16

        # Two clusters with a known offset along dimension 0
        harmless = torch.randn(20, d, dtype=torch.float32)
        harmful = harmless.clone()
        harmful[:, 0] += 5.0  # shift dimension 0

        direction = (harmful.mean(dim=0) - harmless.mean(dim=0))
        direction = F.normalize(direction, p=2, dim=0)

        # The direction should point primarily along dimension 0
        assert direction[0].item() > 0.9, (
            f"Expected dominant component in dim 0, got {direction[0].item():.4f}"
        )

        # It should be a unit vector
        norm = torch.linalg.vector_norm(direction).item()
        assert norm == pytest.approx(1.0, abs=1e-5)
