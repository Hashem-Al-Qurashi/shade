"""Unit tests for shade.utils module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shade.utils import (
    Prompt,
    batchify,
    check_disk_space,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    is_notebook,
)


# ---------------------------------------------------------------------------
# batchify
# ---------------------------------------------------------------------------
class TestBatchify:
    @pytest.mark.parametrize(
        "items, batch_size, expected",
        [
            pytest.param([], 3, [], id="empty_list"),
            pytest.param([1], 1, [[1]], id="single_item_batch_1"),
            pytest.param(
                [1, 2, 3, 4], 2, [[1, 2], [3, 4]], id="exact_divisible"
            ),
            pytest.param(
                [1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]], id="non_divisible"
            ),
            pytest.param(
                [1, 2, 3], 1, [[1], [2], [3]], id="batch_size_1"
            ),
            pytest.param(
                [1, 2], 10, [[1, 2]], id="batch_size_larger_than_list"
            ),
        ],
    )
    def test_batchify(self, items, batch_size, expected):
        assert batchify(items, batch_size) == expected


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------
class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            pytest.param(0, "0s", id="zero_seconds"),
            pytest.param(45, "45s", id="seconds_only"),
            pytest.param(60, "1m 0s", id="exactly_one_minute"),
            pytest.param(90, "1m 30s", id="minutes_and_seconds"),
            pytest.param(3600, "1h 0m", id="exactly_one_hour"),
            pytest.param(5430, "1h 30m", id="hours_and_minutes"),
            pytest.param(0.4, "0s", id="sub_second_rounds_down"),
            pytest.param(59.6, "1m 0s", id="sub_second_rounds_up_to_minute"),
        ],
    )
    def test_format_duration(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_format_duration_hours_drops_seconds(self):
        """When hours > 0, the output shows hours and minutes only."""
        # 1h 0m 45s => "1h 0m" (seconds are not displayed)
        assert format_duration(3645) == "1h 0m"


# ---------------------------------------------------------------------------
# is_notebook
# ---------------------------------------------------------------------------
class TestIsNotebook:
    def test_returns_false_in_test_environment(self):
        """Standard pytest run is not a notebook environment."""
        assert is_notebook() is False

    def test_returns_true_when_colab_env_set(self):
        with patch.dict("os.environ", {"COLAB_GPU": "1"}):
            assert is_notebook() is True

    def test_returns_true_when_kaggle_env_set(self):
        with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            assert is_notebook() is True


# ---------------------------------------------------------------------------
# Prompt dataclass
# ---------------------------------------------------------------------------
class TestPrompt:
    def test_creation_and_field_access(self):
        p = Prompt(system="You are helpful.", user="Hello")
        assert p.system == "You are helpful."
        assert p.user == "Hello"

    def test_equality(self):
        a = Prompt(system="sys", user="usr")
        b = Prompt(system="sys", user="usr")
        assert a == b

    def test_different_prompts_not_equal(self):
        a = Prompt(system="sys", user="usr1")
        b = Prompt(system="sys", user="usr2")
        assert a != b


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------
class TestCheckDiskSpace:
    def _mock_disk_usage(self, free_gb: float):
        """Return a disk_usage result with the given free space in GB."""
        free_bytes = int(free_gb * (1024**3))
        total_bytes = int(100 * (1024**3))
        used_bytes = total_bytes - free_bytes
        return (total_bytes, used_bytes, free_bytes)

    def test_sufficient_space_returns_true(self):
        with patch("shutil.disk_usage", return_value=self._mock_disk_usage(50.0)):
            assert check_disk_space(required_gb=10.0) is True

    def test_insufficient_space_returns_false(self):
        with patch("shutil.disk_usage", return_value=self._mock_disk_usage(5.0)):
            assert check_disk_space(required_gb=10.0) is False

    def test_exact_threshold_returns_true(self):
        with patch("shutil.disk_usage", return_value=self._mock_disk_usage(10.0)):
            assert check_disk_space(required_gb=10.0) is True


# ---------------------------------------------------------------------------
# get_trial_parameters
# ---------------------------------------------------------------------------
class TestGetTrialParameters:
    @staticmethod
    def _make_trial(direction_index, parameters):
        """Create a mock trial with the given user_attrs."""
        trial = SimpleNamespace()
        trial.user_attrs = {
            "direction_index": direction_index,
            "parameters": parameters,
        }
        return trial

    def test_per_layer_direction(self):
        trial = self._make_trial(
            direction_index=None,
            parameters={"attention": {"scale": 0.75}},
        )
        result = get_trial_parameters(trial)
        assert result["direction_index"] == "per layer"
        assert result["attention.scale"] == "0.75"

    def test_fixed_direction_index(self):
        trial = self._make_trial(
            direction_index=0.5,
            parameters={"mlp": {"weight": 1.234}},
        )
        result = get_trial_parameters(trial)
        assert result["direction_index"] == "0.50"
        assert result["mlp.weight"] == "1.23"

    def test_multiple_components_and_params(self):
        trial = self._make_trial(
            direction_index=0.1,
            parameters={
                "attn": {"alpha": 2.0, "beta": 3.5},
                "ffn": {"gamma": 0.001},
            },
        )
        result = get_trial_parameters(trial)
        assert result["direction_index"] == "0.10"
        assert result["attn.alpha"] == "2.00"
        assert result["attn.beta"] == "3.50"
        assert result["ffn.gamma"] == "0.00"


# ---------------------------------------------------------------------------
# get_readme_intro
# ---------------------------------------------------------------------------
class TestGetReadmeIntro:
    @staticmethod
    def _make_trial(direction_index, parameters, kl_divergence, refusals):
        trial = SimpleNamespace()
        trial.user_attrs = {
            "direction_index": direction_index,
            "parameters": parameters,
            "kl_divergence": kl_divergence,
            "refusals": refusals,
        }
        return trial

    def test_contains_expected_markdown_sections(self, default_settings):
        default_settings.model = "org/some-model"

        trial = self._make_trial(
            direction_index=None,
            parameters={"attn": {"scale": 1.0}},
            kl_divergence=0.0042,
            refusals=3,
        )
        bad_prompts = [Prompt(system="sys", user=f"prompt_{i}") for i in range(10)]

        with patch("shade.utils.version", return_value="2.0.0-test"):
            result = get_readme_intro(
                default_settings, trial, base_refusals=8, bad_prompts=bad_prompts
            )

        # Check the top heading references the model with a HF link.
        assert "[org/some-model](https://huggingface.co/org/some-model)" in result
        # Check the markdown contains expected section headers.
        assert "## Abliteration parameters" in result
        assert "## Performance" in result
        # Check trial parameters appear.
        assert "per layer" in result
        assert "attn.scale" in result
        # Check KL divergence value.
        assert "0.0042" in result
        # Check refusal counts.
        assert "3/10" in result
        assert "8/10" in result

    def test_local_model_path_is_hidden(self, default_settings, tmp_path):
        """When the model is a local directory, the path is hidden for privacy."""
        model_dir = tmp_path / "my-secret-model"
        model_dir.mkdir()
        default_settings.model = str(model_dir)

        trial = self._make_trial(
            direction_index=0.5,
            parameters={},
            kl_divergence=0.01,
            refusals=0,
        )

        with patch("shade.utils.version", return_value="2.0.0-test"):
            result = get_readme_intro(
                default_settings, trial, base_refusals=5, bad_prompts=[]
            )

        assert "a model" in result
        assert str(model_dir) not in result
