# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Assem Sabry

"""Unit tests for the Shade config module."""

from __future__ import annotations

import textwrap

import pytest

from shade.config import (
    DatasetSpecification,
    QuantizationMethod,
    RowNormalization,
    Settings,
)


class TestQuantizationMethod:
    """Tests for the QuantizationMethod enum."""

    def test_none_value(self):
        assert QuantizationMethod.NONE == "none"
        assert QuantizationMethod.NONE.value == "none"

    def test_bnb_4bit_value(self):
        assert QuantizationMethod.BNB_4BIT == "bnb_4bit"
        assert QuantizationMethod.BNB_4BIT.value == "bnb_4bit"

    def test_members_are_exhaustive(self):
        members = {m.value for m in QuantizationMethod}
        assert members == {"none", "bnb_4bit"}

    def test_is_str_subclass(self):
        assert isinstance(QuantizationMethod.NONE, str)


class TestRowNormalization:
    """Tests for the RowNormalization enum."""

    def test_none_value(self):
        assert RowNormalization.NONE == "none"
        assert RowNormalization.NONE.value == "none"

    def test_pre_value(self):
        assert RowNormalization.PRE == "pre"
        assert RowNormalization.PRE.value == "pre"

    def test_full_value(self):
        assert RowNormalization.FULL == "full"
        assert RowNormalization.FULL.value == "full"

    def test_members_are_exhaustive(self):
        members = {m.value for m in RowNormalization}
        assert members == {"none", "pre", "full"}

    def test_is_str_subclass(self):
        assert isinstance(RowNormalization.PRE, str)


class TestDatasetSpecification:
    """Tests for the DatasetSpecification model."""

    def test_required_fields(self):
        spec = DatasetSpecification(
            dataset="my/dataset",
            split="train",
            column="text",
        )
        assert spec.dataset == "my/dataset"
        assert spec.split == "train"
        assert spec.column == "text"

    def test_default_optional_fields(self):
        spec = DatasetSpecification(
            dataset="my/dataset",
            split="train",
            column="text",
        )
        assert spec.prefix == ""
        assert spec.suffix == ""
        assert spec.system_prompt is None
        assert spec.residual_plot_label is None
        assert spec.residual_plot_color is None

    def test_custom_optional_fields(self):
        spec = DatasetSpecification(
            dataset="my/dataset",
            split="test[:50]",
            column="prompt",
            prefix="Q: ",
            suffix="\nA:",
            system_prompt="Be concise.",
            residual_plot_label="My data",
            residual_plot_color="red",
        )
        assert spec.prefix == "Q: "
        assert spec.suffix == "\nA:"
        assert spec.system_prompt == "Be concise."
        assert spec.residual_plot_label == "My data"
        assert spec.residual_plot_color == "red"

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            DatasetSpecification(dataset="my/dataset")  # missing split and column


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_instantiation(self, default_settings):
        s = default_settings
        assert s.model is None
        assert s.evaluate_model is None
        assert s.batch_size == 0
        assert s.max_response_length == 100
        assert s.n_trials == 200
        assert s.n_startup_trials == 60
        assert s.kl_divergence_scale == 1.0
        assert s.kl_divergence_target == 0.01

    def test_default_quantization(self, default_settings):
        assert default_settings.quantization == QuantizationMethod.NONE

    def test_default_row_normalization(self, default_settings):
        assert default_settings.row_normalization == RowNormalization.NONE

    def test_default_flags(self, default_settings):
        assert default_settings.print_responses is False
        assert default_settings.print_residual_geometry is False
        assert default_settings.plot_residuals is False
        assert default_settings.orthogonalize_direction is False

    def test_default_system_prompt(self, default_settings):
        assert default_settings.system_prompt == "You are a helpful assistant."


class TestSettingsRefusalMarkers:
    """Tests for the default refusal markers list."""

    def test_refusal_markers_count(self, default_settings):
        assert len(default_settings.refusal_markers) >= 27

    def test_known_markers_present(self, default_settings):
        markers = default_settings.refusal_markers
        expected_subset = [
            "sorry",
            "i cannot",
            "as an ai",
            "ai assistant",
            "illegal",
            "harmful",
            "unethical",
            "ethical boundaries",
        ]
        for marker in expected_subset:
            assert marker in markers, f"Expected refusal marker {marker!r} not found"

    def test_markers_are_lowercase(self, default_settings):
        for marker in default_settings.refusal_markers:
            assert marker == marker.lower(), (
                f"Refusal marker {marker!r} is not lowercase"
            )


class TestSettingsCustomValues:
    """Tests for Settings with user-provided overrides."""

    def test_override_scalar_fields(self, tmp_path):
        s = Settings(
            model="meta-llama/Llama-3-8B",
            n_trials=50,
            n_startup_trials=10,
            batch_size=16,
            max_response_length=200,
            kl_divergence_scale=2.5,
            study_checkpoint_dir=str(tmp_path / "ckpt"),
            _env_file=None,
        )
        assert s.model == "meta-llama/Llama-3-8B"
        assert s.n_trials == 50
        assert s.n_startup_trials == 10
        assert s.batch_size == 16
        assert s.max_response_length == 200
        assert s.kl_divergence_scale == 2.5

    def test_override_enum_fields(self, tmp_path):
        s = Settings(
            quantization=QuantizationMethod.BNB_4BIT,
            row_normalization=RowNormalization.FULL,
            study_checkpoint_dir=str(tmp_path / "ckpt"),
            _env_file=None,
        )
        assert s.quantization == QuantizationMethod.BNB_4BIT
        assert s.row_normalization == RowNormalization.FULL


class TestSettingsDatasetDefaults:
    """Tests for the default dataset specifications embedded in Settings."""

    def test_good_prompts_defaults(self, default_settings):
        gp = default_settings.good_prompts
        assert gp.dataset == "mlabonne/harmless_alpaca"
        assert gp.split == "train[:400]"
        assert gp.column == "text"
        assert gp.residual_plot_label == '"Harmless" prompts'
        assert gp.residual_plot_color == "royalblue"

    def test_bad_prompts_defaults(self, default_settings):
        bp = default_settings.bad_prompts
        assert bp.dataset == "mlabonne/harmful_behaviors"
        assert bp.split == "train[:400]"
        assert bp.column == "text"
        assert bp.residual_plot_label == '"Harmful" prompts'
        assert bp.residual_plot_color == "darkorange"

    def test_good_evaluation_prompts_defaults(self, default_settings):
        gep = default_settings.good_evaluation_prompts
        assert gep.dataset == "mlabonne/harmless_alpaca"
        assert gep.split == "test[:100]"
        assert gep.column == "text"

    def test_bad_evaluation_prompts_defaults(self, default_settings):
        bep = default_settings.bad_evaluation_prompts
        assert bep.dataset == "mlabonne/harmful_behaviors"
        assert bep.split == "test[:100]"
        assert bep.column == "text"


class TestSettingsTomlLoading:
    """Tests for loading Settings from a TOML config file."""

    def test_loads_values_from_toml(self, tmp_path, monkeypatch):
        toml_content = textwrap.dedent("""\
            model = "custom/model-from-toml"
            n_trials = 42
            batch_size = 8
            system_prompt = "You are a pirate."
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        # Settings looks for config.toml in the current directory,
        # so we chdir into the tmp_path where the file lives.
        monkeypatch.chdir(tmp_path)

        s = Settings(
            study_checkpoint_dir=str(tmp_path / "ckpt"),
            _env_file=None,
        )
        assert s.model == "custom/model-from-toml"
        assert s.n_trials == 42
        assert s.batch_size == 8
        assert s.system_prompt == "You are a pirate."

    def test_init_overrides_toml(self, tmp_path, monkeypatch):
        toml_content = textwrap.dedent("""\
            n_trials = 42
        """)
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)
        monkeypatch.chdir(tmp_path)

        s = Settings(
            n_trials=999,
            study_checkpoint_dir=str(tmp_path / "ckpt"),
            _env_file=None,
        )
        # Init settings have higher priority than TOML.
        assert s.n_trials == 999
