"""CLI smoke tests using Click's CliRunner.

No models are loaded — these test that CLI commands are wired
correctly, help text is accurate, and error handling works.
"""

from __future__ import annotations

from click.testing import CliRunner

from shade.cli import cli

runner = CliRunner()


class TestBenchmarkCLI:
    def test_help_shows_options(self):
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--gsm8k" in result.output
        assert "--gsm8k-limit" in result.output
        assert "--perplexity" in result.output
        assert "--output" in result.output

    def test_no_model_id_exits_with_error(self):
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code != 0


class TestSteerCLI:
    def test_help_shows_subcommands(self):
        result = runner.invoke(cli, ["steer", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "apply" in result.output
        assert "evaluate" in result.output

    def test_train_help_shows_options(self):
        result = runner.invoke(cli, ["steer", "train", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--layers" in result.output
        assert "--batch-size" in result.output

    def test_apply_help_shows_options(self):
        result = runner.invoke(cli, ["steer", "apply", "--help"])
        assert result.exit_code == 0
        assert "--vector" in result.output
        assert "--multiplier" in result.output

    def test_evaluate_help_shows_options(self):
        result = runner.invoke(cli, ["steer", "evaluate", "--help"])
        assert result.exit_code == 0
        assert "--vector" in result.output
        assert "--gsm8k" in result.output
        assert "--perplexity" in result.output


class TestCompareCLI:
    def test_help_shows_options(self):
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--gsm8k" in result.output
        assert "--perplexity" in result.output
        assert "--steering-multiplier" in result.output
        assert "--n-trials" in result.output
        assert "--output" in result.output
