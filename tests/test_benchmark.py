"""Tests for shade.benchmark module."""

from __future__ import annotations

import pytest

from shade.benchmark import extract_gsm8k_answer


class TestExtractGsm8kAnswer:
    def test_standard_format(self):
        assert extract_gsm8k_answer("Steps.\n#### 42") == "42"

    def test_negative_number(self):
        assert extract_gsm8k_answer("Negative.\n#### -7") == "-7"

    def test_decimal(self):
        assert extract_gsm8k_answer("Dividing.\n#### 3.5") == "3.5"

    def test_comma_separated(self):
        assert extract_gsm8k_answer("Cost.\n#### 1,200") == "1200"

    def test_model_the_answer_is(self):
        assert extract_gsm8k_answer("Work: 5*4=20. The answer is 20.") == "20"

    def test_model_equals(self):
        assert extract_gsm8k_answer("5 + 3 = 8") == "8"

    def test_boxed_latex(self):
        assert extract_gsm8k_answer("Therefore \\boxed{15}") == "15"

    def test_no_answer(self):
        assert extract_gsm8k_answer("I don't know.") is None

    def test_empty(self):
        assert extract_gsm8k_answer("") is None

    def test_whitespace(self):
        assert extract_gsm8k_answer("Steps.\n####   72  ") == "72"

    def test_multiple_numbers_takes_last(self):
        assert extract_gsm8k_answer("Got 10, then 20, answer is 30.") == "30"


class TestComputeGsm8kAccuracy:
    def test_all_correct(self):
        from shade.benchmark import compute_gsm8k_accuracy
        assert compute_gsm8k_accuracy(
            ["42", "The answer is 10", "#### 7"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(1.0)

    def test_all_wrong(self):
        from shade.benchmark import compute_gsm8k_accuracy
        assert compute_gsm8k_accuracy(
            ["99", "The answer is 0", "I don't know"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(0.0)

    def test_partial(self):
        from shade.benchmark import compute_gsm8k_accuracy
        assert compute_gsm8k_accuracy(
            ["42", "wrong", "#### 7"],
            ["#### 42", "#### 10", "#### 7"],
        ) == pytest.approx(2 / 3)

    def test_empty(self):
        from shade.benchmark import compute_gsm8k_accuracy
        assert compute_gsm8k_accuracy([], []) == pytest.approx(0.0)
