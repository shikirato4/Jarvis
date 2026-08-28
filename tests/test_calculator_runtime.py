from __future__ import annotations

import math

import pytest

from jarvis.calculator_runtime import CalculatorExpressionError, evaluate_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("781 * 94", 73414),
        ("2 + 3 * 4", 14),
        ("-(5 + 2)", -7),
        ("2^10", 1024),
        ("√81", 9),
        ("200 * 15%", 30),
        ("7 / 2", 3.5),
        ("10 % 3", 1),
        ("2**100", 2**100),
    ],
)
def test_calculator_evaluates_bounded_arithmetic(expression, expected) -> None:
    assert evaluate_expression(expression) == expected


def test_calculator_supports_named_constants_and_functions() -> None:
    assert evaluate_expression("round(pi, 4)") == round(math.pi, 4)
    assert evaluate_expression("abs(-12)") == 12


@pytest.mark.parametrize(
    "expression",
    [
        "1 / 0",
        "2 ** 10001",
        "__import__('os').system('whoami')",
        "[1, 2, 3]",
        "",
    ],
)
def test_calculator_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(CalculatorExpressionError):
        evaluate_expression(expression)
