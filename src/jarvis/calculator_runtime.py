from __future__ import annotations

import ast
import math
import operator
import re


class CalculatorExpressionError(ValueError):
    pass


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def evaluate_expression(expression: str) -> int | float:
    normalized = _normalize_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorExpressionError("invalid calculator expression") from exc
    value = _evaluate_node(tree.body, depth=0)
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorExpressionError("calculator result is not finite")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_expression(expression: str) -> str:
    value = str(expression).strip()
    if not value or len(value) > 512:
        raise CalculatorExpressionError("calculator expression length is invalid")
    value = value.replace("×", "*").replace("÷", "/").replace("−", "-")
    value = value.replace("^", "**")
    value = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", value)
    value = re.sub(
        r"(?<![\w)])(\d+(?:\.\d+)?)\s*%(?=\s*(?:$|[+\-*/^),]))",
        r"(\1/100)",
        value,
    )
    return value


def _evaluate_node(node: ast.AST, *, depth: int) -> int | float:
    if depth > 64:
        raise CalculatorExpressionError("calculator expression is too deeply nested")
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](
            _evaluate_node(node.operand, depth=depth + 1)
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, depth=depth + 1)
        right = _evaluate_node(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
            raise CalculatorExpressionError("calculator exponent is too large")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise CalculatorExpressionError(type(exc).__name__) from exc
        if isinstance(result, int) and result.bit_length() > 1_000_000:
            raise CalculatorExpressionError("calculator integer result is too large")
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        arguments = [_evaluate_node(item, depth=depth + 1) for item in node.args]
        try:
            return _FUNCTIONS[node.func.id](*arguments)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise CalculatorExpressionError(type(exc).__name__) from exc
    raise CalculatorExpressionError(
        f"unsupported calculator syntax: {type(node).__name__}"
    )
