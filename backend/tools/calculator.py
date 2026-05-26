"""Calculator tool — safely evaluates math expressions without eval().
"""

import ast
import operator
import math


# Supported operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

# Supported functions
FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def calculate(expression: str) -> str:
    """Safely evaluate a math expression.

    Supports: +, -, *, /, ** (power), sqrt(), sin(), cos(),
              tan(), log(), abs(), round(), pi, e

    Args:
        expression: Math expression string (e.g. "144 / 12", "sqrt(169)").

    Returns:
        Result as string, or error message.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Could not evaluate '{expression}': {e}"


def _eval_node(node):
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):       # number: 42, 3.14
        return node.value

    if isinstance(node, ast.Name):           # constant: pi, e
        if node.id in FUNCTIONS:
            return FUNCTIONS[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp):          # binary: 2 + 3, 4 * 5
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")

    if isinstance(node, ast.UnaryOp):        # unary: -5
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    if isinstance(node, ast.Call):           # function: sqrt(16)
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        return FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")
