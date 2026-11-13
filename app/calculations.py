import ast
import re
from decimal import Decimal, InvalidOperation


_WORD_OPERATORS = (
    (r"\bdivided\s+by\b", "/"),
    (r"\bmultiplied\s+by\b", "*"),
    (r"\b(?:times|x)\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\b(?:δια|διαιρεμενο\s+με|διαιρεμενη\s+με)\b", "/"),
    (r"\b(?:επι|φορες)\b", "*"),
    (r"\bσυν\b", "+"),
    (r"\bμειον\b", "-"),
)

_QUESTION_WRAPPERS = (
    r"\bwhat\s+is\b",
    r"\bcalculate\b",
    r"\bhow\s+much\s+is\b",
    r"\bποσο\s+(?:κανει|ειναι)\b",
    r"\bυπολογισε\b",
)


def calculate_spoken_expression(text: str) -> tuple[str, str] | None:
    """Return a safe arithmetic expression and result for explicit calculations."""
    normalized = _without_accents(text.casefold())
    has_word_operator = any(re.search(pattern, normalized) for pattern, _ in _WORD_OPERATORS)
    has_wrapper = any(re.search(pattern, normalized) for pattern in _QUESTION_WRAPPERS)
    has_symbol_operator = bool(re.search(r"\d\s*[+*/]\s*\d", normalized))
    if not (has_word_operator or has_wrapper or has_symbol_operator):
        return None

    expression = normalized
    for pattern in _QUESTION_WRAPPERS:
        expression = re.sub(pattern, " ", expression)
    for pattern, operator in _WORD_OPERATORS:
        expression = re.sub(pattern, f" {operator} ", expression)

    expression = expression.replace("×", "*").replace("÷", "/")
    expression = re.sub(r"[^0-9+\-*/().\s]", " ", expression)
    expression = " ".join(expression.split())
    if not expression or not re.search(r"[+\-*/]", expression):
        return None

    try:
        node = ast.parse(expression, mode="eval")
        result = _evaluate(node.body)
    except (SyntaxError, ValueError, ZeroDivisionError, InvalidOperation):
        return None

    if not result.is_finite() or abs(result) > Decimal("1e100"):
        return None
    return expression, _format_decimal(result)


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported arithmetic expression")


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".")


def _without_accents(text: str) -> str:
    import unicodedata

    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
