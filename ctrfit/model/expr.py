"""Small safe evaluator for parameter links such as "h2o_z - 0.2" or "0.5 * (ds1.theta + ds2.theta)".

Only numbers, parameter names (dotted names allowed), + - * / ** %, unary signs, parentheses and
a few math functions are accepted. Nothing is passed to eval or exec.
"""
import ast
import math

FUNCTIONS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi}

_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
}
_UNOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}


class ExprError(ValueError):
    pass


def _dotted(node):
    """Name or Attribute chain -> 'a.b.c', or None if the node is something else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class Expr:
    """A parsed link expression. .names lists the parameter names it refers to."""

    def __init__(self, text):
        self.text = text.strip()
        if not self.text:
            raise ExprError("empty expression")
        try:
            self.tree = ast.parse(self.text, mode="eval").body
        except SyntaxError as ex:
            raise ExprError(f"cannot parse {text!r}: {ex.msg}") from None
        self.names = []
        self._check(self.tree)

    def _check(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError(f"{self.text!r}: only numbers are allowed as constants")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ExprError(f"{self.text!r}: operator {type(node.op).__name__} not allowed")
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNOPS:
                raise ExprError(f"{self.text!r}: operator {type(node.op).__name__} not allowed")
            self._check(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExprError(f"{self.text!r}: only {', '.join(FUNCTIONS)} can be called")
            for a in node.args:
                self._check(a)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = _dotted(node)
            if name is None:
                raise ExprError(f"{self.text!r}: unsupported name")
            if name not in CONSTANTS and name not in self.names:
                self.names.append(name)
        else:
            raise ExprError(f"{self.text!r}: {type(node).__name__} is not allowed")

    def evaluate(self, lookup):
        """lookup(name) -> float for every name in self.names."""
        return float(self._eval(self.tree, lookup))

    def _eval(self, node, lookup):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self._eval(node.left, lookup), self._eval(node.right, lookup))
        if isinstance(node, ast.UnaryOp):
            return _UNOPS[type(node.op)](self._eval(node.operand, lookup))
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*(self._eval(a, lookup) for a in node.args))
        name = _dotted(node)
        return CONSTANTS[name] if name in CONSTANTS else lookup(name)

    def __repr__(self):
        return f"Expr({self.text!r})"
