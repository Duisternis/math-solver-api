from sympy.parsing.latex import parse_latex
from sympy import (
    Eq, Derivative, Integral, Limit,
    Symbol, solve, diff, integrate, limit, latex,
    Sum, Product
)
from sympy.core.relational import Relational
from sympy.core.sympify import SympifyError

class MathSolver:
    def __init__(self):
        pass

    def parse_latex(self, latex_expr):
        try:
            return parse_latex(latex_expr)
        except SympifyError as e:
            return None

    def detect_variable(self, expr):
        return list(expr.free_symbols)

    def solve_expression(self, expr):
        try:
            if isinstance(expr, Eq):
                return solve(expr)
            elif isinstance(expr, Derivative):
                var = expr.variables[0] if expr.variables else self.detect_variable(expr)[0]
                return diff(expr.args[0], var)
            elif isinstance(expr, Integral):
                return expr.doit()
            elif isinstance(expr, Limit):
                return expr.doit()
            elif isinstance(expr, (Sum, Product)):
                return expr.doit()
            elif isinstance(expr, Relational):
                return solve(expr)
            elif expr.is_Number or expr.is_Add or expr.is_Mul or expr.is_Pow:
                return expr.evalf()
            else:
                return expr.simplify()
        except Exception as e:
            return f"Error solving expression: {e}"

    def handle_input(self, latex_expr):
        parsed = self.parse_latex(latex_expr)
        if parsed is None:
            return {
                "input": latex_expr,
                "parsed_latex": None,
                "solution_text": "Parsing failed",
                "solution_latex": None
            }

        solution = self.solve_expression(parsed)

        try:
            solution_latex = latex(solution)
        except Exception:
            solution_latex = str(solution)

        return {
            "input": latex_expr,
            "parsed_latex": latex(parsed),
            "solution_text": str(solution),
            "solution_latex": solution_latex
        }
