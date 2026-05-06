import numpy as np
import sympy as sp
from scipy.special import gammaln

__all__ = [
    "parse_expr", "expand_to_loggamma", "split_const_var",
    "eval_sym_func", "print_expansion_report",
]

x0 = sp.Symbol("x0")
logB = sp.Function("logB")
logC = sp.Function("logC")
logfac = sp.Function("logfac")
nlogfac = sp.Function("nlogfac")

def parse_expr(eq_str: str):
    return sp.sympify(eq_str, locals={"x0": x0, "logB": logB, "logC": logC,
                                      "logfac": logfac, "nlogfac": nlogfac})

def expand_to_loggamma(expr: sp.Expr) -> sp.Expr:
    A = sp.Wild('A'); B = sp.Wild('B'); N = sp.Wild('N'); K = sp.Wild('K'); T = sp.Wild('T')
    e = expr
    prev = None
    while prev != e:
        prev = e
        e = e.replace(logB(A, B), sp.loggamma(A) + sp.loggamma(B) - sp.loggamma(A + B))
        e = e.replace(logC(N, K), sp.loggamma(N + 1) - sp.loggamma(K + 1) - sp.loggamma(N - K + 1))
        e = e.replace(logfac(T),  sp.loggamma(T + 1))
        e = e.replace(nlogfac(T), -sp.loggamma(T + 1))
    return sp.simplify(sp.expand(e))

def split_const_var(expr_logg: sp.Expr):
    const_terms, var_terms = [], []
    for term in sp.Add.make_args(expr_logg):
        (const_terms if not term.has(x0) else var_terms).append(term)
    const_part = sp.simplify(sp.Add(*const_terms)) if const_terms else sp.Integer(0)
    var_part   = sp.simplify(sp.Add(*var_terms))   if var_terms   else sp.Integer(0)
    return const_part, var_part

def eval_sym_func(expr: sp.Expr, xs: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([x0], expr, modules=[{"loggamma": gammaln}, "numpy"])
    return fn(xs)

def print_expansion_report(eq_str: str, xs: np.ndarray, y_expr: np.ndarray):
    expr = parse_expr(eq_str)
    expr_logg = expand_to_loggamma(expr)
    const_part, var_part = split_const_var(expr_logg)

    print("\n[Expansion to loggamma]")
    print("var_part(x) =")
    print(var_part)
    c_val = float(sp.N(const_part))
    print("\nconst_part C =")
    print(f"{const_part}  ≈  {c_val:.15f}")

    try:
        y_expanded = eval_sym_func(var_part, xs) + c_val
        diff = y_expanded - y_expr
        rmse = float(np.sqrt(np.mean(diff**2)))
        mx   = float(np.max(np.abs(diff)))
        print(f"\n[Expansion check] RMSE={rmse:.12e}, MaxAbsErr={mx:.12e}")
    except Exception as e:
        print(f"\n[Expansion check] FAILED: {e}")
