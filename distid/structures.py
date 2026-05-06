import re
import numpy as np
import sympy as sp
from scipy.special import gammaln
from .utils_expand import parse_expr, expand_to_loggamma as _expand_to_loggamma, split_const_var, x0

_PAT_LOGC   = re.compile(r"\blogc\s*\(", re.I)
_PAT_LOGB   = re.compile(r"\blogb\s*\(", re.I)
_PAT_GAMMA  = re.compile(r"\b(logfac|nlogfac|loggamma|gammaln)\b", re.I)
_PAT_ABS    = re.compile(r"\babs\s*\(", re.I)
_PAT_LOGX0  = re.compile(r"\blog\s*\(\s*x0\b", re.I)

def extract_structure_flags(equation: str):
    s = equation.lower()
    return {
        "has_logc": bool(_PAT_LOGC.search(s)),
        "has_logb": bool(_PAT_LOGB.search(s)),
        "has_gamma": bool(_PAT_GAMMA.search(s)),
        "has_abs": bool(_PAT_ABS.search(s)),
        "has_logx": bool(_PAT_LOGX0.search(s)),
    }

def expand_to_loggamma(expr_str: str):
    expr_sym = parse_expr(expr_str)
    expr_logg = _expand_to_loggamma(expr_sym)
    const_part, var_part = split_const_var(expr_logg)
    var_fn = sp.lambdify([x0], var_part, modules=[{"loggamma": gammaln}, "numpy"])
    return {
        "expr_sym": expr_sym,
        "expr_loggamma": expr_logg,
        "var_part": var_part,
        "const_part": const_part,
        "var_fn": var_fn,
    }
