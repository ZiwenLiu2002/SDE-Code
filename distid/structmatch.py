import numpy as np
import sympy as sp
from .utils_expand import parse_expr, expand_to_loggamma, split_const_var

__all__ = [
    "analyze_loggamma_terms", "struct_hits_binomial", "struct_hits_negbinomial",
    "struct_hits_hypergeo", "struct_hits_neghypergeo", "struct_hits_poisson", "struct_hits_betanegbinomial"
]

def analyze_loggamma_terms(
    eq_str: str,
    verbose: bool = True,
    coef_int_tol: float = 0.15,
):
    """
    Analyze loggamma structure statistics under var_part:

    - plus_x:   +loggamma(x0 + c)
    - minus_x:  -loggamma(x0 + c)
    - plus_mx:  +loggamma(-x0 + c)
    - minus_mx: -loggamma(-x0 + c)

    Also returns the SymPy expressions for var_part and const_part.

    Important: if a term is a near-integer multiple of loggamma(...), e.g.
      1.9996 * loggamma(x0 + 1)
    then it is counted with multiplicity 2 (within coef_int_tol).
    """
    x0 = sp.Symbol("x0")
    expr = parse_expr(eq_str)
    expr_lg = expand_to_loggamma(expr)
    const_part, var_part = split_const_var(expr_lg)

    plus_x, minus_x, plus_mx, minus_mx = [], [], [], []

    for t in sp.Add.make_args(var_part):
        coef, core = t.as_coeff_Mul()

        if core.func != sp.loggamma:
            continue

        try:
            coef_f = float(coef)
            if not np.isfinite(coef_f) or abs(coef_f) < 1e-12:
                continue
        except Exception:
            continue

        mag = abs(coef_f)
        nearest = int(round(mag))
        mult = 1
        if nearest >= 1 and abs(mag - nearest) <= float(coef_int_tol):
            mult = nearest

        try:
            arg = sp.simplify(core.args[0])
            ax = float(sp.diff(arg, x0))
            bx = float(sp.N(arg.subs({x0: 0})))
        except Exception:
            continue

        if abs(ax - 1.0) < 1e-9:
            if coef_f > 0:
                plus_x.extend([bx] * mult)
            else:
                minus_x.extend([bx] * mult)
        elif abs(ax + 1.0) < 1e-9:
            if coef_f > 0:
                plus_mx.extend([bx] * mult)
            else:
                minus_mx.extend([bx] * mult)

    return {
        "plus_x": plus_x,
        "minus_x": minus_x,
        "plus_mx": plus_mx,
        "minus_mx": minus_mx,
        "var_part": var_part,
        "const_part": const_part,
    }


def struct_hits_binomial(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"]:
        return (False, {})
    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.25:
        return (False, {})
    if not s["minus_mx"]:
        return (False, {})
    bx2 = max(s["minus_mx"])
    n0 = int(round(bx2 - 1.0))
    if n0 < 1:
        return (False, {})
    return (True, {"n0": n0})

def struct_hits_negbinomial(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"] or not s["plus_x"]:
        return (False, {})
    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.25:
        return (False, {})
    r0 = None
    for v in s["plus_x"]:
        if v > 0.5:
            r0 = float(v)
            break
    if r0 is None:
        return (False, {})
    return (True, {"r0": r0})

def struct_hits_hypergeo(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"] or not s["minus_mx"]:
        return (False, {})
    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.25:
        return (False, {})
    ml = sorted(s["minus_mx"], reverse=True)
    if len(ml) < 2:
        return (False, {})
    n0 = int(round(ml[0] - 1.0))
    K0 = int(round(ml[1] - 1.0))
    if n0 < 1 or K0 < 0:
        return (False, {})
    return (True, {"n0": n0, "K0": K0})

def struct_hits_neghypergeo(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"]:
        return (False, {})
    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.25:
        return (False, {})
    if (not s["plus_x"]) or (not s["minus_mx"]):
        return (False, {})
    return (True, {})

def struct_hits_poisson(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"]:
        return (False, {})
    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    return (abs(bx1 - 1.0) <= 0.25, {})

def struct_hits_betanegbinomial(eq_str: str):
    s = analyze_loggamma_terms(eq_str)
    if not s["minus_x"] or len(s["plus_x"]) < 2:
        return (False, {})

    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.25:
        return (False, {})
    others_neg = sorted(s["minus_x"], key=lambda v: v)
    others_neg = [v for v in others_neg if abs(v - bx1) > 1e-6]
    if not others_neg:
        return (False, {})
    tN = max(others_neg)

    pos = sorted([v for v in s["plus_x"] if v > 0.25], reverse=True)
    if len(pos) < 2:
        return (False, {})

    best = None  # (penalty, r0, a0, b0)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            for (r0, a0) in [(pos[i], pos[j]), (pos[j], pos[i])]:
                b0 = tN - r0 - a0
                if b0 <= 0:
                    continue
                pen = 0.0 if b0 > 1.0 else (1.0 - b0)
                cand = (pen, float(r0), float(a0), float(b0))
                if (best is None) or (cand[0] < best[0]):
                    best = cand

    if best is None:
        return (False, {})

    _, r0, a0, b0 = best
    return (True, {"r0": r0, "alpha0": a0, "beta0": b0})
