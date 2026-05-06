from typing import Optional
import numpy as np, math
from scipy.special import gammaln, logsumexp
import sympy as sp
from ..types import FitResult
from ..utils_expand import parse_expr, expand_to_loggamma, split_const_var

def _yulesimon_logpmf(k: np.ndarray, rho: float) -> np.ndarray:
    y = -np.inf * np.ones_like(k, dtype=float)
    if rho <= 0:
        return y
    m = (k >= 1) & np.isfinite(k)
    kk = k[m]
    y[m] = (math.log(rho) + gammaln(rho + 1.0)
            + gammaln(kk) - gammaln(kk + rho + 1.0))
    return y

def _struct_seed_rho(eq_str: str) -> Optional[float]:
    try:
        x0 = sp.Symbol("x0")
        expr = parse_expr(eq_str)
    except Exception:
        return None

    c_candidates = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) and getattr(node, 'func', None) == sp.Function('logB'):
            pass
    try:
        expr_lg = expand_to_loggamma(expr)  # logB/logC/logfac → loggamma
        const_part, var_part = split_const_var(expr_lg)
        const_val = float(const_part.evalf())
    except Exception:
        expr_lg = None
        const_val = None

    try:
        s = str(eq_str)
        i = 0
        while True:
            j = s.find("logB(", i)
            if j < 0:
                break
            p = j + 5
            depth = 1
            args_str = ""
            while p < len(s) and depth > 0:
                ch = s[p]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth > 0:
                    args_str += ch
                p += 1
            parts = [a.strip() for a in args_str.split(",", 1)]
            if len(parts) == 2:
                a1 = sp.sympify(parts[0], locals={"x0": x0})
                a2 = sp.sympify(parts[1], locals={"x0": x0})
                if a1 == x0 and a2.is_number:
                    c_candidates.append(float(a2))
                elif a2 == x0 and a1.is_number:
                    c_candidates.append(float(a1))
            i = p
    except Exception:
        pass

    if not c_candidates:
        if expr_lg is None:
            return None
        plus_x, minus_x = [], []
        x0 = sp.Symbol("x0")
        for t in sp.Add.make_args(var_part):
            coef, core = t.as_coeff_Mul()
            if core.func != sp.loggamma:
                continue
            arg = sp.simplify(core.args[0])
            try:
                ax = float(sp.diff(arg, x0))
                bx = float(sp.N(arg.subs({x0: 0})))
            except Exception:
                continue
            if abs(ax - 1.0) < 1e-9:
                (plus_x if coef > 0 else minus_x).append(bx)
        if (len(plus_x) >= 1) and (len(minus_x) >= 1):
            bx0 = min(plus_x, key=lambda v: abs(v - 0.0))
            if abs(bx0) <= 0.15:
                c = max(minus_x)
                c_candidates.append(c)

    if not c_candidates:
        return None

    c = max(c_candidates)
    rho0 = c - 1.0
    if rho0 <= 0:
        return None

    try:
        if const_val is not None:
            rho1 = math.exp(const_val - float(gammaln(c)))
            if rho1 > 0:
                return float((rho0 + rho1) / 2.0)
    except Exception:
        pass

    return float(rho0)

def recog_yulesimon(row, cfg) -> Optional[FitResult]:
    K = int(cfg.get("grid_pos", 150))
    tail_frac = float(cfg.get("discr_tail_start_frac", 0.6))
    k = np.arange(1, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None

    m = np.isfinite(y)
    if m.sum() < max(6, K // 6):
        return None

    eq_str = str(row.equation)
    rho_seed = _struct_seed_rho(eq_str)

    val_pair = m[:-1] & m[1:]
    if val_pair.sum() < 5:
        return None
    t0 = max(1, int(round(len(k) * tail_frac)) - 1)
    tail_pair = np.zeros_like(val_pair, dtype=bool); tail_pair[t0:] = True
    pair = val_pair & tail_pair
    if pair.sum() < 5:
        pair = val_pair
    kval = k[:-1][pair]
    R_emp = np.exp(y[1:][pair] - y[:-1][pair]); R_emp = np.clip(R_emp, 1e-12, 1e12)

    def ratio_mse(rho: float) -> float:
        if rho <= 0:
            return np.inf
        R_mod = kval / (kval + rho + 1.0)
        return float(np.mean((R_emp - R_mod) ** 2))

    best = None
    if rho_seed and rho_seed > 0:
        for f in (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5):
            r = max(1e-6, rho_seed * f)
            s = ratio_mse(r)
            if (best is None) or (s < best[0]):
                best = (s, r)
    else:
        grid = np.unique(np.concatenate([
            np.logspace(-1.2, 1.7, 36),
            np.array([0.2, 0.5, 1, 2, 3, 5, 10, 20], dtype=float)
        ]))
        for r in grid:
            s = ratio_mse(r)
            if (best is None) or (s < best[0]):
                best = (s, float(r))

    rho = best[1]
    for _ in range(2):
        improved = best
        for f in (0.7, 0.85, 1.0, 1.15, 1.3):
            r2 = max(1e-6, rho * f)
            s2 = ratio_mse(r2)
            if s2 < improved[0]:
                improved = (s2, r2)
        best = improved; rho = best[1]

    y_the = _yulesimon_logpmf(k, rho)
    mm = np.isfinite(y) & np.isfinite(y_the)
    if mm.sum() < 6:
        return None
    rmse = float(np.sqrt(np.mean((y[mm] - y_the[mm]) ** 2)))
    mass_pred = float(math.exp(logsumexp(y[mm])))
    mass_the  = float(math.exp(logsumexp(y_the[mm])))
    tail_mse = best[0]
    score = rmse + abs(mass_pred - mass_the) + tail_mse

    return FitResult(
        "yulesimon",
        {"rho": float(rho)},
        rmse=rmse,
        mass_err=abs(mass_pred - mass_the),
        score=score,
        extras={
            "tail_ratio_mse": float(tail_mse),
            "rho_seed": float(rho_seed) if rho_seed else np.nan
        },
        k=k,
        logpmf_theory=y_the,
        valid=True
    )
