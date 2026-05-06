import math
import numpy as np
import sympy as sp
from typing import Optional, Tuple
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..utils import finite_mask
from ..utils_expand import parse_expr, expand_to_loggamma, split_const_var

try:
    from scipy.optimize import least_squares
    _HAS_OPT = True
except Exception:
    _HAS_OPT = False

def _bb_logpmf_grid(k, n, a, b):
    y = -np.inf * np.ones_like(k, dtype=float)
    m = (k >= 0) & (k <= n)
    km = k[m]
    y[m] = (gammaln(n + 1.0) - gammaln(km + 1.0) - gammaln(n - km + 1.0)
            + (gammaln(km + a) + gammaln(n - km + b) - gammaln(n + a + b))
            - (gammaln(a) + gammaln(b) - gammaln(a + b)))
    return y

def _structural_guess_from_equation(row) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
    try:
        expr = parse_expr(str(row.equation))
        expr_logg = expand_to_loggamma(expr)
        const_part, var_part = split_const_var(expr_logg)
    except Exception:
        return (False, None, None, None)

    x0 = sp.Symbol("x0")
    a_pos = []   # +loggamma(x0 + A)
    a_neg = []   # -loggamma(x0 + A)
    b_pos = []   # +loggamma(-x0 + C)
    b_neg = []   # -loggamma(-x0 + C)

    for t in sp.Add.make_args(var_part):
        coef, core = t.as_coeff_Mul()
        coef = float(coef)
        if core.func != sp.loggamma:
            continue
        arg = sp.simplify(core.args[0])
        try:
            poly = sp.Poly(arg, x0)
        except Exception:
            continue
        if poly.degree() != 1:
            continue
        a = float(poly.coeffs()[0])
        a = float(poly.coeffs()[0]) if poly.monoms()[0][0] == 1 else float(poly.TC())
        ax = float(poly.coeffs()[0]) if poly.degree() == 1 else 0.0
        bx = float(poly.TC())

        ax = float(poly.coeffs()[0]) if poly.monoms()[0][0] == 1 else 0.0
        bx = float(poly.TC())

        try:
            ax = float(sp.diff(arg, x0))
            bx = float(sp.N(arg.subs({x0:0})))
        except Exception:
            pass

        if abs(ax - 1.0) < 1e-9:
            if coef > 0: a_pos.append(bx)
            else:        a_neg.append(bx)
        elif abs(ax + 1.0) < 1e-9:
            if coef > 0: b_pos.append(bx)
            else:        b_neg.append(bx)

    if not a_pos or not a_neg or not b_pos or not b_neg:
        return (False, None, None, None)

    bx_a_neg = min(a_neg, key=lambda v: abs(v-1.0))
    bx_b_neg = max(b_neg, key=lambda v: v)
    a0 = float(min(a_pos, key=lambda v: v))
    n0 = int(round(bx_b_neg - 1.0))
    b0 = float(max(b_pos) - n0)

    if n0 < 1 or not np.isfinite(a0) or not np.isfinite(b0) or a0 <= 0 or b0 <= 0:
        return (False, None, None, None)
    return (True, n0, a0, b0)

def _fit_alpha_beta_by_ratio(y, k, n, a0, b0):
    m = finite_mask(y)
    m &= (k >= 0) & (k <= n)
    pair = m[:-1] & m[1:] & (k[:-1] <= n-1) & (k[1:] <= n)
    if pair.sum() < 4:
        return None
    k0 = k[:-1][pair]
    dlog = (y[1:][pair] - y[:-1][pair])
    base = np.log(n - k0) - np.log(k0 + 1.0)

    def residual(theta):
        a = max(theta[0], 1e-8); b = max(theta[1], 1e-8)
        return dlog - (base + np.log(k0 + a) - np.log(n - k0 - 1.0 + b))

    if _HAS_OPT:
        res = least_squares(residual, x0=np.array([a0, b0]), bounds=(1e-8, 1e8),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
        a_hat = float(max(res.x[0], 1e-8)); b_hat = float(max(res.x[1], 1e-8))
        return a_hat, b_hat
    a_grid = np.geomspace(max(a0/5,1e-3), a0*5, 25)
    b_grid = np.geomspace(max(b0/5,1e-3), b0*5, 25)
    best = None
    for a in a_grid:
        for b in b_grid:
            r = residual([a,b]); s = float(np.mean(r*r))
            if (best is None) or (s < best[0]): best = (s, a, b)
    return (float(best[1]), float(best[2])) if best else None

def recog(row, cfg) -> Optional[FitResult]:
    K = cfg.get("grid_poisson", 150)
    w_struct = cfg.get("w_struct", 5.0 if cfg.get("prefer_structure", True) else 1.0)

    k = np.arange(0, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    if finite_mask(y).sum() < max(5, K // 5):
        return None

    struct_ok, n0, a0, b0 = _structural_guess_from_equation(row)

    if not struct_ok:
        m = finite_mask(y)
        p = np.exp(y[m] - logsumexp(y[m])); km = k[m]
        mean = float((km * p).sum()); var = float(((km - mean) ** 2 * p).sum())
        if var <= 1e-12: return None
        n_min = max(int(np.ceil(mean + 2*np.sqrt(max(var,1e-12)))), 1)
        n_max = min(max(int(mean + 6*np.sqrt(max(var,1e-12))), n_min+1), int(K*2))
        n0 = max(1, (n_min + n_max)//2)
        mu = min(max(mean / n0, 1e-6), 1-1e-6)
        A = var / (n0 * mu * (1 - mu));  A = 1.000001 if abs(A-1.0)<1e-6 else A
        kappa = (n0 - A) / (A - 1.0);  kappa = 10.0 if kappa <= 0 else kappa
        a0 = max(mu * kappa, 1e-5); b0 = max((1.0-mu) * kappa, 1e-5)

    n_candidates = [n0-5, n0-2, n0-1, n0, n0+1, n0+2, n0+5]
    n_candidates = [int(n) for n in n_candidates if n is not None and n >= 1]

    best = None
    for n in n_candidates:
        ab = _fit_alpha_beta_by_ratio(y, k, n, a0, b0)
        if ab is None: continue
        a_hat, b_hat = ab
        y_the = _bb_logpmf_grid(k, n, a_hat, b_hat)
        mm = np.isfinite(y) & np.isfinite(y_the)
        if mm.sum() < 5: continue
        rmse = float(np.sqrt(np.mean((y[mm] - y_the[mm]) ** 2)))
        mass_pred = float(math.exp(logsumexp(y[mm])))
        mass_the  = float(math.exp(logsumexp(y_the[mm])))
        struct_penalty = 0.0 if struct_ok else 1.0
        score = rmse + abs(mass_pred - mass_the) + w_struct * struct_penalty
        fr = FitResult("betabinomial",
                       {"n": float(n), "alpha": float(a_hat), "beta": float(b_hat)},
                       rmse, abs(mass_pred - mass_the), score,
                       {"struct_ok": int(struct_ok), "a0": a0, "b0": b0},
                       k=k, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score):
            best = fr
    return best
