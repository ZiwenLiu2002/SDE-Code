import math, numpy as np
from scipy.special import logsumexp
from ..types import FitResult
from ..utils import finite_mask, zeta_riemann, H_N

def _fit_for_loc(row, K: int, loc: int):
    n = np.arange(1, K + 1, dtype=float); k = n + loc
    try: y = row.lambda_format(k.reshape(-1, 1))
    except Exception: return None
    m = finite_mask(y)
    if m.sum() < max(5, K // 5): return None
    x = np.log(n[m]); X = np.vstack([np.ones_like(x), x]).T
    C_hat, slope = np.linalg.lstsq(X, y[m], rcond=None)[0]
    return n, y, float(-slope), float(C_hat)

def recog(row, cfg):
    K = cfg.get("grid_pos", 150)
    loc_min, loc_max = cfg.get("loc_min", -5), cfg.get("loc_max", 5)
    best = None
    for loc in range(loc_min, loc_max + 1):
        out = _fit_for_loc(row, K, loc)
        if out is None: continue
        n, y, a_hat, C_hat = out
        if a_hat <= 1.0: continue
        C_the = -math.log(zeta_riemann(a_hat))
        y_the = C_the - a_hat * np.log(n)
        rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
        mass_pred = float(math.exp(logsumexp(y)))
        mass_the  = H_N(a_hat, K) / zeta_riemann(a_hat)
        score = rmse + abs(C_hat - C_the) + abs(mass_pred - mass_the) + cfg.get("zipf_loc_l1", 0.0) * abs(loc)
        fr = FitResult("zipf", {"a": float(a_hat), "loc": float(loc)}, rmse,
                       abs(mass_pred - mass_the), score, {"intercept_hat": float(C_hat), "intercept_the": float(C_the)},
                       k=n + loc, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score): best = fr
    return best
