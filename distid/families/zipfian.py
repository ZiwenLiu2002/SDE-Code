import math, numpy as np
from scipy.special import logsumexp
from ..types import FitResult
from ..utils import finite_mask, H_N, zeta_riemann

def _fit_for_loc(row, K: int, loc: int):
    n = np.arange(1, K + 1, dtype=float); k = n + loc
    try: y = row.lambda_format(k.reshape(-1, 1))
    except Exception: return None
    m = finite_mask(y)
    if m.sum() < max(5, K // 5): return None
    x = np.log(n[m]); X = np.vstack([np.ones_like(x), x]).T
    C_hat, slope = np.linalg.lstsq(X, y[m], rcond=None)[0]
    return n, y, float(-slope), float(C_hat)

def _infer_N_from_intercept(a_hat: float, C_hat: float, K: int, N_cap: int = 10_000_000):
    target_H = math.exp(-C_hat); H_inf = zeta_riemann(a_hat)
    if target_H <= 1.0 or target_H > H_inf: return None
    lo = max(1, K)
    if H_N(a_hat, lo) >= target_H: return lo
    hi = lo
    while hi < N_cap and H_N(a_hat, hi) < target_H: hi = min(N_cap, hi * 2)
    if hi >= N_cap and H_N(a_hat, hi) < target_H: return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if H_N(a_hat, mid) < target_H: lo = mid
        else: hi = mid
    return hi

def recog(row, cfg):
    K = cfg.get("grid_pos", 150)
    loc_min, loc_max = cfg.get("loc_min", -5), cfg.get("loc_max", 5)
    best = None
    for loc in range(loc_min, loc_max + 1):
        out = _fit_for_loc(row, K, loc)
        if out is None: continue
        n, y, a_hat, C_hat = out
        if a_hat <= 1.0: continue
        N_hat = _infer_N_from_intercept(a_hat, C_hat, K)
        if N_hat is None: continue
        C_the = -math.log(H_N(a_hat, N_hat))
        y_the = C_the - a_hat * np.log(n)
        rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
        mass_pred = float(math.exp(logsumexp(y)))
        mass_the  = H_N(a_hat, K) / H_N(a_hat, N_hat)
        score = rmse + abs(C_hat - C_the) + abs(mass_pred - mass_the) + cfg.get("zipf_loc_l1", 0.0) * abs(loc)
        fr = FitResult("zipfian", {"a": float(a_hat), "N": float(N_hat), "loc": float(loc)}, rmse,
                       abs(mass_pred - mass_the), score, {"intercept_hat": float(C_hat), "intercept_the": float(C_the)},
                       k=n + loc, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score): best = fr
    return best
