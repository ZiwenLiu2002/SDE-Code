import math, numpy as np
from scipy.special import logsumexp
from ..types import FitResult
from ..utils import finite_mask

def recog(row, cfg):
    K = cfg.get("grid_signed", 60)
    loc_min, loc_max = cfg.get("loc_min", -5), cfg.get("loc_max", 5)
    best = None
    for loc in range(loc_min, loc_max + 1):
        k = loc + np.arange(-K, K + 1, dtype=float)
        try: y = row.lambda_format(k.reshape(-1, 1))
        except Exception: continue
        m = finite_mask(y)
        if m.sum() < max(5, 2*K//5): continue
        d = np.abs(k[m] - loc); X = np.vstack([np.ones_like(d), d]).T
        C_hat, slope = np.linalg.lstsq(X, y[m], rcond=None)[0]
        a_hat = -slope
        if not (a_hat > 0): continue
        C_the = math.log(math.tanh(a_hat / 2.0))
        y_the = C_the - a_hat * np.abs(k - loc)
        rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
        r = math.exp(-a_hat)
        mass_window = (1.0 + 2.0 * (r * (1.0 - r**K) / (1.0 - r))) * math.tanh(a_hat / 2.0)
        mass_pred = float(math.exp(logsumexp(y)))
        score = rmse + abs(C_hat - C_the) + abs(mass_pred - mass_window)
        fr = FitResult("dlaplace", {"a": float(a_hat), "loc": float(loc)}, rmse,
                       abs(mass_pred - mass_window), score,
                       {"intercept_hat": float(C_hat), "intercept_the": float(C_the)},
                       k=k, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score): best = fr
    return best
