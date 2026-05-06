import math, numpy as np
from scipy.special import logsumexp
from ..types import FitResult
from ..utils import finite_mask

def recog(row, cfg):
    K = cfg.get("grid_pos", 150)
    k = np.arange(1, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    m = finite_mask(y)
    if m.sum() < max(5, K // 5): return None
    X = np.vstack([np.ones_like(k[m]), k[m]]).T
    a_fit, b_fit = np.linalg.lstsq(X, y[m], rcond=None)[0]
    p = 1.0 - math.exp(b_fit)
    if not (0 < p < 1): return None
    a_the = math.log(p) - math.log(1.0 - p)
    y_the = (k - 1.0) * math.log(1.0 - p) + math.log(p)
    rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
    mass_pred = float(math.exp(logsumexp(y[m])))
    mass_the  = float(math.exp(logsumexp(y_the[m])))
    score = rmse + abs(a_fit - a_the) + abs(mass_pred - mass_the)
    return FitResult("geometric(k>=1)", {"p": float(p)}, rmse, abs(mass_pred - mass_the),
                     score, {"a_fit": float(a_fit), "b_fit": float(b_fit), "a_the": float(a_the)},
                     k, y_the, True)
