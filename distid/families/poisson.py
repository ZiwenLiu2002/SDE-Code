import math
import numpy as np
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..utils import finite_mask
from ..structmatch import struct_hits_poisson

def recog(row, cfg):
    K = cfg.get("grid_poisson", 150)
    w_struct = cfg.get("w_struct", 5.0 if cfg.get("prefer_structure", True) else 1.0)

    k = np.arange(0, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    m = finite_mask(y)
    if m.sum() < max(5, K // 5):
        return None

    struct_ok, _ = struct_hits_poisson(str(row.equation))

    y_lin = y[m] + gammaln(k[m] + 1.0)
    X = np.vstack([np.ones_like(k[m]), k[m]]).T
    c0, c1 = np.linalg.lstsq(X, y_lin, rcond=None)[0]
    lam = math.exp(c1)
    y_the = k * math.log(lam) - lam - gammaln(k + 1.0)
    rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
    mass_pred = float(math.exp(logsumexp(y[m])))
    mass_the  = float(math.exp(logsumexp(y_the[m])))
    struct_penalty = 0.0 if struct_ok else 1.0
    score = rmse + abs(mass_pred - mass_the) + abs(c0 + lam) + w_struct * struct_penalty
    return FitResult("poisson", {"lam": float(lam)}, rmse, abs(mass_pred - mass_the),
                     score, {"c0": float(c0), "c1": float(c1), "struct_ok": int(struct_ok)},
                     k=k, logpmf_theory=y_the, valid=True)
