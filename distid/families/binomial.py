import math
import numpy as np
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..utils import finite_mask
from ..structmatch import struct_hits_binomial

def recog(row, cfg):
    Kgrid = cfg.get("grid_poisson", 150)
    w_struct = cfg.get("w_struct", 5.0 if cfg.get("prefer_structure", True) else 1.0)

    k = np.arange(0, Kgrid + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    m0 = finite_mask(y)
    if m0.sum() < max(5, Kgrid // 5):
        return None

    struct_ok, payload = struct_hits_binomial(str(row.equation))
    n0 = payload["n0"] if struct_ok else int(np.nanmax(k[m0]))
    n0 = max(n0, 1)

    best = None
    for n in range(max(1, n0 - 5), n0 + 6):
        m = m0 & (k <= n)
        if m.sum() < max(5, int(0.2 * (n + 1))):
            continue
        y_lin = y[m] - (gammaln(n + 1.0) - gammaln(k[m] + 1.0) - gammaln(n - k[m] + 1.0))
        X = np.vstack([np.ones_like(k[m]), k[m]]).T
        a_fit, b_fit = np.linalg.lstsq(X, y_lin, rcond=None)[0]
        p = 1.0 / (1.0 + math.exp(-b_fit))
        if not (0 < p < 1):
            continue
        a_the = n * math.log(1.0 - p)
        y_the = (gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
                 + k * math.log(p) + (n - k) * math.log(1.0 - p))
        rmse = float(np.sqrt(np.mean((y[m] - y_the[m]) ** 2)))
        mass_pred = float(math.exp(logsumexp(y[m])))
        mass_the  = 1.0
        struct_penalty = 0.0 if struct_ok else 1.0
        score = rmse + abs(a_fit - a_the) + abs(mass_pred - mass_the) + w_struct * struct_penalty
        fr = FitResult("binomial", {"n": float(n), "p": float(p)}, rmse,
                       abs(mass_pred - mass_the), score,
                       {"a_fit": float(a_fit), "a_the": float(a_the), "struct_ok": int(struct_ok)},
                       k=k, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score):
            best = fr
    return best
