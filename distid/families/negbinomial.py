import math
import numpy as np
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..utils import finite_mask
from ..structmatch import struct_hits_negbinomial

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

    struct_ok, payload = struct_hits_negbinomial(str(row.equation))
    r0 = payload.get("r0", 2.0)

    r_grid = np.unique(np.concatenate([
        np.array([max(0.5, r0/3), max(0.75, r0/2), r0, r0*1.5, r0*2, r0*3]),
        np.linspace(0.5, 20.0, 10)
    ]))
    best = None
    for r in r_grid:
        y_lin = y[m] + gammaln(r) + gammaln(k[m] + 1.0) - gammaln(k[m] + r)
        X = np.vstack([np.ones_like(k[m]), k[m]]).T
        c0, c1 = np.linalg.lstsq(X, y_lin, rcond=None)[0]
        p = math.exp(c1)
        if not (0 < p < 1):
            continue
        c0_the = r * math.log(1.0 - p)
        y_the = (gammaln(k + r) - gammaln(r) - gammaln(k + 1.0)
                 + r * math.log(1.0 - p) + k * math.log(p))
        rmse = float(np.sqrt(np.mean((y - y_the) ** 2)))
        mass_pred = float(math.exp(logsumexp(y[m])))
        mass_the  = float(math.exp(logsumexp(y_the[m])))
        struct_penalty = 0.0 if struct_ok else 1.0
        score = rmse + abs(c0 - c0_the) + abs(mass_pred - mass_the) + w_struct * struct_penalty
        fr = FitResult("negbinomial", {"r": float(r), "p": float(p)}, rmse,
                       abs(mass_pred - mass_the), score,
                       {"c0": float(c0), "c1": float(c1), "struct_ok": int(struct_ok)},
                       k=k, logpmf_theory=y_the, valid=True)
        if (best is None) or (fr.score < best.score):
            best = fr
    return best
