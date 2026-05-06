import math
import numpy as np
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..utils import finite_mask
from ..structmatch import struct_hits_hypergeo

def _logpmf_hg(k, N, K, n):
    y = -np.inf * np.ones_like(k, dtype=float)
    low = max(0, n - (N - K)); high = min(n, K)
    m = (k >= low) & (k <= high)
    km = k[m]
    y[m] = (gammaln(K + 1.0) - gammaln(km + 1.0) - gammaln(K - km + 1.0)
            + gammaln(N - K + 1.0) - gammaln(n - km + 1.0) - gammaln(N - K - (n - km) + 1.0)
            - (gammaln(N + 1.0) - gammaln(n + 1.0) - gammaln(N - n + 1.0)))
    return y

def recog(row, cfg):
    Kgrid = cfg.get("grid_poisson", 150)
    w_struct = cfg.get("w_struct", 5.0 if cfg.get("prefer_structure", True) else 1.0)

    k = np.arange(0, Kgrid + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    m = finite_mask(y)
    if m.sum() < max(5, Kgrid // 5):
        return None

    struct_ok, payload = struct_hits_hypergeo(str(row.equation))
    n0 = int(payload.get("n0", max(1, int(0.2 * Kgrid))))
    K0 = int(payload.get("K0", max(1, int(0.3 * Kgrid))))

    p_hat = np.exp(y[m] - logsumexp(y[m])); km = k[m]
    mean = float((km * p_hat).sum())
    var  = float(((km - mean) ** 2 * p_hat).sum())
    if var <= 1e-12:
        return None

    best = None
    n_cand = list(range(max(1, n0 - 5), n0 + 6))
    K_cand = list(range(max(1, K0 - 5), K0 + 6))
    for n in n_cand:
        if n >= Kgrid: continue
        mu = mean / n
        if not (1e-6 < mu < 1-1e-6):
            continue
        A = var / (n * mu * (1 - mu))
        if abs(A - 1.0) < 1e-6:
            A = 1.000001
        N_est = (A - n) / (A - 1.0)
        if not np.isfinite(N_est): continue
        N_int = int(round(N_est))
        if N_int < n + 1:
            continue
        for K_ in K_cand:
            if not (0 <= K_ <= N_int):
                continue
            y_the = _logpmf_hg(k, N_int, K_, n)
            mm = np.isfinite(y) & np.isfinite(y_the)
            if mm.sum() < 5:
                continue
            rmse = float(np.sqrt(np.mean((y[mm] - y_the[mm]) ** 2)))
            mass_pred = float(math.exp(logsumexp(y[mm])))
            mass_the  = float(math.exp(logsumexp(y_the[mm])))
            struct_penalty = 0.0 if struct_ok else 1.0
            score = rmse + abs(mass_pred - mass_the) + w_struct * struct_penalty
            fr = FitResult("hypergeometric",
                           {"N": float(N_int), "K": float(K_), "n": float(n)},
                           rmse, abs(mass_pred - mass_the), score,
                           {"struct_ok": int(struct_ok)},
                           k=k, logpmf_theory=y_the, valid=True)
            if (best is None) or (fr.score < best.score):
                best = fr
    return best
