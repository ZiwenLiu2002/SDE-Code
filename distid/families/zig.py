import numpy as np
from scipy.special import logsumexp
from ..types import FitResult
from ..utils import finite_mask


def recog(row, cfg):
    K = cfg.get("grid_pos", 150)

    k = np.arange(0, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None

    m = finite_mask(y)
    if m.sum() < 5 or not m[0]:
        return None

    m_pos = m & (k >= 1)
    if m_pos.sum() < 3:
        return None

    k_pos = k[m_pos]
    y_pos = y[m_pos]
    X = np.vstack([np.ones_like(k_pos), k_pos]).T
    intercept, slope = np.linalg.lstsq(X, y_pos, rcond=None)[0]

    if slope >= 0.0:
        return None
    log_1mp = slope
    log_p = -np.log1p(np.exp(-intercept))

    best = None
    for log_p_cand in np.linspace(-3.0, -0.01, 30):
        p_cand = float(np.exp(log_p_cand))
        log_1mp_cand = float(np.log1p(-p_cand))

        y_adj = y_pos - k_pos * log_1mp_cand
        log_1mpi_plus_logp = float(np.mean(y_adj))
        log_1mpi = log_1mpi_plus_logp - log_p_cand
        if log_1mpi > 0.0:
            continue
        pi_cand = float(1.0 - np.exp(log_1mpi))
        if not (0.0 < pi_cand < 1.0):
            continue

        log_delta0 = np.where(k == 0, 0.0, -1e300)
        log_geo = log_p_cand + k * log_1mp_cand
        y_the = np.logaddexp(np.log(pi_cand) + log_delta0,
                             log_1mpi + log_geo)

        rmse = float(np.sqrt(np.mean((y[m] - y_the[m]) ** 2)))
        mass_err = abs(float(np.exp(logsumexp(y[m]))) - 1.0)
        score = rmse + mass_err

        fr = FitResult(
            "zig",
            {"pi": pi_cand, "p": p_cand},
            rmse, mass_err, score, {},
            k=k, logpmf_theory=y_the, valid=True,
        )
        if best is None or fr.score < best.score:
            best = fr

    return best
