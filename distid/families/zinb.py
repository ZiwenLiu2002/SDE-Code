import numpy as np
from scipy.special import gammaln, logsumexp
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
    if m.sum() < 6 or not m[0]:
        return None

    m_pos = m & (k >= 1)
    if m_pos.sum() < 4:
        return None

    k_pos = k[m_pos]
    y_pos = y[m_pos]

    r_grid = np.unique(np.concatenate([
        np.linspace(0.5, 5.0, 10),
        np.linspace(5.0, 20.0, 6),
    ]))

    best = None
    for r in r_grid:
        # For k>0: rearrange log NB(k|r,p) to isolate linear terms in k
        # y(k) = log(1-pi) + r*log(p) + k*log(1-p) + [gammaln terms]
        y_lin = y_pos + gammaln(k_pos + 1.0) + gammaln(r) - gammaln(k_pos + r)
        X = np.vstack([np.ones_like(k_pos), k_pos]).T
        c0, c1 = np.linalg.lstsq(X, y_lin, rcond=None)[0]
        if c1 >= 0.0:
            continue
        log_1mp = float(c1)
        p = float(1.0 - np.exp(log_1mp))
        if not (0.0 < p < 1.0):
            continue
        log_p = float(np.log(p))
        log_1mpi = float(c0 - r * log_p)
        if log_1mpi > 0.0:
            continue
        pi = float(1.0 - np.exp(log_1mpi))
        if not (0.0 < pi < 1.0):
            continue

        log_nb = (gammaln(k + r) - gammaln(k + 1.0) - gammaln(r)
                  + r * log_p + k * log_1mp)
        log_delta0 = np.where(k == 0, 0.0, -1e300)
        y_the = np.logaddexp(np.log(pi) + log_delta0, log_1mpi + log_nb)

        rmse = float(np.sqrt(np.mean((y[m] - y_the[m]) ** 2)))
        mass_err = abs(float(np.exp(logsumexp(y[m]))) - 1.0)
        score = rmse + mass_err

        fr = FitResult(
            "zinb",
            {"pi": pi, "r": float(r), "p": p},
            rmse, mass_err, score, {},
            k=k, logpmf_theory=y_the, valid=True,
        )
        if best is None or fr.score < best.score:
            best = fr

    return best
