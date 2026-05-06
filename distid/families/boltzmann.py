from typing import Optional
import numpy as np, math
from scipy.special import logsumexp
from ..types import FitResult

# Z1(β,N) = sum_{k=1}^N e^{-βk} = e^{-β} * (1 - e^{-βN}) / (1 - e^{-β})
# log Z1  = -β + log(1 - e^{-βN}) - log(1 - e^{-β})
def _logZ1(beta: float, N: int) -> float:
    if beta <= 0 or N < 1:
        return math.inf
    eb = math.exp(-beta)
    try:
        return -beta + math.log1p(-math.exp(-beta * N)) - math.log1p(-eb)
    except ValueError:
        return math.inf

# Z0(β,N) = sum_{k=0}^N e^{-βk} = (1 - e^{-β(N+1)}) / (1 - e^{-β})
# log Z0  = log(1 - e^{-β(N+1)}) - log(1 - e^{-β})
def _logZ0(beta: float, N: int) -> float:
    if beta <= 0 or N < 0:
        return math.inf
    eb = math.exp(-beta)
    try:
        return math.log1p(-math.exp(-beta * (N + 1))) - math.log1p(-eb)
    except ValueError:
        return math.inf

def _boltz_logpmf_1toN(k: np.ndarray, beta: float, N: int) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    y = np.full_like(k, -np.inf, dtype=float)
    if beta <= 0 or N < 1:
        return y
    m = (k >= 1) & (k <= N)
    if not np.any(m):
        return y
    logZ = _logZ1(beta, N)
    if not np.isfinite(logZ):
        return y
    y[m] = -beta * k[m] - logZ
    return y

def _boltz_logpmf_0toN(k: np.ndarray, beta: float, N: int) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    y = np.full_like(k, -np.inf, dtype=float)
    if beta <= 0 or N < 0:
        return y
    m = (k >= 0) & (k <= N)
    if not np.any(m):
        return y
    logZ = _logZ0(beta, N)
    if not np.isfinite(logZ):
        return y
    y[m] = -beta * k[m] - logZ
    return y

def _invertN_from_logZ1(beta: float, logZ1: float) -> float:
    if beta <= 0:
        return np.nan
    try:
        t = math.exp(logZ1 + beta) * (1.0 - math.exp(-beta))
        x = 1.0 - t
        if not (0.0 < x < 1.0):
            return np.nan
        return -math.log(x) / beta
    except (OverflowError, ValueError):
        return np.nan

def _invertN_from_logZ0(beta: float, logZ0: float) -> float:
    if beta <= 0:
        return np.nan
    try:
        t = math.exp(logZ0) * (1.0 - math.exp(-beta))
        x = 1.0 - t
        if not (0.0 < x < 1.0):
            return np.nan
        return -math.log(x) / beta - 1.0
    except (OverflowError, ValueError):
        return np.nan

def _median_slope(y: np.ndarray, k: np.ndarray, trim: float = 0.15) -> float:
    m = np.isfinite(y)
    pair = m[:-1] & m[1:]
    if pair.sum() < 4:
        return np.nan
    dy = y[1:][pair] - y[:-1][pair]
    n = len(dy)
    i0 = int(n * trim); i1 = max(i0 + 1, int(n * (1.0 - trim)))
    core = np.sort(dy)[i0:i1]
    if core.size == 0:
        core = dy
    return -float(np.median(core))

def _robust_intercept(y: np.ndarray, k: np.ndarray, beta: float, trim: float = 0.1) -> float:
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    vals = y[m] + beta * k[m]
    vals = np.sort(vals)
    n = len(vals); i0 = int(n * trim); i1 = max(i0 + 1, int(n * (1.0 - trim)))
    core = vals[i0:i1] if i0 < i1 else vals
    return float(np.median(core))

def recog_boltzmann(row, cfg) -> Optional[FitResult]:
    K = int(cfg.get("grid_pos", 150))
    k = np.arange(1, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None

    if not np.isfinite(y).any():
        return None
    m = np.isfinite(y)
    if m.sum() < max(6, K // 6):
        return None

    k_fin = k[m]
    kmax_fin = int(k_fin.max()) if k_fin.size else 0
    truncated_hint = (m.sum() < len(k)) or (kmax_fin < K)

    beta0 = _median_slope(y, k, trim=0.15)
    if not np.isfinite(beta0) or beta0 <= 0:
        return None

    a_hat = _robust_intercept(y, k, beta0, trim=0.1)
    if not np.isfinite(a_hat):
        return None

    logZ_assume = -a_hat
    N1_cont = _invertN_from_logZ1(beta0, logZ_assume)
    N0_cont = _invertN_from_logZ0(beta0, logZ_assume)

    N_seeds = []
    for Nc in (N1_cont, N0_cont):
        if np.isfinite(Nc) and Nc > 0:
            N_seeds.append(int(round(Nc)))
    if not N_seeds:
        N_seeds = [max(5, kmax_fin)]

    beta_cands = [beta0 * f for f in (0.9, 0.95, 1.0, 1.05, 1.1) if beta0 * f > 0]
    cand_tuples = []
    for N0 in N_seeds:
        pad = max(5, int(0.2 * max(N0, 1)))
        lo = max(1, N0 - pad)
        hi = N0 + pad
        if truncated_hint:
            lo = max(lo, kmax_fin)
        for N in range(lo, hi + 1):
            for beta in beta_cands:
                for sup_kind in ("1toN", "0toN"):
                    if sup_kind == "1toN":
                        y_the = _boltz_logpmf_1toN(k, beta, N)
                    else:
                        y_the = _boltz_logpmf_0toN(k, beta, N)

                    mm = np.isfinite(y) & np.isfinite(y_the)
                    if mm.sum() < 6:
                        continue
                    rmse = float(np.sqrt(np.mean((y[mm] - y_the[mm]) ** 2)))
                    mass_pred = float(math.exp(logsumexp(y[mm])))
                    mass_the  = float(math.exp(logsumexp(y_the[mm])))
                    score = rmse + abs(mass_pred - mass_the)

                    if truncated_hint:
                        score *= 0.97

                    cand_tuples.append((score, rmse, abs(mass_pred - mass_the), beta, N, sup_kind, y_the))

    if not cand_tuples:
        return None

    score, rmse, mass_err, beta_star, N_star, sup_kind, y_the = min(cand_tuples, key=lambda t: t[0])
    family_name = "boltzmann"
    return FitResult(
        family_name,
        {"beta": float(beta_star), "N": float(N_star)},
        rmse, mass_err, score,
        extras={"support": sup_kind, "beta0": float(beta0), "a_hat": float(a_hat)},
        k=k, logpmf_theory=y_the, valid=True
    )
