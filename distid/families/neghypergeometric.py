import math
import numpy as np
from scipy.special import gammaln
from ..types import FitResult
from ..utils import finite_mask
from ..structmatch import analyze_loggamma_terms, struct_hits_neghypergeo

def _logpmf_nhg(k, N, K, r):
    y = -np.inf * np.ones_like(k, dtype=float)
    m = (k >= 0) & (k <= (N - K)) & np.isfinite(k)
    if not np.any(m):
        return y
    km = k[m]
    y[m] = (
        gammaln(km + r) - gammaln(km + 1.0) - gammaln(r)
        + gammaln(N - r - km + 1.0)
        - gammaln(N - K - km + 1.0)
    )
    return y

def _align_rmse(y_obs, y_the):
    mm = np.isfinite(y_obs) & np.isfinite(y_the)
    if mm.sum() < 5:
        return math.inf
    delta = float(np.mean(y_obs[mm] - y_the[mm]))
    diff = (y_obs[mm] - (y_the[mm] + delta))
    return float(np.sqrt(np.mean(diff * diff)))

def _guess_from_structure(eq_str, debug=False):
    """
    Read (N, K, r) initial guesses from the loggamma expansion.

    var_part ≈ +logΓ(x+r) - logΓ(x+1) + logΓ(-x + (N-r+1)) - logΓ(-x + (N-K+1))

    Returns a de-duplicated list of candidate (N0, K0, r0) tuples; may be empty.
    """
    s = analyze_loggamma_terms(eq_str)

    if (not s["minus_x"]) or (not s["plus_x"]) or (not s["plus_mx"]) or (not s["minus_mx"]):
        return []

    bx1 = min(s["minus_x"], key=lambda v: abs(v - 1.0))
    if abs(bx1 - 1.0) > 0.35:
        return []

    r_cands = [v for v in s["plus_x"] if v > 0.25]
    A_cands = sorted(s["plus_mx"])
    B_cands = sorted(s["minus_mx"])

    out = set()
    for r0 in r_cands:
        for A in A_cands:
            N0 = A + r0 - 1.0
            if N0 <= 1.0:
                continue
            for B in B_cands:
                K0 = N0 - B + 1.0
                if K0 <= 0:
                    continue
                if (r0 <= K0 + 1e-6) and (K0 <= N0 + 1e-6):
                    out.add((
                        int(max(2, round(N0))),
                        int(max(1, round(K0))),
                        int(max(1, round(r0)))
                    ))
    if debug:
        print(f"[NHG-struct] plus_x={s['plus_x']} minus_x={s['minus_x']} "
              f"plus_mx={s['plus_mx']} minus_mx={s['minus_mx']}")
        print(f"[NHG-struct] guesses={sorted(out)}")
    return sorted(out)

def _neighbors_int(v, pct=0.12, absw=6, lo=1, hi=10**9):
    S = {int(v)}
    for d in (-absw, -max(1, absw//2), -2, -1, 0, 1, 2, max(1, absw//2), absw):
        S.add(int(round(v + d)))
    for f in (1.0 - pct, 1.0 - pct/2, 1.0 + pct/2, 1.0 + pct):
        S.add(int(round(v * f)))
    return sorted([int(min(max(x, lo), hi)) for x in S])

def recog(row, cfg):
    Kgrid = int(cfg.get("grid_poisson", 150))
    n_cap = int(cfg.get("binom_n_cap", 100000))
    debug = bool(cfg.get("debug_gate", False))

    k = np.arange(0, Kgrid + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None
    m = finite_mask(y)
    if m.sum() < max(5, Kgrid // 5):
        return None

    eq_str = str(row.equation)
    struct_ok, _ = struct_hits_neghypergeo(eq_str)

    init_cands = _guess_from_structure(eq_str, debug=debug)

    cand_list = []
    for (N0, K0, r0) in init_cands:
        N0 = int(max(2, min(n_cap, N0)))
        K0 = int(max(1, min(N0 - 1, K0)))
        r0 = int(max(1, min(K0, r0)))
        cand_list.append((N0, K0, r0))
        Ns = _neighbors_int(N0, lo=2, hi=n_cap)
        Ks = _neighbors_int(K0, lo=1, hi=N0-1)
        rs = _neighbors_int(r0, lo=1, hi=K0)
        for N in Ns:
            K_hi = max(1, min(N - 1, max(Ks)))
            for K_ in Ks:
                if 1 <= K_ <= N - 1:
                    for r_ in rs:
                        if 1 <= r_ <= K_:
                            cand_list.append((N, K_, r_))

    if not cand_list:
        p_hat = np.exp(y[m] - np.log(np.sum(np.exp(y[m]))))
        mean_est = float((k[m] * p_hat).sum())
        for N in [int(2*mean_est)+10, int(3*mean_est)+20, int(4*mean_est)+40]:
            N = int(max(2, min(n_cap, N)))
            for r in [1, 2, 3, 5, 10]:
                for K_ in [int(N*0.2), int(N*0.4), int(N*0.6)]:
                    if 1 <= K_ <= N-1 and r <= K_:
                        cand_list.append((N, K_, r))

    cand_list = sorted(set(cand_list))

    best = None
    for (N, K, r) in cand_list:
        y_the = _logpmf_nhg(k, N, K, r)
        s = _align_rmse(y, y_the)
        if not math.isfinite(s):
            continue
        fr = FitResult(
            "neghypergeometric",
            {"N": float(N), "K": float(K), "r": float(r)},
            rmse=s,
            mass_err=0.0,
            score=s,
            extras={
                "struct_ok": int(bool(struct_ok)),
                "from_struct": int(len(init_cands) > 0),
            },
            k=k, logpmf_theory=y_the, valid=True
        )
        if (best is None) or (fr.score < best.score):
            best = fr

    if debug and best is not None:
        print(f"[NHG] best N={best.params['N']:.0f}, K={best.params['K']:.0f}, r={best.params['r']:.0f}, rmse={best.rmse:.4g}")

    return best
