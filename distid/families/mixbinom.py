import re
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from ..types import FitResult
from ..utils import finite_mask

EPS = 1e-9


def _infer_K_from_eq(eq_str: str) -> int:
    """Infer mixture K from logaddexp count, discounting the ZI wrapper logaddexp.

    A ZI distribution contributes exactly 1 logaddexp (the pi-mix with logdelta0).
    Each remaining logaddexp merges one additional mixture component, so:
        K = (total_logaddexp - zi_logaddexp) + 1
    """
    total = len(re.findall(r"logaddexp", eq_str, re.IGNORECASE))
    has_zi = bool(re.search(r"logdelta0", eq_str, re.IGNORECASE))
    mixture_lae = total - (1 if has_zi else 0)
    return max(1, mixture_lae + 1)


def _logbinom_pmf(k, n, p):
    lp = float(np.log(max(p, EPS)))
    l1mp = float(np.log(max(1.0 - p, EPS)))
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0) + k * lp + (n - k) * l1mp


def _mix_logpmf(k, n, ps, ws):
    K_comp = len(ps)
    terms = np.array([np.log(ws[i]) + _logbinom_pmf(k, n, ps[i]) for i in range(K_comp)])
    return logsumexp(terms, axis=0)


_MIN_WEIGHT = 0.02  # reject solutions where any component has < 2% weight


def _fit_mixture(k_vals, y_obs, n, K_comp, n_restarts=None, extra_inits=None):
    """Fit K_comp-component binomial mixture by minimising RMSE vs y_obs."""
    if n_restarts is None:
        n_restarts = max(30, 8 * K_comp)

    m = np.isfinite(y_obs)
    k_m = k_vals[m]
    y_m = y_obs[m]

    def unpack(x):
        ps = 1.0 / (1.0 + np.exp(-np.clip(x[:K_comp], -500.0, 500.0)))
        if K_comp == 1:
            ws = np.array([1.0])
        else:
            raw_w = np.exp(np.clip(x[K_comp: 2 * K_comp], -500.0, 500.0))
            ws = raw_w / raw_w.sum()
            ws = np.clip(ws, 1e-300, 1.0)
        return ps, ws

    def loss(x):
        ps, ws = unpack(x)
        y_pred = np.array([_mix_logpmf(ki, n, ps, ws) for ki in k_m])
        return float(np.mean((y_pred - y_m) ** 2))

    best_res = None
    rng = np.random.default_rng(42)
    inits = []
    if extra_inits:
        inits.extend(extra_inits)
    for _ in range(n_restarts):
        x0_p = rng.uniform(-3.0, 3.0, K_comp)
        x0_w = rng.uniform(0.0, 1.0, K_comp)
        inits.append(np.concatenate([x0_p, x0_w]))

    for x0 in inits:
        try:
            res = minimize(loss, x0, method="Nelder-Mead",
                           options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-9})
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        except Exception:
            continue

    if best_res is None:
        return None, None, None, None

    ps, ws = unpack(best_res.x)
    if K_comp > 1 and float(ws.min()) < _MIN_WEIGHT:
        return None, None, None, None

    y_the = np.array([_mix_logpmf(ki, n, ps, ws) for ki in k_vals])
    rmse = float(np.sqrt(np.mean((y_obs[m] - y_the[m]) ** 2)))
    return ps, ws, rmse, y_the


def recog(row, cfg):
    Kgrid = cfg.get("grid_poisson", 150)
    K_penalty = cfg.get("mixbinom_K_penalty", 0.05)
    K_max_try = cfg.get("mixbinom_max_K", 6)

    k = np.arange(0, Kgrid + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None

    m = finite_mask(y)
    if m.sum() < 4:
        return None

    log_Z = float(logsumexp(y[m]))

    eq_str = str(row.equation)
    K_struct = _infer_K_from_eq(eq_str)

    K_candidates = sorted(set(
        k_ for k_ in [K_struct, K_struct + 1]
        if 1 <= k_ <= K_max_try
    ))

    slope_hits = re.findall(
        r'x0\s*\*\s*([+-]?[0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', eq_str, re.IGNORECASE)
    struct_logit_ps = [float(s) for s in slope_hits]
    const_lae_args = re.findall(r'logaddexp\(\s*([0-9.e+\-]+)\s*,', eq_str, re.IGNORECASE)
    struct_logit_ps += [0.0] * len(const_lae_args)
    if re.search(r'logaddexp\(\s*x0\s*,', eq_str, re.IGNORECASE):
        struct_logit_ps.append(1.0)

    # gammaln-based logC is continuous so the PMF is finite even for k > true n,
    # making k_max on a wide grid unreliable as a proxy for n; prefer parsing n from logC(n, x0).
    logc_match = re.search(r'logC\(\s*([0-9.e+\-]+)\s*,\s*x0\s*\)', eq_str, re.IGNORECASE)
    if logc_match:
        n_from_eq = float(logc_match.group(1))
        n_base = int(round(n_from_eq))
        n_cands = sorted(set(n_ for n_ in range(max(1, n_base - 1), n_base + 3)))
    else:
        k_max = int(k[m].max())
        n_cands = sorted(set(range(max(1, k_max - 2), k_max + 3)))

    best_overall = None

    for n_cand in n_cands:
        km = m & (k <= n_cand)
        if km.sum() < 4:
            continue
        k_sub = k[km]
        # Re-normalize within k=0..n_cand so the equation and mixture model are on the same log-scale.
        # The original global normalization makes y[k<=n_cand] very negative when
        # the continuous gammaln logC gives finite mass at large k outside [0, n].
        y_sub_raw = y[km]
        finite_sub = np.isfinite(y_sub_raw)
        if finite_sub.sum() < 4:
            continue
        log_Z_sub = float(logsumexp(y_sub_raw[finite_sub]))
        y_sub = y_sub_raw - log_Z_sub
        mass_err_global = abs(float(np.exp(log_Z_sub - log_Z)) - 1.0)

        for K_comp in K_candidates:
            extra_inits = []
            if len(struct_logit_ps) >= K_comp:
                logits = struct_logit_ps[:K_comp]
                x0_struct = np.array(logits + [0.0] * K_comp, dtype=float)
                extra_inits.append(x0_struct)
            result = _fit_mixture(k_sub, y_sub, n_cand, K_comp, extra_inits=extra_inits)
            if result[0] is None:
                continue
            ps, ws, _, y_the_sub = result

            y_the = np.array([_mix_logpmf(ki, n_cand, ps, ws) for ki in k])
            rmse = float(np.sqrt(np.mean((y_sub - y_the[km]) ** 2)))
            score = rmse + mass_err_global + K_penalty * (K_comp - 1)

            params = {"n": float(n_cand), "K": float(K_comp)}
            for i, (pi_val, wi_val) in enumerate(zip(ps, ws), 1):
                params[f"p{i}"] = float(pi_val)
                params[f"w{i}"] = float(wi_val)

            fr = FitResult(
                "mixbinom",
                params,
                rmse, mass_err_global, score,
                {"K_struct": float(K_struct), "K_components": float(K_comp)},
                k=k, logpmf_theory=y_the, valid=True,
            )
            if best_overall is None or fr.score < best_overall.score:
                best_overall = fr

    return best_overall
