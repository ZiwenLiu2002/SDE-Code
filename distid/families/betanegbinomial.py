from typing import Optional
import numpy as np, math
import sympy as sp
from scipy.special import gammaln, logsumexp
from ..types import FitResult
from ..structures import expand_to_loggamma

def recog_betanegbinomial(row, cfg) -> Optional[FitResult]:
    K = int(cfg.get("grid_poisson", 150))
    k = np.arange(0, K + 1, dtype=float)
    try:
        y = row.lambda_format(k.reshape(-1, 1))
    except Exception:
        return None

    m = np.isfinite(y)
    if m.sum() < max(5, K // 5):
        return None

    def _logpmf_bnb(kv, r, a, b):
        yy = -np.inf * np.ones_like(kv, dtype=float)
        if r <= 0 or a <= 0 or b <= 0:
            return yy
        mm = (kv >= 0)
        kk = kv[mm]
        yy[mm] = (
            gammaln(kk + r) - gammaln(r) - gammaln(kk + 1.0) +
            (gammaln(kk + a) + gammaln(r + b) - gammaln(kk + r + a + b)) -
            (gammaln(a) + gammaln(b) - gammaln(a + b))
        )
        return yy

    def _score_params(r, a, b):
        y_the = _logpmf_bnb(k, r, a, b)
        mm = np.isfinite(y) & np.isfinite(y_the)
        if mm.sum() < 5:
            return np.inf, None
        rmse = float(np.sqrt(np.mean((y[mm] - y_the[mm])**2)))
        mass_pred = float(math.exp(logsumexp(y[mm])))
        mass_the  = float(math.exp(logsumexp(y_the[mm])))
        mass_err  = abs(mass_pred - mass_the)
        return rmse + mass_err, (rmse, mass_err, y_the)

    struct_used = False
    best = None  # (score, r, a, b, rmse, mass_err, y_the)

    try:
        exp = expand_to_loggamma(str(row.equation))
        var_part = exp["var_part"]
        x0 = sp.Symbol("x0")

        pos_offsets, neg_offsets = [], []
        for term in sp.Add.make_args(sp.expand(var_part)):
            coeff = 1.0
            core = term
            if isinstance(term, sp.Mul):
                coeff, core = term.as_coeff_Mul()
            if getattr(core, "func", None) != sp.loggamma:
                continue
            arg = core.args[0]
            a = sp.diff(arg, x0)
            if not a.is_Number:
                continue
            a_val = float(a)
            if abs(a_val - 1.0) > 1e-8:
                continue
            c_val = float(sp.N(arg.subs(x0, 0)))
            if float(coeff) >= 0:
                pos_offsets.append(c_val)
            else:
                neg_offsets.append(c_val)

        neg_offsets_sorted = sorted(neg_offsets, key=lambda t: abs(t - 1.0))
        k1 = None
        tol_one = float(cfg.get("bnb_tol_k1", 0.2))
        if neg_offsets_sorted and abs(neg_offsets_sorted[0] - 1.0) <= tol_one:
            k1 = neg_offsets_sorted[0]
            others_neg = [t for t in neg_offsets if t is not k1]
        else:
            others_neg = list(neg_offsets)

        if len(pos_offsets) >= 2 and len(others_neg) >= 1:
            tN = max(others_neg)
            for i in range(len(pos_offsets)):
                for j in range(i+1, len(pos_offsets)):
                    for (r0, a0) in [(pos_offsets[i], pos_offsets[j]),
                                     (pos_offsets[j], pos_offsets[i])]:
                        b0 = tN - r0 - a0
                        if b0 <= 1.0:
                            continue
                        s, extra = _score_params(r0, a0, b0)
                        if np.isfinite(s):
                            rmse, me, y_the = extra
                            cand = (s, r0, a0, b0, rmse, me, y_the)
                            if (best is None) or (cand[0] < best[0]):
                                best = cand
            struct_used = best is not None
    except Exception:
        struct_used = False

    if best is None:
        logZ = float(logsumexp(y[m]))
        p_hat = np.exp(y[m] - logZ)
        km = k[m]
        m1 = float((km * p_hat).sum())

        alpha_grid = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
        beta_grid  = np.array([2.2, 3.0, 5.0, 10.0, 20.0, 50.0], dtype=float)

        for a in alpha_grid:
            for b in beta_grid:
                r0 = (m1 * (b - 1.0) / a) if a > 0 else None
                if r0 is None or r0 <= 0:
                    continue
                s, extra = _score_params(r0, a, b)
                if np.isfinite(s):
                    rmse, me, y_the = extra
                    cand = (s, r0, a, b, rmse, me, y_the)
                    if (best is None) or (cand[0] < best[0]):
                        best = cand

        if best is None:
            return None

    mults = np.array([0.8, 0.9, 1.0, 1.15, 1.3], dtype=float)
    _, r_c, a_c, b_c, _, _, _ = best
    for _ in range(2):
        best_round = best
        for fr in mults:
            for fa in mults:
                for fb in mults:
                    r2 = max(1e-6, r_c * fr)
                    a2 = max(1e-6, a_c * fa)
                    b2 = max(1.05, b_c * fb)
                    s, extra = _score_params(r2, a2, b2)
                    if np.isfinite(s):
                        rmse, me, y_the = extra
                        cand = (s, r2, a2, b2, rmse, me, y_the)
                        if cand[0] < best_round[0]:
                            best_round = cand
        best = best_round
        _, r_c, a_c, b_c, _, _, _ = best

    s, r_best, a_best, b_best, rmse_best, mass_err_best, y_best = best

    if struct_used:
        s *= float(cfg.get("bnb_struct_bonus", 0.9))

    return FitResult(
        "betanegbinomial",
        {"r": float(r_best), "alpha": float(a_best), "beta": float(b_best)},
        rmse_best,
        mass_err_best,
        s,
        {"struct_used": int(struct_used)},
        k=k,
        logpmf_theory=y_best,
        valid=True,
    )
