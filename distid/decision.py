import math
import numpy as np
from .utils import normalize_logpmf, sym_kl, model_logpmf_on_grid
from scipy.special import logsumexp

def discriminate_with_tail_and_skl(row_best,
                                   family_candidates: list,
                                   params_list: list,
                                   k_fit: np.ndarray,
                                   k_ext: np.ndarray,
                                   tail_start_frac: float = 0.6,
                                   w_fit_rmse: float = 1.0,
                                   w_skl: float = 4.0,
                                   w_tail: float = 2.0,
                                   w_taillim: float = 3.0):
    y_ext_full = row_best.lambda_format(k_ext.reshape(-1, 1))
    t0 = int(round(len(k_ext) * tail_start_frac))
    t0 = max(2, min(t0, len(k_ext) - 2))

    finite_ext = np.isfinite(y_ext_full)
    pair_mask_all = finite_ext[:-1] & finite_ext[1:]
    tail_pair_mask = np.zeros_like(pair_mask_all, dtype=bool); tail_pair_mask[t0-1:] = True
    pair_mask_tail = pair_mask_all & tail_pair_mask
    if pair_mask_tail.any():
        R_hat_tail = np.exp(y_ext_full[1:][pair_mask_tail] - y_ext_full[:-1][pair_mask_tail])
        R_hat_tail = np.clip(R_hat_tail, 1e-12, 1e12)
        r_emp = float(np.median(R_hat_tail))
        r_emp = max(1e-6, min(r_emp, 1.0))
    else:
        r_emp = None

    def _r_infinite(fam: str, par: dict):
        f = fam.lower()
        try:
            if f.startswith("zipf"): return 1.0
            if f.startswith("logseries"): return float(par["p"])
            if f.startswith("geometric"): return 1.0 - float(par["p"])
            if f == "poisson": return 0.0
            if f == "binomial": return 0.0
            if f == "dlaplace": return math.exp(-float(par["a"]))
        except Exception:
            return None
        return None

    scores = []
    for fam, par in zip(family_candidates, params_list):
        y_mod_ext = model_logpmf_on_grid(fam, par, k_ext)
        m_ext = np.isfinite(y_ext_full) & np.isfinite(y_mod_ext)
        if m_ext.sum() < 5:
            continue

        p_hat = normalize_logpmf(y_ext_full[m_ext])
        q_mod = normalize_logpmf(y_mod_ext[m_ext])
        skl = sym_kl(p_hat, q_mod)

        val_pair = m_ext[:-1] & m_ext[1:]
        tail_mask_pairs = np.zeros_like(val_pair, dtype=bool); tail_mask_pairs[t0-1:] = True
        val_pair &= tail_mask_pairs
        if val_pair.sum() == 0:
            tail_err = 1e6
        else:
            R_hat = np.exp(y_ext_full[1:][val_pair] - y_ext_full[:-1][val_pair])
            R_mod = np.exp(y_mod_ext[1:][val_pair] - y_mod_ext[:-1][val_pair])
            R_hat = np.clip(R_hat, 1e-12, 1e12); R_mod = np.clip(R_mod, 1e-12, 1e12)
            tail_err = float(np.mean((R_hat - R_mod) ** 2))

        y_fit_expr = row_best.lambda_format(k_fit.reshape(-1, 1))
        y_fit_mod  = model_logpmf_on_grid(fam, par, k_fit)
        m_fit = np.isfinite(y_fit_expr) & np.isfinite(y_fit_mod)
        rmse_fit = float(np.sqrt(np.mean((y_fit_expr[m_fit] - y_fit_mod[m_fit]) ** 2))) if m_fit.sum()>=5 else 1e6

        final = w_fit_rmse * rmse_fit + w_skl * skl + w_tail * tail_err
        r_inf = _r_infinite(fam, par)
        if (r_emp is not None) and (r_inf is not None) and np.isfinite(r_inf):
            r_inf = max(0.0, min(r_inf, 1.0))
            final += w_taillim * abs(r_emp - r_inf)

        scores.append((final, fam, par, {"skl": skl, "tail_err": tail_err, "rmse_fit": rmse_fit}))

    if not scores:
        return {"winner": None, "second": None, "ambiguous": True}

    scores.sort(key=lambda t: t[0])
    best, second = scores[0], (scores[1] if len(scores) > 1 else None)
    ambiguous = second is not None and (second[0] - best[0]) < 1e-3

    return {
        "winner": {"family": best[1], "params": best[2], "score": best[0], **best[3]},
        "second": {"family": second[1], "params": second[2], "score": second[0], **second[3]} if second else None,
        "ambiguous": ambiguous
    }
