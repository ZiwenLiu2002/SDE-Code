import re
import math
import numpy as np
from .structmatch import analyze_loggamma_terms

NON_LOGGAMMA_FAMS = {"zipf", "zipfian", "logseries", "geometric", "dlaplace", "boltzmann"}

def _has_any_loggamma_tokens(eq_str: str) -> bool:
    s = eq_str.lower()
    return any(tok in s for tok in ["gammaln", "loggamma", "logfac", "nlogfac", "logc(", "logb("])

def _safe_eval_row(row, k: np.ndarray):
    try:
        y = row.lambda_format(k.reshape(-1, 1))
        if np.isscalar(y):
            y = np.full_like(k, float(y), dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != k.shape[0]:
            return None
        return y
    except Exception:
        return None

def _lin_mse(y: np.ndarray, X: np.ndarray):
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(np.mean(resid**2)), beta
    except Exception:
        return math.inf, None

ABS_RE = re.compile(r'\babs\s*\(|\bAbs\s*\(', re.I)
SQ_ABS_RE = re.compile(
    r'sqrt\s*\(\s*\(\s*x0\s*[\+\-][^)]*\)\s*\*\s*\(\s*x0\s*[\+\-][^)]*\)\s*\)',
    re.I,
)

def _has_abs_like(eqs: str) -> bool:
    s = eqs.replace('**', '^')
    if ABS_RE.search(s):
        return True
    if 'sqrt' in s and SQ_ABS_RE.search(s):
        return True
    return False

def gate_families(row, cfg_dict, s_counts=None, has_gamma=None):
    eqs = str(row.equation)
    strict_struct_filter      = bool(cfg_dict.get("strict_struct_filter", True))
    strict_mutual_exclusion   = bool(cfg_dict.get("strict_mutual_exclusion", True))
    prefer_boltz_no_abs       = bool(cfg_dict.get("prefer_boltzmann_without_abs", True))
    dl_noabs_override_margin  = float(cfg_dict.get("dl_noabs_override_margin", 0.90))
    dl_margin                 = float(cfg_dict.get("dlaplace_margin", 0.65))
    debug_gate                = bool(cfg_dict.get("debug_gate", False))

    if has_gamma is None:
        has_gamma = _has_any_loggamma_tokens(eqs)

    if has_gamma:
        s = s_counts if (s_counts is not None) else analyze_loggamma_terms(eqs, verbose=False)
        n_plus_x, n_minus_x = len(s["plus_x"]), len(s["minus_x"])
        n_plus_mx, n_minus_mx = len(s["plus_mx"]), len(s["minus_mx"])

        cand_scores = {}

        hit_bin = (n_minus_x >= 1 and n_minus_mx >= 1)
        hit_hg  = (n_minus_x >= 1 and n_minus_mx >= 2)
        hit_nhg = (n_plus_x >= 1 and n_minus_x >= 1 and n_minus_mx >= 1)

        if hit_nhg:
            cand_scores["neghypergeometric"] = 0.00
        if hit_hg:
            cand_scores["hypergeometric"] = min(cand_scores.get("hypergeometric", 1.0), 0.00)
        if hit_bin:
            base = 0.20 if (hit_hg or hit_nhg) else 0.00
            cand_scores["binomial"] = min(cand_scores.get("binomial", 1.0), base)

        if (n_plus_x >= 1 and n_minus_x >= 1):
            cand_scores["negbinomial"] = min(cand_scores.get("negbinomial", 1.0), 0.15)

        if (n_plus_x + n_plus_mx) >= 1 and (n_minus_x + n_minus_mx) >= 2:
            cand_scores["betabinomial"] = min(cand_scores.get("betabinomial", 1.0), 0.20)

        def _closest_to(vs, target, tol=0.25):
            if not vs:
                return None
            b = min(vs, key=lambda x: abs(x - target))
            return b if abs(b - target) <= tol else None

        bx1 = _closest_to(s["minus_x"], 1.0)  # -logΓ(x+1)
        other_minus = [v for v in s["minus_x"] if (bx1 is None) or (abs(v - bx1) > 1e-6)]
        if (bx1 is not None) and (len(s["plus_x"]) >= 2) and (len(other_minus) >= 1) and (n_plus_mx + n_minus_mx == 0):
            cand_scores["betanegbinomial"] = min(cand_scores.get("betanegbinomial", 1.0), 0.12)

        if (n_plus_mx + n_minus_mx) == 0:
            closest_1 = _closest_to(s["minus_x"], 1.0)
            is_near1 = (closest_1 is not None)
            if (len(s["plus_x"]) >= 1) and (len(s["minus_x"]) >= 1) and (not is_near1):
                cand_scores["yulesimon"] = min(cand_scores.get("yulesimon", 1.0), 0.18)

        if "neghypergeometric" in cand_scores:
            cand_scores.pop("hypergeometric", None)
            cand_scores.pop("binomial", None)
        elif "hypergeometric" in cand_scores:
            cand_scores.pop("binomial", None)

        if (n_plus_mx + n_minus_mx) > 0:
            cand_scores.pop("betanegbinomial", None)
        else:
            cand_scores.pop("betabinomial", None)

        if strict_struct_filter:
            for f in list(NON_LOGGAMMA_FAMS):
                cand_scores.pop(f, None)

        if not cand_scores:
            return []

        items = sorted(cand_scores.items(), key=lambda t: t[1])
        top_n = int(cfg_dict.get("gate_top_n", 3))
        selected = [fam for fam, _ in items[:top_n]]

        def _ensure(name):
            if name in cand_scores and name not in selected:
                selected.append(name)
        for name in ["neghypergeometric", "hypergeometric", "binomial",
                     "negbinomial", "betabinomial", "betanegbinomial", "yulesimon"]:
            _ensure(name)

        if strict_struct_filter:
            selected = [f for f in selected if f not in NON_LOGGAMMA_FAMS]

        if debug_gate:
            print(f"[gate-A] selected={selected} (gamma-structure)")

        return selected

    cand_scores = {}
    Kp = int(cfg_dict.get("grid_pos", 150))
    loc_min = int(cfg_dict.get("loc_min", -5))
    loc_max = int(cfg_dict.get("loc_max", 5))

    k = np.arange(1, Kp + 1, dtype=float)
    y = _safe_eval_row(row, k)
    if y is None or np.isfinite(y).sum() < max(5, Kp // 6):
        return []

    if _has_abs_like(eqs) and strict_mutual_exclusion:
        if debug_gate:
            print("[gate-B] hard choose dlaplace due to abs-like pattern in equation")
        return ["dlaplace"]

    m = np.isfinite(y)

    Xz = np.vstack([np.ones_like(k[m]), np.log(k[m])]).T
    mse_zipf, beta_z = _lin_mse(y[m], Xz)
    if beta_z is not None and beta_z[1] >= 0:
        mse_zipf *= 3.0
    cand_scores["zipf"] = mse_zipf
    cand_scores["zipfian"] = mse_zipf * 1.05

    yls = y[m] + np.log(k[m])
    Xl = np.vstack([np.ones_like(k[m]), k[m]]).T
    mse_logser, beta_l = _lin_mse(yls, Xl)
    if beta_l is not None and beta_l[1] >= 0:
        mse_logser *= 2.0
    cand_scores["logseries"] = mse_logser

    mse_geo, beta_g = _lin_mse(y[m], Xl)
    if beta_g is not None and beta_g[1] >= 0:
        mse_geo *= 2.0
    cand_scores["geometric"] = mse_geo

    mse_dl_best, loc_best, beta_dl = math.inf, None, None
    for loc in range(loc_min, loc_max + 1):
        d = np.abs(k[m] - loc)
        Xd = np.vstack([np.ones_like(d), d]).T
        mse_dl, beta_d = _lin_mse(y[m], Xd)
        if mse_dl < mse_dl_best:
            mse_dl_best, loc_best, beta_dl = mse_dl, loc, beta_d
    if beta_dl is not None and beta_dl[1] >= 0:
        mse_dl_best *= 3.0
    cand_scores["dlaplace"] = mse_dl_best

    cand_scores["boltzmann"] = mse_geo * 1.02
    finite_ratio = np.isfinite(y).mean()
    if finite_ratio < 0.9:
        cand_scores["boltzmann"] *= 0.8

    s_eq = eqs.lower()
    if "log(x0" in s_eq or " log(x0" in s_eq:
        cand_scores["zipf"] *= 0.85
        cand_scores["logseries"] *= 0.9

    others = ["zipf", "zipfian", "logseries", "geometric", "boltzmann"]
    other_mins = [cand_scores[f] for f in others if f in cand_scores]
    if strict_mutual_exclusion and other_mins:
        best_other = min(other_mins)

        if prefer_boltz_no_abs and not _has_abs_like(eqs):
            if cand_scores["dlaplace"] <= dl_noabs_override_margin * best_other:
                if debug_gate:
                    print(f"[gate-B] no-abs override → choose boltzmann "
                          f"(MSE_dl={cand_scores['dlaplace']:.3g}, best_other={best_other:.3g})")
                return ["boltzmann"]

        if cand_scores["dlaplace"] <= dl_margin * best_other:
            if debug_gate:
                print(f"[gate-B] choose dlaplace (loc≈{loc_best}) | "
                      f"MSE_dl={cand_scores['dlaplace']:.3g}, best_other={best_other:.3g}")
            return ["dlaplace"]

    items = sorted(cand_scores.items(), key=lambda t: t[1])
    top_n = int(cfg_dict.get("gate_top_n", 3))
    selected = [fam for fam, _ in items[:top_n]]

    if debug_gate:
        dbg = {k: (None if not np.isfinite(v) else round(float(v), 6)) for k, v in cand_scores.items()}
        print("[gate-B] non-gamma linear-MSE:", dbg, f"→ selected={selected}")

    return selected
