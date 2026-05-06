import os, argparse, sys, shutil, json, pickle, ast
from pathlib import Path
import numpy as np
import pandas as pd
from pysr import PySRRegressor

from datasets import get_dataset_maker, list_datasets
from ops.primitive_ops import BASE_SYMPY_MAP
from configs.universal_ops import get_combined_ops
from configs.complexity_profiles import COMPLEXITY_PROFILES, resolve_profile
from configs.operator_sets import (
    LOSS_THRESH_PER_SET as LOSS_THRESH,
    SUM_TOL, LOGP_MAX_TOL, OP_LIMITS,
)
from utils.eval import pmf_checks, passes_op_limits

EPS = 1e-300


def _split_kv_items(s: str):
    items = []
    start = 0
    depth = 0
    quote = None
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            items.append(s[start:i])
            start = i + 1
    items.append(s[start:])
    return items


def parse_kv(s: str):
    out = {}
    if not s:
        return out
    for part in _split_kv_items(s):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad param '{part}', expect key=value")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)):
                    out[k] = list(parsed)
                elif isinstance(parsed, (int, float, str)):
                    out[k] = parsed
                else:
                    out[k] = v
            except Exception:
                try:
                    if "." in v or "e" in v.lower():
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                except Exception:
                    out[k] = v
    return out


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, x):
        self.stdout.write(x)
        self.file.write(x)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def stable_log_softmax(logv: np.ndarray) -> np.ndarray:
    m = float(np.max(logv))
    logZ = m + np.log(np.sum(np.exp(logv - m)))
    return logv - logZ


def linear_calibrated_rmse(y_true, y_pred, w=None):
    if w is None:
        A = np.vstack([y_pred, np.ones_like(y_pred)]).T
        (a, b), *_ = np.linalg.lstsq(A, y_true, rcond=None)
        y_cal = a * y_pred + b
        rmse = float(np.sqrt(np.mean((y_cal - y_true) ** 2)))
        return rmse, float(a), float(b)
    else:
        W = np.sqrt(w)
        A = np.vstack([y_pred, np.ones_like(y_pred)]).T * W[:, None]
        Y = y_true * W
        (a, b), *_ = np.linalg.lstsq(A, Y, rcond=None)
        y_cal = a * y_pred + b
        rmse = float(np.sqrt(np.average((y_cal - y_true) ** 2, weights=w)))
        return rmse, float(a), float(b)


def evaluate_equations(model, X, y_target, w=None, save_csv_path=None, top_k=15):
    rows = []
    for row in model.equations_.itertuples():
        eq = row.equation
        try:
            y_hat = row.lambda_format(X).reshape(-1)
            if y_hat.shape != y_target.shape:
                raise ValueError("shape mismatch")
            diff = y_hat - y_target
            if w is None:
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
                max_abs = float(np.max(np.abs(diff)))
            else:
                rmse = float(np.sqrt(np.average(diff ** 2, weights=w)))
                mae = float(np.average(np.abs(diff), weights=w))
                max_abs = float(np.max(np.abs(diff)))
            cal_rmse, a_fit, b_fit = linear_calibrated_rmse(y_target, y_hat, w=w)
        except Exception:
            rmse = mae = max_abs = cal_rmse = np.inf
            a_fit = b_fit = np.nan
        rows.append({
            "equation": eq,
            "pysr_loss": float(row.loss),
            "complexity": int(row.complexity),
            "rmse": rmse,
            "mae": mae,
            "max_abs": max_abs,
            "cal_rmse": cal_rmse,
            "cal_a": a_fit,
            "cal_b": b_fit,
        })
    df = pd.DataFrame(rows).sort_values(["rmse", "pysr_loss", "complexity"])
    if save_csv_path:
        df.to_csv(save_csv_path, index=False, float_format="%.16g")
        print(f"[Eval] Saved metrics → {save_csv_path} (n={len(df)})")
    print(f"\n[Eval] Top {top_k} by rmse:")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 160):
        print(df.head(top_k))
    return df


def _var_names_for_X(X: np.ndarray):
    X = np.asarray(X)
    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        return ["x"]
    assert X.ndim == 2, f"X must be 1D/2D, got shape={X.shape}"
    d = X.shape[1]
    return [f"x{i+1}" for i in range(d)]


def _flatten_coord_cols(arr: np.ndarray, base_name="x"):
    a = np.asarray(arr)
    if a.ndim == 1:
        return {base_name: a.reshape(-1)}
    elif a.ndim == 2:
        return {f"{base_name}{i+1}": a[:, i] for i in range(a.shape[1])}
    else:
        raise ValueError(f"{base_name} must be 1D/2D, got shape={a.shape}")


def _validate_positive_finite(name: str, x: float):
    if (not np.isfinite(x)) or (x <= 0.0):
        raise ValueError(f"--{name} must be finite and > 0, got {x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=str, default="hypergeo",
                    help=f"dataset name, one of: {', '.join(list_datasets())}")
    ap.add_argument("--params", type=str, default="",
                    help="dataset params, e.g. 'N_total=200,K_good=100,n_draw=100'")
    ap.add_argument("--julia", type=str, default=None,
                    help="Path to Julia binary. Required if 'julia' is not on PATH.")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--runs_dir", type=str, default="runs")

    ap.add_argument("--profiles", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)

    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--M", type=int, default=50_000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--exp_thr", type=float, default=4.0)
    ap.add_argument("--use_weights", action="store_true")

    ap.add_argument("--weight_gamma", type=float, default=1.0,
                    help="Weight damping exponent gamma: w_final = (w_raw / median_fit(w_raw)) ** gamma")
    ap.add_argument("--weight_median_norm", action="store_true", default=True,
                    help="Enable median normalization for weights over X_fit.")
    ap.add_argument("--no_weight_median_norm", action="store_false", dest="weight_median_norm",
                    help="Disable median normalization for weights.")

    ap.add_argument("--pmf_check", type=str, default="soft",
                    choices=["off", "soft", "hard"])
    ap.add_argument("--rmse_target", type=str, default="auto",
                    choices=["auto", "true_raw", "noisy"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save_sampling_logtrue", action="store_true")

    ap.add_argument(
        "--loss_thresh",
        type=float,
        default=None,
        help="Override acceptance loss threshold (global). If set, overrides profile loss_thresh and LOSS_THRESH_PER_SET."
    )

    ap.add_argument("--extension", action="store_true",
                    help="Enable extension operators: logaddexp + logdelta0 (for ZI/mixture distributions).")
    ap.add_argument("--select_candidate", choices=["rmse", "loss", "complexity", "complexity_then_rmse"],
                    default="loss",
                    help="How to rank winner candidates: rmse (default), loss, complexity, or complexity_then_rmse.")
    ap.add_argument("--populations", type=int, default=30,
                    help="Number of populations in PySR.")
    ap.add_argument("--population_size", type=int, default=60,
                    help="Size of each population in PySR.")
    ap.add_argument("--maxsize", type=int, default=15,
                    help="PySR maxsize (max total complexity of an equation).")
    ap.add_argument("--expensive_ratio", type=float, default=0.5,
                    help="Cost of expensive operators in each profile = round(maxsize * expensive_ratio); "
                         "cheap operators are always 1. Smaller ratio = looser (expensive ops appear more easily).")

    args = ap.parse_args()

    if args.loss_thresh is not None:
        _validate_positive_finite("loss_thresh", float(args.loss_thresh))

    if args.noisy and args.use_weights:
        _validate_positive_finite("weight_gamma", float(args.weight_gamma))

    if args.seed is not None:
        np.random.seed(int(args.seed))

    if args.julia is not None:
        assert os.path.isfile(args.julia), f"Julia not found at {args.julia}"
        os.environ["JULIA_CMD"] = args.julia
        os.environ["PATH"] = os.path.dirname(args.julia) + ":" + os.environ["PATH"]
    assert shutil.which("julia") is not None, "Julia not found. Install Julia 1.10.2 and ensure it is on PATH, or pass --julia /path/to/julia."

    maker = get_dataset_maker(args.dist)
    ds_params = parse_kv(args.params)
    X, y_true_formula, support = maker(**ds_params)
    y_true_raw = np.asarray(y_true_formula, dtype=float).reshape(-1).copy()

    var_names = _var_names_for_X(X)

    print(f"[Data] dist={args.dist}, params={ds_params}, points={len(np.asarray(support))}  std(logtrue)={np.std(y_true_raw):.3e}")
    print(f"[Data] X shape={np.asarray(X).shape}, support shape={np.asarray(support).shape}, variables={var_names}")

    run_root = os.path.join(args.runs_dir, f"{args.dist}")
    os.makedirs(run_root, exist_ok=True)

    master_log = Tee(os.path.join(run_root, "sweep_log.txt"))
    sys.stdout = master_log

    logp_for_sampling = stable_log_softmax(y_true_raw)
    p_true_for_sampling = np.exp(logp_for_sampling)

    truth_csv = os.path.join(run_root, f"{args.dist}_truth.csv")
    truth_cols = {}
    truth_cols.update(_flatten_coord_cols(support, base_name="x"))
    truth_cols["logpmf_true"] = y_true_raw
    truth_cols["p_true_for_sampling"] = p_true_for_sampling
    if args.save_sampling_logtrue:
        truth_cols["logpmf_true_for_sampling"] = logp_for_sampling
    pd.DataFrame(truth_cols).to_csv(truth_csv, index=False, float_format="%.16g")
    print(f"[Saved] truth table → {truth_csv}")

    use_mask = np.ones_like(y_true_raw, dtype=bool)
    y_target = y_true_raw.copy()
    weights = None
    weight_raw = None
    weight_norm = None
    weight_med = None

    if args.noisy:
        K = len(y_true_raw)
        counts = np.random.multinomial(args.M, p_true_for_sampling)
        p_hat = (counts + args.alpha) / (args.M + args.alpha * K)
        logphat = np.log(np.maximum(p_hat, EPS))
        expected = args.M * p_true_for_sampling
        use_mask = (expected >= args.exp_thr)
        kept = int(use_mask.sum())
        print(f"[Noisy] M={args.M:,}, alpha={args.alpha}, exp_thr={args.exp_thr} → kept {kept}/{K} points")

        var_logphat = (1.0 - p_hat) / np.maximum(args.M * p_hat, EPS)

        if args.use_weights:
            gamma = float(args.weight_gamma)
            weight_raw = 1.0 / np.maximum(var_logphat, EPS)

            if args.weight_median_norm:
                wr_fit = weight_raw[use_mask]
                wr_fit = wr_fit[np.isfinite(wr_fit)]
                if wr_fit.size == 0:
                    raise RuntimeError("No finite weights on X_fit to compute median normalization.")
                weight_med = float(np.median(wr_fit))
                if (not np.isfinite(weight_med)) or (weight_med <= 0.0):
                    raise RuntimeError(f"Invalid median weight on X_fit: {weight_med}")
                weight_norm = weight_raw / weight_med
            else:
                weight_med = np.nan
                weight_norm = weight_raw.copy()

            weights = np.power(weight_norm, gamma) if gamma != 1.0 else weight_norm.copy()

            msg = "[Noisy] weights enabled: "
            if args.weight_median_norm:
                msg += f"median_norm=ON (median_fit={weight_med:.6g}), "
            else:
                msg += "median_norm=OFF, "
            msg += f"gamma={gamma:g}  -> w_final=(w_raw/median)^gamma"
            print(msg)
        else:
            weights = None
            weight_raw = None
            weight_norm = None
            weight_med = None

        y_target = logphat

        noisy_csv = os.path.join(run_root, f"{args.dist}_noisy_compare.csv")
        noisy_cols = {}
        noisy_cols.update(_flatten_coord_cols(support, base_name="x"))
        noisy_cols.update({
            "logpmf_true": y_true_raw.copy(),
            "p_true_for_sampling": p_true_for_sampling,
            "expected_count_float": expected,
            "count_int": counts.astype(int),
            "p_hat_prob": p_hat,
            "logpmf_noisy": logphat,
            "used_in_fit": use_mask,
        })
        if args.use_weights:
            noisy_cols["weight_raw"] = weight_raw
            noisy_cols["weight_median_fit"] = np.full_like(logphat, float(weight_med) if weight_med is not None else np.nan)
            noisy_cols["weight_norm"] = weight_norm
            noisy_cols["weight_final"] = weights
        else:
            noisy_cols["weight_raw"] = np.full_like(logphat, np.nan)
            noisy_cols["weight_median_fit"] = np.full_like(logphat, np.nan)
            noisy_cols["weight_norm"] = np.full_like(logphat, np.nan)
            noisy_cols["weight_final"] = np.full_like(logphat, np.nan)

        if args.save_sampling_logtrue:
            noisy_cols["logpmf_true_for_sampling"] = logp_for_sampling

        pd.DataFrame(noisy_cols).to_csv(noisy_csv, index=False, float_format="%.16g")
        print(f"[Saved] noisy compare → {noisy_csv}")

        _max_diff = float(np.max(np.abs(np.asarray(noisy_cols["logpmf_true"]) - y_true_raw)))
        assert _max_diff == 0.0, f"logpmf_true changed in noisy table (max |Δ|={_max_diff})"
        _sum_phat = float(np.sum(noisy_cols["p_hat_prob"]))
        assert 0.9999 <= _sum_phat <= 1.0001, f"sum(p_hat_prob)={_sum_phat} (should be ~1)"

        print("[DEBUG] true head:", y_true_raw[:3])
        print("[DEBUG] noisy head (true vs noisy):", y_true_raw[:3], logphat[:3])

    _unary, _binary, _nested, _constraints = get_combined_ops(extension=args.extension)
    COMMON = dict(
        niterations=args.iters,
        should_optimize_constants=True,
        optimizer_nrestarts=8,
        maxsize=args.maxsize,
        populations=args.populations,
        population_size=args.population_size,
        procs=0,
        verbosity=1,
        variable_names=var_names,
        use_frequency_in_tournament=False,
        extra_sympy_mappings=BASE_SYMPY_MAP,
        nested_constraints=_nested,
        unary_operators=_unary,
        binary_operators=_binary,
        constraints=_constraints,
    )

    if args.profiles is not None:
        names = [s.strip() for s in args.profiles.split(",") if s.strip()]
    elif args.profile is not None:
        names = [args.profile.strip()]
    else:
        names = []

    if names:
        name_set = {p["name"] for p in COMPLEXITY_PROFILES}
        unknown = [n for n in names if n not in name_set]
        if unknown:
            raise AssertionError(
                f"Unknown profile(s): {unknown}. Available: {sorted(name_set)}"
            )
        PROFILES = [p for p in COMPLEXITY_PROFILES if p["name"] in names]
        print(f"[PROFILE] using selected profiles: {names}")
    else:
        PROFILES = COMPLEXITY_PROFILES
        print(f"[PROFILE] using all profiles: {[p['name'] for p in PROFILES]}")

    X = np.asarray(X)
    X_fit = X[use_mask]
    y_fit = y_target[use_mask]
    w_fit = (weights[use_mask] if (weights is not None) else None)

    candidates = []
    models_by_dir = {}
    fallback_best = None

    for prof in PROFILES:
        name = prof["name"]
        model_dir = os.path.join(run_root, name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n=== [{name}] training (ops same, complexity profile custom) ===")

        if args.loss_thresh is not None:
            loss_thresh_used = float(args.loss_thresh)
        else:
            loss_thresh_used = float(prof.get("loss_thresh", LOSS_THRESH))

        op_costs = resolve_profile(prof, args.maxsize, args.expensive_ratio, extension=args.extension)
        print(f"[{name}] maxsize={args.maxsize}, expensive_ratio={args.expensive_ratio:g} "
              f"-> op_costs={op_costs}")

        model = PySRRegressor(
            complexity_of_operators=op_costs,
            model_selection="pareto",
            **COMMON,
        )

        if w_fit is None:
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_fit, y_fit, weights=w_fit)

        models_by_dir[str(model_dir)] = model

        eqs = model.equations_
        eqs_path = os.path.join(model_dir, "equations_scored.csv")
        eqs.to_csv(eqs_path, index=False, float_format="%.16g")
        print(f"[{name}] saved: {eqs_path}")

        for ridx, r in enumerate(eqs.itertuples()):
            try:
                rloss = float(r.loss)
            except Exception:
                continue
            rc = int(getattr(r, "complexity", 10**9))
            cand_fb = {
                "profile": name,
                "equation": r.equation,
                "loss": rloss,
                "complexity": rc,
                "checkpoint_dir": model_dir,
                "row_idx": ridx,
            }
            if (fallback_best is None) or ((cand_fb["loss"], cand_fb["complexity"]) < (fallback_best["loss"], fallback_best["complexity"])):
                fallback_best = cand_fb

        picked = None
        for row in eqs.itertuples():
            reasons = []

            if float(row.loss) > loss_thresh_used:
                reasons.append(f"loss {row.loss:.3e} > {loss_thresh_used:.3e}")

            pmf_ok, pmf_stats = True, {"norm_dev": np.nan, "max_logp": np.nan}
            if args.pmf_check != "off":
                pmf_ok, pmf_stats = pmf_checks(row.lambda_format, X, SUM_TOL, LOGP_MAX_TOL)
                if args.pmf_check == "hard" and not pmf_ok:
                    reasons.append(
                        f"pmf_check_fail |sum(exp)-1|={pmf_stats.get('norm_dev'):.2e}, maxlogp={pmf_stats.get('max_logp'):.3e}"
                    )

            cnt = {}
            if not reasons:
                ok_ops, cnt = passes_op_limits(row.equation, OP_LIMITS)
                if not ok_ops:
                    reasons.append(f"op_quota_exceeded {cnt}")

            if reasons:
                print(f"[{name}] reject: {row.equation}  :: {'; '.join(reasons)}")
                continue

            y_hat_all = row.lambda_format(X).reshape(-1)

            if args.rmse_target == "auto":
                y_ref = (y_target if args.noisy else y_true_raw)
                w_ref = (weights if (args.noisy and args.use_weights) else None)
            elif args.rmse_target == "true_raw":
                y_ref = y_true_raw
                w_ref = None
            else:
                if not args.noisy:
                    y_ref = y_true_raw
                    w_ref = None
                else:
                    y_ref = y_target
                    w_ref = (weights if args.use_weights else None)

            if w_ref is None:
                rmse = float(np.sqrt(np.mean((y_hat_all - y_ref) ** 2)))
            else:
                rmse = float(np.sqrt(np.average((y_hat_all - y_ref) ** 2, weights=w_ref)))

            picked = {
                "profile": name,
                "equation": row.equation,
                "loss": float(row.loss),
                "rmse": rmse,
                "complexity": int(row.complexity),
                "norm_dev": pmf_stats.get("norm_dev"),
                "max_logp": pmf_stats.get("max_logp"),
                "op_count": cnt if isinstance(cnt, dict) else {},
                "checkpoint_dir": model_dir,
                "loss_thresh_used": float(loss_thresh_used),
            }
            print(f"[{name}] ✓ pick: loss={picked['loss']:.3e}, rmse={picked['rmse']:.3e}, complexity={picked['complexity']}, loss_th={loss_thresh_used:.3e}")
            break

        metrics_csv = os.path.join(model_dir, "equation_metrics.csv")
        _ = evaluate_equations(model, X_fit, y_fit, w=w_fit, save_csv_path=metrics_csv, top_k=15)

        if picked is None:
            print(f"[{name}] ✗ no candidate under constraints (loss_th={loss_thresh_used:.3e})")
        else:
            candidates.append(picked)

    if not candidates:
        if fallback_best is None:
            print("\n[Result] No candidate passed, and no fallback equation found.")
            sys.stdout = sys.__stdout__
            master_log.close()
            return

        print("\n[Result] No candidate passed constraints; using fallback (min loss over all equations).")
        winner = pd.Series({
            "profile": fallback_best["profile"],
            "equation": fallback_best["equation"],
            "loss": float(fallback_best["loss"]),
            "rmse": np.nan,
            "complexity": int(fallback_best["complexity"]),
            "norm_dev": np.nan,
            "max_logp": np.nan,
            "op_count": {},
            "checkpoint_dir": fallback_best["checkpoint_dir"],
            "loss_thresh_used": float(args.loss_thresh) if args.loss_thresh is not None else float(LOSS_THRESH),
        })
        pass_constraints = False

        out_csv = os.path.join(run_root, f"{args.dist}_candidates_summary.csv")
        pd.DataFrame([winner.to_dict()]).to_csv(out_csv, index=False, float_format="%.16g")
        print(f"[Saved] {out_csv}")
    else:
        sel = args.select_candidate
        if sel == "complexity_then_rmse":
            dfc_tmp = pd.DataFrame(candidates)
            dfc_tmp["complexity_round"] = dfc_tmp["complexity"].round().astype(int)
            dfc = dfc_tmp.sort_values(["complexity_round", "rmse", "loss"]).drop(columns=["complexity_round"])
        else:
            sort_keys = {"rmse": ["rmse", "loss", "complexity"],
                         "loss": ["loss", "rmse", "complexity"],
                         "complexity": ["complexity", "rmse", "loss"]}[sel]
            dfc = pd.DataFrame(candidates).sort_values(sort_keys)
        winner = dfc.iloc[0]
        pass_constraints = True

        out_csv = os.path.join(run_root, f"{args.dist}_candidates_summary.csv")
        dfc.to_csv(out_csv, index=False, float_format="%.16g")
        print(f"[Saved] {out_csv}")

    print("\n================ [WINNER] ================")
    print(f"Dist         : {args.dist}  params={ds_params}")
    print(f"Noisy?       : {args.noisy}  M={args.M if args.noisy else '-'}  alpha={args.alpha if args.noisy else '-'}  "
          f"exp_thr={args.exp_thr if args.noisy else '-'}  seed={args.seed if args.seed is not None else '-'}  "
          f"use_weights={args.use_weights}")
    if args.noisy and args.use_weights:
        print(f"Weight shrink: median_norm={bool(args.weight_median_norm)}  gamma={float(args.weight_gamma):g}")

    print(f"Pass constraints: {pass_constraints}")
    print(f"Profile      : {winner['profile']}")
    print(f"Complexity   : {winner['complexity']}")
    print(f"Equation     : {winner['equation']}")
    print(f"Loss         : {float(winner['loss']):.3e}")
    print(f"Loss_th_used : {float(winner.get('loss_thresh_used', np.nan)):.3e}" if pd.notna(winner.get("loss_thresh_used", np.nan)) else "Loss_th_used : nan")

    try:
        rmse_val = float(winner["rmse"])
    except Exception:
        rmse_val = np.nan
    print(f"RMSE(target) : {rmse_val:.3e}" if np.isfinite(rmse_val) else "RMSE(target) : nan")
    print(f"|sum(exp)-1| : {float(winner.get('norm_dev', np.nan)):.2e}" if pd.notna(winner.get("norm_dev", np.nan)) else "|sum(exp)-1| : nan")
    print(f"max(log p)   : {float(winner.get('max_logp', np.nan)):.3e}" if pd.notna(winner.get("max_logp", np.nan)) else "max(log p)   : nan")
    print(f"Op counts    : {winner['op_count'] if 'op_count' in winner else {}}")

    winner_dir = str(winner["checkpoint_dir"])
    winner_model = models_by_dir.get(winner_dir, None)
    if winner_model is None:
        raise RuntimeError(
            f"Cannot find winner model object for winner_dir={winner_dir}. "
            f"Ensure models_by_dir[model_dir]=model was executed."
        )

    dist_root = Path(run_root)
    ckpt_path = dist_root / "winner_model.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(winner_model, f)

    handoff = {
        "dist": args.dist,
        "params": ds_params,
        "winner_dir": winner_dir,
        "winner_ckpt": str(ckpt_path),
        "winner_profile": str(winner["profile"]),
        "winner_equation": str(winner["equation"]),
        "winner_complexity": int(winner["complexity"]),
        "winner_rmse": float(winner["rmse"]) if pd.notna(winner.get("rmse", np.nan)) else None,
        "winner_loss": float(winner["loss"]),
        "loss_thresh_used": float(winner.get("loss_thresh_used", np.nan)) if pd.notna(winner.get("loss_thresh_used", np.nan)) else None,
        "pass_constraints": bool(pass_constraints),
        "noisy": bool(args.noisy),
        "M": int(args.M) if args.noisy else None,
        "alpha": float(args.alpha) if args.noisy else None,
        "exp_thr": float(args.exp_thr) if args.noisy else None,
        "use_weights": bool(args.use_weights) if args.noisy else None,
        "weight_gamma": float(args.weight_gamma) if (args.noisy and args.use_weights) else None,
        "weight_median_norm": bool(args.weight_median_norm) if (args.noisy and args.use_weights) else None,
        "seed": int(args.seed) if args.seed is not None else None,
    }
    handoff_path = dist_root / "handoff_inference.json"
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    print(f"[Handoff] winner_model.pkl -> {ckpt_path}")
    print(f"[Handoff] handoff_inference.json -> {handoff_path}")

    sys.stdout = sys.__stdout__
    master_log.close()


if __name__ == "__main__":
    main()
