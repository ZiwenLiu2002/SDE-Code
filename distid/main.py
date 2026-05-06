import argparse, pickle, json
import numpy as np
import pandas as pd
from typing import Dict, List
from .types import Recognizer
from .structures import extract_structure_flags, expand_to_loggamma
from .structmatch import analyze_loggamma_terms
from .gate import gate_families
from .decision import discriminate_with_tail_and_skl
from .families import REGISTRY as FAMILY_REG
from .utils_expand import print_expansion_report

DISALLOW_IF_GAMMA = {"zipf", "zipfian", "logseries", "geometric", "dlaplace", "boltzmann"}
EXTENSION_FAMILIES = {"zinb", "zig", "mixbinom"}


def _to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "equations_") and isinstance(obj.equations_, pd.DataFrame):
        return obj.equations_
    if isinstance(obj, dict):
        for k in ["equations_", "equations", "paretofront", "equations_full"]:
            if k in obj and isinstance(obj[k], pd.DataFrame):
                return obj[k]
    raise RuntimeError("Cannot find equations DataFrame in checkpoint")


def _has_any_loggamma_tokens(eq_str: str) -> bool:
    s = eq_str.lower()
    return any(tok in s for tok in ["gammaln", "loggamma", "logfac", "nlogfac", "logc(", "logb("])

def _has_logaddexp(eq_str: str) -> bool:
    return "logaddexp" in eq_str.lower()


def _apply_struct_filter(fams: List[str], eq_str: str, cfg) -> List[str]:
    if not getattr(cfg, "strict_struct_filter", True):
        return fams
    if _has_any_loggamma_tokens(eq_str):
        return [f for f in fams if f not in DISALLOW_IF_GAMMA]
    return fams


def _rank_key(row, final_score, mode="complexity_then_score", round_complexity=True):
    comp = float(getattr(row, "complexity", 1e9))
    if round_complexity:
        comp = int(round(comp))
    if mode == "score":
        return (final_score,)
    elif mode == "complexity":
        return (comp,)
    else:
        return (comp, final_score)


def evaluate_checkpoint(ckpt_path: str, cfg, recognizers: Dict[str, Recognizer]) -> None:
    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)
    eqs: pd.DataFrame = _to_df(model)

    rows_all = []
    candidates_global = []  # list of (rank_tuple, row, decision)

    enabled = set(recognizers.keys())

    min_keep = max(1, int(getattr(cfg, "min_keep", 1)))
    top_k = max(1, int(getattr(cfg, "top_k", 1)))

    all_rows = list(eqs.itertuples())
    n_total = len(all_rows)
    passing_rows = [r for r in all_rows if float(r.loss) <= cfg.loss_th]
    n_pass = len(passing_rows)

    if n_pass >= min_keep:
        rows_to_process = passing_rows
        promoted = 0
    else:
        rows_sorted = sorted(all_rows, key=lambda r: float(r.loss))
        rows_to_process = rows_sorted[: min(min_keep, n_total)]
        promoted = max(0, len(rows_to_process) - n_pass)
        print(f"[Info] {n_pass}/{n_total} eqs pass loss_th={cfg.loss_th:.3e}; "
              f"promoting {promoted} more by lowest loss to reach min_keep={min_keep} "
              f"(actual entering inference: {len(rows_to_process)}).")

    n_with_fit = 0

    for row in rows_to_process:
        eq_str = str(row.equation)
        _ = extract_structure_flags(eq_str)

        if getattr(cfg, "print_expansion", False):
            try:
                exp = expand_to_loggamma(eq_str)
                const_val = float(exp["const_part"].evalf())
                print("\n[Expand→loggamma] equation:", eq_str)
                print("  var_part(x):", exp["var_part"])
                print("  const_part :", exp["const_part"], f"≈ {const_val:.12g}")
                s_counts_dbg = analyze_loggamma_terms(eq_str)
                print("  plus_x=", len(s_counts_dbg["plus_x"]),
                      ", minus_x=", len(s_counts_dbg["minus_x"]),
                      ", plus_mx=", len(s_counts_dbg["plus_mx"]),
                      ", minus_mx=", len(s_counts_dbg["minus_mx"]), sep="")
            except Exception as e:
                print("\n[Expand→loggamma FAILED]", e)

        s_counts = analyze_loggamma_terms(eq_str)
        n_plus_x, n_minus_x = len(s_counts["plus_x"]), len(s_counts["minus_x"])
        n_plus_mx, n_minus_mx = len(s_counts["plus_mx"]), len(s_counts["minus_mx"])
        has_minus_side = (n_plus_mx + n_minus_mx) > 0

        if getattr(cfg, "print_struct_counts", False):
            print(f"[struct] plus_x={n_plus_x}, minus_x={n_minus_x}, plus_mx={n_plus_mx}, minus_mx={n_minus_mx}")

        gated = gate_families(row, vars(cfg)) or []

        has_gamma_tokens = _has_any_loggamma_tokens(eq_str)
        has_logaddexp = _has_logaddexp(eq_str)
        extension_mode = getattr(cfg, "extension", False)

        if extension_mode and has_logaddexp:
            if has_gamma_tokens:
                # logaddexp + logfac/logC → likely ZINB or mixbinom
                ext_allow = {"zinb", "mixbinom"}
            else:
                # logaddexp without logfac → likely ZIG
                ext_allow = {"zig", "zinb"}
            ext_enabled = ext_allow & enabled
            if ext_enabled:
                fam_list = list(ext_enabled)
            else:
                fam_list = []
        elif has_gamma_tokens:
            if has_minus_side:
                allow = {"binomial", "betabinomial", "hypergeometric", "neghypergeometric"}
            else:
                allow = {"poisson", "negbinomial", "betanegbinomial", "yulesimon"}
            fam_list = [f for f in gated if (f in allow) and (f in enabled)]
            if not fam_list:
                fam_list = [f for f in allow if f in enabled]
        else:
            allow = {"zipf", "zipfian", "logseries", "geometric", "dlaplace", "poisson", "boltzmann"}
            fam_list = [f for f in gated if (f in allow) and (f in enabled)]
            if not fam_list:
                fam_list = [f for f in allow if f in enabled]

        fam_list_before = list(fam_list)
        fam_list = _apply_struct_filter(fam_list, eq_str, cfg)
        if getattr(cfg, "strict_struct_filter", True):
            removed = set(fam_list_before) - set(fam_list)
            if removed:
                print(f"[gate] loggamma detected → removed non-loggamma families: {sorted(removed)}")

        def run_eval(fams: List[str]):
            out = []
            for fam in fams:
                recog = recognizers[fam]
                try:
                    fr = recog(row, vars(cfg))
                except Exception:
                    continue
                if fr is None or (not fr.valid):
                    continue
                out.append(fr)
                rows_all.append({
                    "equation": row.equation,
                    "complexity": row.complexity,
                    "loss": float(row.loss),
                    "dist": fr.name,
                    "params": json.dumps(fr.params),
                    "rmse": fr.rmse,
                    "mass_err": fr.mass_err,
                    "score": fr.score,
                    "stage": 1,
                })
            return out

        fitresults = run_eval(fam_list)
        if not fitresults:
            continue
        n_with_fit += 1

        if len(fitresults) == 1:
            fr = fitresults[0]
            decision = {
                "winner": {"family": fr.name, "params": fr.params, "score": fr.score,
                           "skl": np.nan, "tail_err": np.nan, "rmse_fit": fr.rmse},
                "second": None,
                "ambiguous": False
            }
        else:
            k_fit = np.arange(1, cfg.grid_pos + 1, dtype=float)
            K_ext = max(int(round(cfg.grid_pos * cfg.discr_ext_multiplier)), cfg.grid_pos + 10)
            k_ext = np.arange(1, K_ext + 1, dtype=float)
            fams = [fr.name for fr in fitresults]
            pars = [fr.params for fr in fitresults]
            decision = discriminate_with_tail_and_skl(
                row_best=row, family_candidates=fams, params_list=pars,
                k_fit=k_fit, k_ext=k_ext,
                tail_start_frac=cfg.discr_tail_start_frac,
                w_fit_rmse=cfg.discr_w_rmse, w_skl=cfg.discr_w_skl,
                w_tail=cfg.discr_w_tail, w_taillim=cfg.discr_w_taillim
            )
            if decision["winner"] is None:
                fr = min(fitresults, key=lambda x: x.score)
                decision = {"winner": {"family": fr.name, "params": fr.params, "score": fr.score,
                                       "skl": np.nan, "tail_err": np.nan, "rmse_fit": fr.rmse},
                            "second": None, "ambiguous": True}

        final_score = decision["winner"]["score"]
        rank_tuple = _rank_key(
            row,
            final_score,
            mode=getattr(cfg, "select_equation", "complexity_then_score"),
            round_complexity=getattr(cfg, "complexity_round", True),
        )
        candidates_global.append((rank_tuple, row, decision))

    print(f"[Info] entered={len(rows_to_process)}/{n_total}, "
          f"with_valid_fit={n_with_fit}, candidates_total={len(candidates_global)}")

    if cfg.save_csv and rows_all:
        pd.DataFrame(rows_all).sort_values(["score", "complexity", "loss"]).to_csv(cfg.save_csv, index=False)
        print(f"[Info] saved metrics -> {cfg.save_csv}")

    if not candidates_global:
        print("[Result] No valid equation/distribution found.")
        return

    candidates_global.sort(key=lambda c: c[0])
    top_candidates = candidates_global[:top_k]

    sel_mode = getattr(cfg, "select_equation", "complexity_then_score")
    print(f"\n[Top {len(top_candidates)} Candidates] (selection mode: {sel_mode})")

    for rank, (_, best_row, dec) in enumerate(top_candidates, start=1):
        comp_used = int(round(float(best_row.complexity))) if getattr(cfg, "complexity_round", True) \
            else float(best_row.complexity)

        print(f"\n=== Candidate #{rank} ===")
        print(f"equation: {best_row.equation}")
        print(f"complexity={best_row.complexity} (used={comp_used}), loss={float(best_row.loss):.3e}")

        w = dec["winner"]
        print(f"[Identified] type: {w['family']}")
        for k, v in w["params"].items():
            print(f"  {k}: {float(v):.8g}")
        print(f"[Scores] final={w['score']:.6e}, skl={w['skl']:.6e}, "
              f"tail_err={w['tail_err']:.6e}, rmse_fit={w['rmse_fit']:.6e}")

        if dec["second"] is not None:
            s = dec["second"]
            print(f"[Runner-up] type: {s['family']}, "
                  f"params: {{{', '.join(f'{k}={float(v):.6g}' for k, v in s['params'].items())}}}, "
                  f"score={s['score']:.6e}")

        if dec["ambiguous"]:
            print("[Note] Result is ambiguous (scores are very close).")

    if getattr(cfg, "print_expansion", False):
        _, best_row, _ = top_candidates[0]
        k_debug = np.arange(0, cfg.grid_poisson + 1, dtype=float)
        try:
            y_expr = best_row.lambda_format(k_debug.reshape(-1, 1))
        except Exception:
            y_expr = None
        if y_expr is None or np.isfinite(y_expr).sum() < 3:
            k_debug = np.arange(1, cfg.grid_pos + 1, dtype=float)
            y_expr = best_row.lambda_format(k_debug.reshape(-1, 1))
        print_expansion_report(str(best_row.equation), k_debug, y_expr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--grid_pos", type=int, default=150)
    ap.add_argument("--grid_poisson", type=int, default=150)
    ap.add_argument("--grid_signed", type=int, default=60)
    ap.add_argument("--loss_th", type=float, default=1e-3,
                    help="Soft loss threshold; if fewer than --min_keep equations pass, "
                         "the lowest-loss ones are promoted to fill up to min_keep.")
    ap.add_argument("--min_keep", type=int, default=5,
                    help="Minimum number of equations to enter the recognition pipeline, "
                         "regardless of loss_th. Use 1 for the old hard-filter behavior.")
    ap.add_argument("--top_k", type=int, default=3,
                    help="Number of top equation+distribution candidates to print at the end.")
    ap.add_argument("--loc_min", type=int, default=-5)
    ap.add_argument("--loc_max", type=int, default=5)
    ap.add_argument("--binom_n_cap", type=int, default=100000)
    ap.add_argument("--save_csv", type=str, default=None)
    ap.add_argument("--discr_ext_multiplier", type=float, default=3.0)
    ap.add_argument("--discr_tail_start_frac", type=float, default=0.6)
    ap.add_argument("--discr_w_rmse", type=float, default=1.0)
    ap.add_argument("--discr_w_skl", type=float, default=4.0)
    ap.add_argument("--discr_w_tail", type=float, default=2.0)
    ap.add_argument("--discr_w_taillim", type=float, default=3.0)
    ap.add_argument("--gate_top_n", type=int, default=3)
    ap.add_argument("--zipf_loc_l1", type=float, default=0.1)
    ap.add_argument(
        "--enable", type=str,
        default="poisson,geometric,logseries,zipf,zipfian,binomial,negbinomial,betabinomial,betanegbinomial,hypergeometric,neghypergeometric,dlaplace,yulesimon,boltzmann",
        help="comma list of families to enable"
    )
    ap.add_argument("--extension", action="store_true",
                    help="Enable extension families (zinb, zig, mixbinom) and route logaddexp equations to them first.")
    ap.add_argument("--mixbinom_max_K", type=int, default=3,
                    help="Max number of components to try for mixture binomial (default 3).")
    ap.add_argument("--mixbinom_K_penalty", type=float, default=0.05,
                    help="Score penalty per extra component in mixture binomial.")
    ap.add_argument("--print_expansion", action="store_true",
                    help="Print loggamma expansion report for candidates and the final winner")
    ap.add_argument("--prefer_structure", action="store_true",
                    help="Prefer structure matching; missing structure incurs a penalty in the score (note: strict structure filter has higher priority)")
    ap.add_argument("--w_struct", type=float, default=5.0,
                    help="Structure penalty weight (matched=0, unmatched=1)")
    ap.add_argument("--strict_struct_filter", dest="strict_struct_filter", action="store_true", default=True,
                    help="If loggamma tokens are detected, hard-remove zipf/logseries/geometric/dlaplace",)
    ap.add_argument("--no_strict_struct_filter", dest="strict_struct_filter", action="store_false",
                    help="Disable strict structure filter",)
    ap.add_argument("--strict_mutual_exclusion", dest="strict_mutual_exclusion", action="store_true", default=True,
                    help="Enable mutual-exclusion decisions in gate (e.g., NHG>HG>Binom, BB vs BNB, etc.)",)
    ap.add_argument("--no_strict_mutual_exclusion", dest="strict_mutual_exclusion", action="store_false",
                    help="Disable strict mutual-exclusion decisions in gate",)
    ap.add_argument("--dlaplace_margin", type=float, default=0.65,
                    help="Dlaplace mutual-exclusion threshold: linear MSE must beat other candidates by this ratio (smaller is stricter)")
    ap.add_argument("--select_equation",
                    choices=["score", "complexity", "complexity_then_score"],
                    default="score",
                    help="Final selection rule: by score / by complexity / complexity then score (default)")
    ap.add_argument("--complexity_round", action="store_true", default=True,
                    help="Compare using integer complexity (default on); disable to use raw float complexity")

    args = ap.parse_args()

    cfg = args
    enabled_list = [s.strip().lower() for s in args.enable.split(",") if s.strip()]
    if args.extension:
        for fam in ("zinb", "zig", "mixbinom"):
            if fam not in enabled_list:
                enabled_list.append(fam)
    enabled = set(enabled_list)
    recognizers = {k: v for k, v in FAMILY_REG.items() if k in enabled}

    evaluate_checkpoint(args.ckpt, cfg, recognizers)


if __name__ == "__main__":
    main()
