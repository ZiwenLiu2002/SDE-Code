import argparse
import shlex
import subprocess
from pathlib import Path

def run(cmd, cwd=None):
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def find_winner_ckpt(runs_dir: Path, dist: str) -> Path:
    run_root = runs_dir / dist
    p = run_root / "winner_model.pkl"
    if p.is_file():
        return p

    cands = sorted(run_root.rglob("winner_model.pkl"))
    if cands:
        return cands[0]

    raise FileNotFoundError(f"Cannot find winner_model.pkl under: {run_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, default="sweep/train_sweep.py",
                    help="Path to the search script.")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--dist", type=str, required=True)
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--julia", type=str, default=None,
                    help="Path to Julia binary. Required if 'julia' is not on PATH.")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--profiles", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)

    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--M", type=int, default=50_000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--exp_thr", type=float, default=4.0)
    ap.add_argument("--use_weights", action="store_true")
    ap.add_argument("--weight_gamma", type=float, default=None,
                    help="Weight damping exponent (0=equal weights, 1=full inverse-variance). Forwarded to train_sweep.py.")
    ap.add_argument("--no_weight_median_norm", action="store_true", default=False,
                    help="Disable median normalization for weights. Forwarded to train_sweep.py.")

    ap.add_argument("--loss_thresh", type=float, default=None,
                    help="Override acceptance loss threshold (global). If set, overrides profile loss_thresh and LOSS_THRESH_PER_SET.")
    ap.add_argument("--populations", type=int, default=None,
                    help="Forwarded to train_sweep.py: number of PySR populations.")
    ap.add_argument("--population_size", type=int, default=None,
                    help="Forwarded to train_sweep.py: size of each PySR population.")
    ap.add_argument("--maxsize", type=int, default=None,
                    help="Forwarded to train_sweep.py: PySR maxsize (max equation complexity).")
    ap.add_argument("--expensive_ratio", type=float, default=None,
                    help="Forwarded to train_sweep.py: expensive op cost = round(maxsize * expensive_ratio).")
    ap.add_argument("--extension", action="store_true",
                    help="Enable extension operators (logaddexp + logdelta0). Forwarded to train_sweep.py.")
    ap.add_argument("--select_candidate", type=str, default=None,
                    help="Forwarded to train_sweep.py: rmse (default), loss, complexity, complexity_then_rmse.")
    ap.add_argument("--pmf_check", type=str, default="soft")
    ap.add_argument("--rmse_target", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save_sampling_logtrue", action="store_true")

    ap.add_argument("--infer_extra", type=str, default="",
                    help="extra args forwarded to distid.main, e.g. \"--grid_pos 200 --save_csv out.csv\"")
    ap.add_argument("--skip_train", action="store_true",
                    help="only run inference using existing winner_model.pkl")
    ap.add_argument("--cwd", type=str, default=".",
                    help="working directory to run commands from (project root)")

    args = ap.parse_args()

    cwd = Path(args.cwd).resolve()
    runs_dir = Path(args.runs_dir).resolve() if Path(args.runs_dir).is_absolute() else (cwd / args.runs_dir)

    if not args.skip_train:
        train_cmd = ["python", str((cwd / args.train_script).resolve())]
        train_cmd += ["--dist", args.dist]
        train_cmd += ["--params", args.params]
        if args.julia is not None:
            train_cmd += ["--julia", args.julia]
        train_cmd += ["--iters", str(args.iters)]
        train_cmd += ["--runs_dir", str(runs_dir)]

        if args.profiles is not None:
            train_cmd += ["--profiles", args.profiles]
        if args.profile is not None:
            train_cmd += ["--profile", args.profile]

        if args.noisy:
            train_cmd += ["--noisy", "--M", str(args.M), "--alpha", str(args.alpha), "--exp_thr", str(args.exp_thr)]
            if args.use_weights:
                train_cmd += ["--use_weights"]
                if args.weight_gamma is not None:
                    train_cmd += ["--weight_gamma", str(args.weight_gamma)]
                if args.no_weight_median_norm:
                    train_cmd += ["--no_weight_median_norm"]


        if args.loss_thresh is not None:
            train_cmd += ["--loss_thresh", str(args.loss_thresh)]

        if args.populations is not None:
            train_cmd += ["--populations", str(args.populations)]
        if args.population_size is not None:
            train_cmd += ["--population_size", str(args.population_size)]
        if args.maxsize is not None:
            train_cmd += ["--maxsize", str(args.maxsize)]
        if args.expensive_ratio is not None:
            train_cmd += ["--expensive_ratio", str(args.expensive_ratio)]

        if args.extension:
            train_cmd += ["--extension"]
        if args.select_candidate is not None:
            train_cmd += ["--select_candidate", args.select_candidate]
        train_cmd += ["--pmf_check", args.pmf_check, "--rmse_target", args.rmse_target]
        if args.seed is not None:
            train_cmd += ["--seed", str(args.seed)]
        if args.save_sampling_logtrue:
            train_cmd += ["--save_sampling_logtrue"]

        run(train_cmd, cwd=str(cwd))

    ckpt = find_winner_ckpt(runs_dir, args.dist)
    print(f"\n[Pipeline] Found winner checkpoint: {ckpt}")

    infer_cmd = ["python", "-m", "distid.main", "--ckpt", str(ckpt)]
    if args.extension:
        infer_cmd += ["--extension"]
    if args.infer_extra.strip():
        infer_cmd += shlex.split(args.infer_extra)

    run(infer_cmd, cwd=str(cwd))

if __name__ == "__main__":
    main()
