# SDE: Symbolic Distribution Estimation

SDE recovers the closed-form log-PMF of a discrete distribution and identifies its family and parameters.

**Stage 1 — Search** (`sweep/train_sweep.py`): symbolic regression over a log-PMF dataset. Evaluates candidates against loss, PMF validity, and operator constraints. Saves the winning equation as `winner_model.pkl`.

**Stage 2 — Inference** (`python -m distid.main`): reads the checkpoint, routes the equation to candidate distribution families based on its structure, fits each family's parameters, and reports the best match.

---

## Setup

```bash
conda env create -f environment.yml
conda activate sde
```

Julia 1.10.2 is required. Install it from [julialang.org](https://julialang.org/downloads/) and ensure the `julia` binary is on your `PATH`. If it is not on `PATH`, pass `--julia /path/to/julia` explicitly.

---

## Supported distributions

Each distribution has an exact and a noisy variant. The noisy variant draws `M` samples from the true PMF and uses the smoothed empirical log-probabilities as the training target.

| Name | Default parameters | Noisy variant |
|------|--------------------|---------------|
| Zipf (`zipf`) | `alpha=1.5, K_MAX=200` | `zipf_noisy` |
| Zipfian (`zipfian`) | `alpha=1.5, N=200` | `zipfian_noisy` |
| Logarithmic series (`logseries`) | `p=0.37, K_MAX=200` | `logseries_noisy` |
| Geometric (`geometric`) | `p=0.37, K_MAX=200` | `geometric_noisy` |
| Discrete Laplace (`dlaplace`) | `a=0.85, loc=0.0, K_HALF=100` | `dlaplace_noisy` |
| Boltzmann (`boltzmann`) | `beta=0.4, N=100` | `boltzmann_noisy` |
| Poisson (`poisson`) | `lmbda=10.0, K_MAX=60` | `poisson_noisy` |
| Negative Binomial (`negbinomial`) | `r=10.0, p=0.7, K_MAX=100` | `negbinomial_noisy` |
| Yule-Simon (`yulesimon`) | `rho=1.7, K_MAX=200` | `yulesimon_noisy` |
| Beta-Negative Binomial (`betanegbinomial`) | `r=5.0, alpha=2.0, beta=5.0, K_MAX=100` | `betanegbinomial_noisy` |
| Binomial (`binomial`) | `n=50, p=0.3` | `binomial_noisy` |
| Hypergeometric (`hypergeo`) | `N_total=200, K_good=60, n_draw=80` | `hypergeo_noisy` |
| Negative Hypergeometric (`neghypergeo`) | `N_total=100, K_good=60, r=5` | `neghypergeo_noisy` |
| Beta-Binomial (`betabinomial`) | `n=50, alpha=2.0, beta=5.0` | `betabinomial_noisy` |
| Zero-Inflated Negative Binomial (`zinb`) | `pi=0.3, r=5.0, p=0.4, K_MAX=60` | `zinb_noisy` |
| Zero-Inflated Geometric (`zig`) | `pi=0.3, p=0.4, K_MAX=60` | `zig_noisy` |
| Mixture of Binomials (`mixbinom`) | `n=20, ps=[0.2,0.5,0.8], ws=[0.3,0.4,0.3]` | `mixbinom_noisy` |

Override any parameter with `--params key=value,key=value`.

---

## Operator sets and complexity profiles

The search runs each **complexity profile** as a separate job, then selects the winner across all profiles.

| Profile | Cheap operators (cost 1) | Expensive operators (cost = `round(maxsize × expensive_ratio)`) |
|---------|--------------------------|------------------------------------------------------------------|
| `bias_logC` | `logC`, `logB`, `+`, `-` | `logfac`, `log`, `exp`, `abs`, `sin`, `cos`, `*`, `^` |
| `bias_gamma` | `logfac`, `log`, `*`, `+`, `-` | `logC`, `logB`, `exp`, `abs`, `sin`, `cos`, `^` |

With `--maxsize 15` and `--expensive_ratio 0.5` (defaults), expensive operators cost 8 each. The two profiles bias the search towards combinatorial-style equations (`logC`, `logB`) and gamma-style equations (`logfac`, `log`) respectively. Running both and taking the best improves coverage.

**Extension mode** (`--extension`) adds two operators for zero-inflated and mixture distributions:

| Operator | Cost | Definition |
|----------|------|------------|
| `logdelta0(x)` | 1 | 0 if x = 0, else −10⁶ |
| `logaddexp(a, b)` | 3 | log(eᵃ + eᵇ), numerically stable |

Nesting constraints prevent these from appearing inside `logfac`, `logC`, `log`, `exp`, etc.

---

## Run end-to-end

```bash
python run_search_then_infer.py \
  --cwd "$(pwd)" \
  --train_script sweep/train_sweep.py \
  --runs_dir sweep/runs \
  --dist yulesimon \
  --iters 3000
```

With noisy data and extension operators:

```bash
python run_search_then_infer.py \
  --cwd "$(pwd)" \
  --train_script sweep/train_sweep.py \
  --runs_dir sweep/runs \
  --dist zinb \
  --iters 3000 \
  --noisy --M 50000 --alpha 0.5 --exp_thr 4.0 \
  --extension \
  --infer_extra "--extension --loss_th 1e-4"
```

Use `--skip_train` to re-run inference on an existing checkpoint without repeating the search.

---

## Stage 1: Search (`sweep/train_sweep.py`)

```bash
python sweep/train_sweep.py --dist yulesimon --iters 3000 --runs_dir sweep/runs
```

### How it works

1. Generates the log-PMF dataset for `--dist` (exact or noisy).
2. For each complexity profile, runs a symbolic search with shared operators and profile-specific operator costs.
3. Scans the resulting Pareto front top-to-bottom and accepts the first equation passing all of:
   - **Loss threshold**: `equation.loss ≤ loss_thresh`
   - **PMF check** (`--pmf_check soft/hard`): `|Σ exp(logp) − 1| ≤ 1e-3` and `max(logp) ≤ 1e-3`
   - **Operator quota**: each of `logB`, `logC`, `logfac`, `log`, `exp`, `abs`, `sin`, `cos`, `*` appears at most 3 times
4. Selects the winner across profiles according to `--select_candidate`.
5. Saves `winner_model.pkl` and `handoff_inference.json`.

If no equation passes all constraints, the lowest-loss equation across all profiles is used as a fallback.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dist` | `hypergeo` | Target distribution (see table above) |
| `--params` | | Override distribution parameters, e.g. `n=10,p=0.3` |
| `--iters` | `3000` | Search iterations per profile |
| `--profiles` | all | Comma-separated profile names to run (`bias_logC`, `bias_gamma`) |
| `--profile` | | Run a single named profile |
| `--maxsize` | `15` | Maximum equation complexity (node count) |
| `--expensive_ratio` | `0.5` | Expensive operator cost = `round(maxsize × ratio)`, min 2 |
| `--populations` | `30` | Number of parallel search populations |
| `--population_size` | `60` | Individuals per population |
| `--loss_thresh` | `1e-3` | Override acceptance loss threshold |
| `--select_candidate` | `loss` | Winner selection: `loss`, `rmse`, `complexity`, `complexity_then_rmse` |
| `--extension` | off | Add `logaddexp` + `logdelta0` operators |
| `--pmf_check` | `soft` | `off` / `soft` (warn) / `hard` (reject on failure) |
| `--rmse_target` | `auto` | RMSE reference: `auto` (noisy target if noisy, else true), `true_raw`, `noisy` |
| `--seed` | random | Random seed |

### Noisy mode flags

| Flag | Default | Description |
|------|---------|-------------|
| `--noisy` | off | Use sampled log-PMF as training target instead of exact |
| `--M` | `50000` | Total sample count |
| `--alpha` | `0.5` | Dirichlet smoothing added to each count before normalising |
| `--exp_thr` | `4.0` | Minimum expected count to include a point in the fit |
| `--use_weights` | off | Weight fit points by inverse sampling variance |
| `--weight_gamma` | `1.0` | Damping: `w_final = (w_raw / median(w_raw))^gamma` (0 = equal weights, 1 = full inverse-variance) |
| `--no_weight_median_norm` | off | Disable median normalisation of weights |

---

## Stage 2: Inference (`python -m distid.main`)

```bash
python -m distid.main --ckpt sweep/runs/yulesimon/winner_model.pkl
```

### How it works

1. Loads the checkpoint and collects equations below `--loss_th` (at least `--min_keep` are always included).
2. Detects structural tokens in each equation (`logC`, `logfac`, `logaddexp`, …) to select candidate families.
3. Each candidate recogniser evaluates the equation's log-PMF on a grid and fits that family's parameters.
4. Scores each fit as `rmse + mass_err + complexity_penalty`; close matches are discriminated by symmetric KL divergence and tail behaviour.
5. Prints the top-`k` equation / family pairs.

### Routing logic

| Equation tokens | Candidate families |
|-----------------|-------------------|
| `logaddexp` + `logC`/`logfac` (extension mode) | `zinb`, `mixbinom` |
| `logaddexp`, no `logC`/`logfac` (extension mode) | `zig`, `zinb` |
| `logC`/`logfac` with minus-side loggamma terms | `binomial`, `betabinomial`, `hypergeometric`, `neghypergeometric` |
| `logC`/`logfac`, additive loggamma terms only | `poisson`, `negbinomial`, `betanegbinomial`, `yulesimon` |
| No gamma tokens | `zipf`, `zipfian`, `logseries`, `geometric`, `dlaplace`, `boltzmann` |

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt` | required | Path to `winner_model.pkl` |
| `--loss_th` | `1e-3` | Equations with `loss ≤ loss_th` enter inference |
| `--min_keep` | `5` | Minimum equations to process regardless of `loss_th` |
| `--top_k` | `3` | Number of top candidates to print |
| `--extension` | off | Enable `zinb`, `zig`, `mixbinom` families |
| `--mixbinom_max_K` | `3` | Maximum mixture components tried for `mixbinom` |
| `--mixbinom_K_penalty` | `0.05` | Score penalty per extra mixture component |
| `--select_equation` | `score` | Final ranking: `score`, `complexity`, `complexity_then_score` |
| `--grid_pos` | `150` | Evaluation grid size for positive-support families |
| `--grid_poisson` | `150` | Evaluation grid size for zero-indexed families |
| `--save_csv` | | Save per-equation fit metrics to CSV |
| `--print_expansion` | off | Print loggamma symbolic expansion for the winning equation |
| `--no_strict_struct_filter` | off | Allow non-gamma families even when `logC`/`logfac` is detected |

---

## Outputs

| File | Description |
|------|-------------|
| `sweep/runs/<dist>/winner_model.pkl` | Winning equation checkpoint |
| `sweep/runs/<dist>/handoff_inference.json` | Search metadata: dist, params, equation, loss, noisy settings |
| `sweep/runs/<dist>/sweep_log.txt` | Full stdout log of the search run |
| `sweep/runs/<dist>/<profile>/equations_scored.csv` | Full equation Pareto front from that profile |
| `sweep/runs/<dist>/<profile>/equation_metrics.csv` | RMSE / MAE / calibrated-RMSE for each equation |
| `sweep/runs/<dist>/<dist>_truth.csv` | Ground-truth log-PMF and probabilities |
| `sweep/runs/<dist>/<dist>_noisy_compare.csv` | Noisy vs. true log-PMF (noisy mode only) |
| `sweep/runs/<dist>/<dist>_candidates_summary.csv` | Per-profile winners sorted by `select_candidate` |
| `sweep/runs/<dist>/infer_metrics.csv` | Per-equation inference scores (if `--save_csv` set) |

---

## Project structure

```
.
├── run_search_then_infer.py      # end-to-end wrapper (search → inference)
├── sweep/
│   ├── train_sweep.py            # search stage entry point
│   ├── configs/
│   │   ├── complexity_profiles.py  # operator cost profiles (bias_logC, bias_gamma)
│   │   ├── operator_sets.py        # global loss threshold, PMF tolerances, op quotas
│   │   └── universal_ops.py        # operator definitions, nesting constraints, extension ops
│   ├── datasets/
│   │   └── extra_dists.py          # log-PMF generators and noisy variants
│   ├── ops/
│   │   └── primitive_ops.py        # Julia operator definitions and sympy mappings
│   └── utils/
│       └── eval.py                 # PMF validity check and operator count helpers
└── distid/
    ├── main.py                   # inference entry point: routing, scoring, output
    ├── gate.py                   # structural pre-filter to rank family candidates
    ├── decision.py               # discriminate between close family matches (KL + tail)
    ├── structmatch.py            # loggamma term parser (±x, ±mx coefficients)
    ├── structures.py             # equation token flags
    ├── types.py                  # FitResult dataclass; Recognizer type alias
    ├── utils.py                  # shared math utilities and model_logpmf_on_grid
    ├── utils_expand.py           # symbolic loggamma expansion
    └── families/                 # one recognizer per distribution family
        ├── poisson.py
        ├── geometric.py
        ├── logseries.py
        ├── zipf.py / zipfian.py
        ├── binomial.py
        ├── negbinomial.py
        ├── betabinomial.py
        ├── hypergeometric.py
        ├── neghypergeometric.py
        ├── dlaplace.py
        ├── betanegbinomial.py
        ├── yulesimon.py
        ├── boltzmann.py
        ├── zig.py                # zero-inflated geometric
        ├── zinb.py               # zero-inflated negative binomial
        └── mixbinom.py           # K-component mixture of binomials
```
