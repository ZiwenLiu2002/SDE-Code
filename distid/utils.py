import math
import numpy as np
from scipy.special import gammaln, zeta as sp_zeta, logsumexp

def zeta_riemann(a: float) -> float:
    return float(sp_zeta(a, 1.0))

def H_N(a: float, N: int) -> float:
    if N < 1:
        raise ValueError("N must be >= 1")
    return float(sp_zeta(a, 1.0) - sp_zeta(a, N + 1.0))

def finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)

def normalize_logpmf(logw: np.ndarray) -> np.ndarray:
    logZ = float(logsumexp(logw))
    p = np.exp(logw - logZ)
    p = np.clip(p, 1e-300, 1.0)
    p = p / p.sum()
    return p

def sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-300
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum() + (q * (np.log(q) - np.log(p))).sum())

def model_logpmf_on_grid(name: str, params: dict, k: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    name = name.lower()
    import numpy as np

    if name in ["zipf", "zeta"]:
        a = float(params["a"]); loc = int(round(params.get("loc", 0)))
        n = k - loc
        y = -np.inf * np.ones_like(k, dtype=float)
        from .utils import zeta_riemann
        m = (n >= 1)
        C = -math.log(zeta_riemann(a))
        y[m] = C - a * np.log(n[m])
        return y

    if name in ["zipfian", "zipfian(1..n)"]:
        a = float(params["a"]); N = int(round(params["N"])); loc = int(round(params.get("loc", 0)))
        n = k - loc
        y = -np.inf * np.ones_like(k, dtype=float)
        from .utils import H_N
        m = (n >= 1) & (n <= N)
        C = -math.log(H_N(a, N))
        y[m] = C - a * np.log(n[m])
        return y

    if name in ["logseries", "log-ser", "logser"]:
        p = float(params["p"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 1)
        y[m] = k[m] * math.log(p) - np.log(k[m]) - math.log(-math.log(1.0 - p))
        return y

    if name.startswith("geometric"):
        p = float(params["p"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 1)
        y[m] = (k[m] - 1.0) * math.log(1.0 - p) + math.log(p)
        return y

    if name == "poisson":
        lam = float(params["lam"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0)
        y[m] = k[m] * math.log(lam) - lam - gammaln(k[m] + 1.0)
        return y

    if name == "binomial":
        n_ = int(round(params["n"])); p = float(params["p"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0) & (k <= n_)
        y[m] = (gammaln(n_ + 1.0) - gammaln(k[m] + 1.0) - gammaln(n_ - k[m] + 1.0)
                + k[m] * math.log(p) + (n_ - k[m]) * math.log(1.0 - p))
        return y

    if name in ["negbinomial", "negativebinomial"]:
        r = float(params["r"]); p = float(params["p"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0)
        km = k[m]
        y[m] = (gammaln(km + r) - gammaln(r) - gammaln(km + 1.0)
                + r * math.log(1.0 - p) + km * math.log(p))
        return y

    if name in ["betabinomial", "beta-binomial", "bb"]:
        n_ = int(round(params["n"])); a = float(params["alpha"]); b = float(params["beta"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0) & (k <= n_)
        km = k[m]
        y[m] = (gammaln(n_ + 1.0) - gammaln(km + 1.0) - gammaln(n_ - km + 1.0)
                + (gammaln(km + a) + gammaln(n_ - km + b) - gammaln(n_ + a + b))
                - (gammaln(a) + gammaln(b) - gammaln(a + b)))
        return y

    if name in ["hypergeometric", "hypergeo", "hg"]:
        N = int(round(params["N"])); K = int(round(params["K"])); n_ = int(round(params["n"]))
        y = -np.inf * np.ones_like(k, dtype=float)
        low = max(0, n_ - (N - K)); high = min(n_, K)
        m = (k >= low) & (k <= high)
        km = k[m]
        y[m] = (gammaln(K + 1.0) - gammaln(km + 1.0) - gammaln(K - km + 1.0)
                + gammaln(N - K + 1.0) - gammaln(n_ - km + 1.0) - gammaln(N - K - (n_ - km) + 1.0)
                - (gammaln(N + 1.0) - gammaln(n_ + 1.0) - gammaln(N - n_ + 1.0)))
        return y

    if name in ["neghypergeometric", "neghypergeo", "neg-hypergeo"]:
        N = int(round(params["N"])); K = int(round(params["K"])); r = int(round(params["r"]))
        y = -np.inf * np.ones_like(k, dtype=float)
        if not (1 <= r <= K <= N):
            return y
        m = (k >= 0) & (k <= (N - K))
        km = k[m]
        y[m] = (gammaln(km + r) - gammaln(km + 1.0) - gammaln(r)
                + gammaln(N - r - km + 1.0) - gammaln(K - r + 1.0) - gammaln(N - K - km + 1.0)
                - (gammaln(N + 1.0) - gammaln(K + 1.0) - gammaln(N - K + 1.0)))
        return y

    if name == "dlaplace":
        a = float(params["a"]); loc = float(params.get("loc", 0))
        return math.log(math.tanh(a / 2.0)) - a * np.abs(k - loc)

    # BNB(r, alpha, beta): logpmf = logC(k+r-1, k) + logB(k+α, r+β) - logB(α, β)
    #   = [logΓ(k+r) - logΓ(r) - logΓ(k+1)] + [logΓ(k+α)+logΓ(r+β)-logΓ(k+r+α+β)] - [logΓ(α)+logΓ(β)-logΓ(α+β)]
    if name in ["betanegbinomial", "beta-negative-binomial", "bnb"]:
        r = float(params["r"])
        a = float(params["alpha"])
        b = float(params["beta"])
        if r <= 0 or a <= 0 or b <= 0:
            return -np.inf * np.ones_like(k, dtype=float)
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0)
        km = k[m]
        from scipy.special import gammaln
        y[m] = (
            gammaln(km + r) - gammaln(r) - gammaln(km + 1.0)
            + (gammaln(km + a) + gammaln(r + b) - gammaln(km + r + a + b))
            - (gammaln(a) + gammaln(b) - gammaln(a + b))
        )
        return y

    if name in ["yulesimon", "yule-simon"]:
        rho = float(params["rho"])
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 1)
        if rho > 0:
            from scipy.special import gammaln
            y[m] = (math.log(rho) + gammaln(rho + 1.0)
                    + gammaln(k[m]) - gammaln(k[m] + rho + 1.0))
        return y

    if name in ["boltzmann", "boltzman", "gibbs"]:
        beta = float(params["beta"]); N = int(round(params["N"]))
        y = -np.inf * np.ones_like(k, dtype=float)
        if beta > 0 and N >= 1:
            q = math.exp(-beta)
            if 0 < q < 1:
                logZ = math.log(q * (1.0 - q**N) / (1.0 - q))
                m = (k >= 1) & (k <= N)
                y[m] = -beta * k[m] - logZ
        return y

    if name == "zig":
        pi = float(params["pi"]); p = float(params["p"])
        log_delta0 = np.where(k == 0, 0.0, -1e300)
        log_geo = math.log(p) + k * math.log(1.0 - p)
        return np.logaddexp(math.log(pi) + log_delta0,
                            math.log(1.0 - pi) + log_geo)

    if name == "zinb":
        pi = float(params["pi"]); r = float(params["r"]); p = float(params["p"])
        log_delta0 = np.where(k == 0, 0.0, -1e300)
        m = k >= 0
        log_nb = np.full_like(k, -np.inf)
        log_nb[m] = (gammaln(k[m] + r) - gammaln(r) - gammaln(k[m] + 1.0)
                     + r * math.log(p) + k[m] * math.log(1.0 - p))
        return np.logaddexp(math.log(pi) + log_delta0,
                            math.log(1.0 - pi) + log_nb)

    if name == "mixbinom":
        n_ = int(round(params["n"])); K_comp = int(round(params["K"]))
        y = -np.inf * np.ones_like(k, dtype=float)
        m = (k >= 0) & (k <= n_)
        log_binom_base = gammaln(n_ + 1.0) - gammaln(k + 1.0) - gammaln(n_ - k + 1.0)
        terms = []
        for i in range(1, K_comp + 1):
            pi_val = float(params[f"p{i}"]); wi = float(params[f"w{i}"])
            log_bi = log_binom_base + k * math.log(max(pi_val, 1e-300)) + (n_ - k) * math.log(max(1.0 - pi_val, 1e-300))
            terms.append(math.log(wi) + log_bi)
        y[m] = np.logaddexp.reduce([t[m] for t in terms], axis=0)
        return y

    raise ValueError(f"unknown family {name}")
