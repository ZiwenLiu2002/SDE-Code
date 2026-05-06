import numpy as np
from typing import Dict, Optional
from scipy.special import gammaln
from scipy.stats import dlaplace as _scipy_dlaplace

EPS = 1e-300

def _logC(a, b):
    return gammaln(a + 1.0) - gammaln(b + 1.0) - gammaln(a - b + 1.0)

def _logB(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)

def _stable_log_softmax(logp: np.ndarray) -> np.ndarray:
    m = float(np.max(logp))
    logZ = m + np.log(np.sum(np.exp(logp - m)))
    return logp - logZ

def _noisy_from_logp(
    logp: np.ndarray,
    M_samples: int = 2_000_000,
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:

    assert M_samples >= 1 and alpha >= 0.0
    rng = np.random.default_rng(seed)
    logp = np.asarray(logp, dtype=float).reshape(-1)
    p_true = np.exp(_stable_log_softmax(logp))
    K = p_true.size
    counts = rng.multinomial(M_samples, p_true)
    p_hat = (counts + alpha) / (M_samples + alpha * K)
    p_hat = np.clip(p_hat, EPS, 1.0)
    return np.log(p_hat)

def make_zipf(alpha: float = 1.5, K_MAX: int = 200):
    assert alpha > 1.0
    k = np.arange(1, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = -alpha * np.log(k)
    return X, logp, k.astype(int)

def make_zipfian(alpha: float = 1.5, N: int = 200):
    assert N >= 1
    k = np.arange(1, N + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = -alpha * np.log(k)
    return X, logp, k.astype(int)

def make_logseries(p: float = 0.37, K_MAX: int = 200):
    assert 0.0 < p < 1.0
    k = np.arange(1, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = k * np.log(p) - np.log(k)
    return X, logp, k.astype(int)

def make_geometric(p: float = 0.37, K_MAX: int = 200):
    assert 0.0 < p < 1.0
    k = np.arange(1, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = np.log(p) + (k - 1.0) * np.log(1.0 - p)
    return X, logp, k.astype(int)

def make_dlaplace(a: float = 0.85, loc: float = 0.0, K_HALF: int = 100):
    assert a > 0.0 and K_HALF >= 1
    k = np.arange(int(np.floor(loc)) - K_HALF, int(np.ceil(loc)) + K_HALF + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = np.log(np.tanh(a / 2.0)) - a * np.abs(k - loc)
    return X, logp, k.astype(int)

def make_boltzmann(beta: float = 0.4, N: int = 100):
    assert N >= 1 and beta > 0.0
    k = np.arange(1, N + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = -beta * k
    return X, logp, k.astype(int)

def make_poisson(lmbda: float = 10.0, K_MAX: int = 60):
    assert lmbda > 0.0
    k = np.arange(0, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = k * np.log(lmbda) - lmbda - gammaln(k + 1.0)
    return X, logp, k.astype(int)

def make_negbinomial(r: float = 10.0, p: float = 0.7, K_MAX: int = 100):
    assert r > 0.0 and 0.0 < p < 1.0
    k = np.arange(0, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = (gammaln(k + r) - gammaln(k + 1.0) - gammaln(r)) + r * np.log(p) + k * np.log(1.0 - p)
    return X, logp, k.astype(int)

def make_yulesimon(rho: float = 1.7, K_MAX: int = 200):
    assert rho > 0.0
    k = np.arange(1, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = np.log(rho) + gammaln(rho + 1.0) + gammaln(k) - gammaln(k + rho + 1.0)
    return X, logp, k.astype(int)

def make_betanegbinomial(r: float = 5.0, alpha: float = 2.0, beta: float = 5.0, K_MAX: int = 100):
    assert r > 0.0 and alpha > 0.0 and beta > 0.0
    k = np.arange(0, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = (gammaln(k + r) - gammaln(k + 1.0) - gammaln(r)) \
           + _logB(r + alpha, k + beta) - _logB(alpha, beta)
    return X, logp, k.astype(int)

def make_binomial(n: int = 50, p: float = 0.3):
    assert n >= 0 and 0.0 <= p <= 1.0
    k = np.arange(0, n + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = _logC(n, k) + k * np.log(max(p, EPS)) + (n - k) * np.log(max(1.0 - p, EPS))
    return X, logp, k.astype(int)

def make_hypergeo(N_total: int = 200, K_good: int = 60, n_draw: int = 80):
    assert 0 <= K_good <= N_total and 0 <= n_draw <= N_total
    x_min = max(0, n_draw - (N_total - K_good))
    x_max = min(n_draw, K_good)
    k = np.arange(x_min, x_max + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = _logC(K_good, k) + _logC(N_total - K_good, n_draw - k) - _logC(N_total, n_draw)
    return X, logp, k.astype(int)

def make_neghypergeo(N_total: int = 100, K_good: int = 60, r: int = 5, r_fail: Optional[int] = None):
    if r_fail is not None:
        r = r_fail
    N_bad = N_total - K_good
    assert 1 <= r <= N_bad
    k_max = min(K_good, N_total - r)
    k = np.arange(0, k_max + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = _logC(k + r - 1.0, k) + _logC(N_bad, r) + _logC(K_good, k) - _logC(N_total, k + r)
    return X, logp, k.astype(int)

def make_betabinomial(n: int = 50, alpha: float = 2.0, beta: float = 5.0):
    assert n >= 0 and alpha > 0.0 and beta > 0.0
    k = np.arange(0, n + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = _logC(n, k) + _logB(k + alpha, n - k + beta) - _logB(alpha, beta)
    return X, logp, k.astype(int)

def make_zipf_noisy(alpha: float = 1.5, K_MAX: int = 200,
                    M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_zipf(alpha=alpha, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_zipfian_noisy(alpha: float = 1.5, N: int = 200,
                       M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_zipfian(alpha=alpha, N=N)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_logseries_noisy(p: float = 0.37, K_MAX: int = 200,
                         M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_logseries(p=p, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_geometric_noisy(p: float = 0.37, K_MAX: int = 200,
                         M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_geometric(p=p, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_dlaplace_noisy(a: float = 0.85, loc: float = 0.0, K_HALF: int = 100,
                        M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    k = np.arange(int(np.floor(loc)) - K_HALF, int(np.ceil(loc)) + K_HALF + 1, dtype=float)
    X = k.reshape(-1, 1)

    rng = np.random.default_rng(seed)
    samp = _scipy_dlaplace.rvs(a=a, loc=loc, size=M_samples, random_state=rng).astype(int)

    k_min, k_max = int(k[0]), int(k[-1])
    samp = np.clip(samp, k_min, k_max)
    counts = np.bincount(samp - k_min, minlength=len(k))[:len(k)]

    K = len(k)
    p_hat = (counts + alpha_smooth) / (M_samples + alpha_smooth * K)
    p_hat = np.clip(p_hat, EPS, 1.0)
    logph = np.log(p_hat)
    return X, logph, k.astype(int)

def make_boltzmann_noisy(beta: float = 0.4, N: int = 100,
                         M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_boltzmann(beta=beta, N=N)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_poisson_noisy(lmbda: float = 10.0, K_MAX: int = 60,
                       M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_poisson(lmbda=lmbda, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_negbinomial_noisy(r: float = 10.0, p: float = 0.7, K_MAX: int = 100,
                           M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_negbinomial(r=r, p=p, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_yulesimon_noisy(rho: float = 0.5, K_MAX: int = 200,
                         M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_yulesimon(rho=rho, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_betanegbinomial_noisy(r: float = 5.0, alpha: float = 2.0, beta: float = 5.0, K_MAX: int = 100,
                               M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_betanegbinomial(r=r, alpha=alpha, beta=beta, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_binomial_noisy(n: int = 10, p: float = 0.3,
                        M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_binomial(n=n, p=p)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_hypergeo_noisy(N_total: int = 200, K_good: int = 60, n_draw: int = 80,
                        M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_hypergeo(N_total=N_total, K_good=K_good, n_draw=n_draw)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp


def make_neghypergeo_noisy(N_total: int = 100, K_good: int = 60, r: int = 5, r_fail: Optional[int] = None,
                           M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_neghypergeo(N_total=N_total, K_good=K_good, r=r, r_fail=r_fail)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_betabinomial_noisy(n: int = 50, alpha: float = 2.0, beta: float = 5.0,
                            M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_betabinomial(n=n, alpha=alpha, beta=beta)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp

def make_zinb(pi: float = 0.3, r: float = 5.0, p: float = 0.40, K_MAX: int = 60):
    """Zero-Inflated Negative Binomial (failure parameterization)."""
    assert 0.0 < pi < 1.0 and r > 0.0 and 0.0 < p < 1.0
    k = np.arange(0, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    log_nb = (gammaln(k + r) - gammaln(k + 1.0) - gammaln(r)
              + r * np.log(p) + k * np.log1p(-p))
    log_delta0 = np.where(k == 0, 0.0, -1e300)
    logp = np.logaddexp(np.log(pi) + log_delta0, np.log1p(-pi) + log_nb)
    return X, logp, k.astype(int)


def make_zig(pi: float = 0.3, p: float = 0.40, K_MAX: int = 60):
    """Zero-Inflated Geometric (support starts at 0)."""
    assert 0.0 < pi < 1.0 and 0.0 < p < 1.0
    k = np.arange(0, K_MAX + 1, dtype=float)
    X = k.reshape(-1, 1)
    log_geo = np.log(p) + k * np.log1p(-p)
    log_delta0 = np.where(k == 0, 0.0, -1e300)
    logp = np.logaddexp(np.log(pi) + log_delta0, np.log1p(-pi) + log_geo)
    return X, logp, k.astype(int)


def make_mixbinom(n: int = 20, ps=None, ws=None,
                  p1: float = None, p2: float = None, w1: float = None):
    """K-component mixture of Binomials."""
    if ps is None:
        if p1 is not None or p2 is not None or w1 is not None:
            p1 = 0.25 if p1 is None else p1
            p2 = 0.65 if p2 is None else p2
            w1 = 0.60 if w1 is None else w1
            ps = [p1, p2]
            ws = [w1, 1.0 - w1] if ws is None else ws
        else:
            ps = [0.2, 0.5, 0.8]
            ws = [0.3, 0.4, 0.3]
    if ws is None:
        ws = np.ones(len(ps), dtype=float) / float(len(ps))
    ps = np.asarray(ps, dtype=float)
    ws = np.asarray(ws, dtype=float)
    assert n >= 1 and ps.ndim == 1 and ws.ndim == 1 and ps.size == ws.size and ps.size >= 1
    assert np.all((0.0 < ps) & (ps < 1.0)) and np.all(ws > 0.0)
    ws = ws / np.sum(ws)
    k = np.arange(0, n + 1, dtype=float)
    X = k.reshape(-1, 1)
    logp = None
    for p_i, w_i in zip(ps, ws):
        log_b = _logC(n, k) + k * np.log(p_i) + (n - k) * np.log1p(-p_i)
        term = np.log(w_i) + log_b
        logp = term if logp is None else np.logaddexp(logp, term)
    return X, logp, k.astype(int)


def make_zinb_noisy(pi: float = 0.3, r: float = 5.0, p: float = 0.40, K_MAX: int = 60,
                    M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_zinb(pi=pi, r=r, p=p, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp


def make_zig_noisy(pi: float = 0.3, p: float = 0.40, K_MAX: int = 60,
                   M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_zig(pi=pi, p=p, K_MAX=K_MAX)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp


def make_mixbinom_noisy(n: int = 20, ps=None, ws=None,
                        p1: float = None, p2: float = None, w1: float = None,
                        M_samples: int = 2_000_000, alpha_smooth: float = 0.5, seed: int = 123):
    X, logp, supp = make_mixbinom(n=n, ps=ps, ws=ws, p1=p1, p2=p2, w1=w1)
    return X, _noisy_from_logp(logp, M_samples, alpha_smooth, seed), supp


EXTRA_DATASETS: Dict[str, callable] = {
    "zipf": make_zipf,
    "zipfian": make_zipfian,
    "logseries": make_logseries,
    "geometric": make_geometric,
    "dlaplace": make_dlaplace,
    "boltzmann": make_boltzmann,
    "poisson": make_poisson,
    "negbinomial": make_negbinomial,
    "yulesimon": make_yulesimon,
    "betanegbinomial": make_betanegbinomial,
    "binomial": make_binomial,
    "hypergeo": make_hypergeo,
    "neghypergeo": make_neghypergeo,
    "betabinomial": make_betabinomial,
    "zipf_noisy": make_zipf_noisy,
    "zipfian_noisy": make_zipfian_noisy,
    "logseries_noisy": make_logseries_noisy,
    "geometric_noisy": make_geometric_noisy,
    "dlaplace_noisy": make_dlaplace_noisy,
    "boltzmann_noisy": make_boltzmann_noisy,
    "poisson_noisy": make_poisson_noisy,
    "negbinomial_noisy": make_negbinomial_noisy,
    "yulesimon_noisy": make_yulesimon_noisy,
    "betanegbinomial_noisy": make_betanegbinomial_noisy,
    "binomial_noisy": make_binomial_noisy,
    "hypergeo_noisy": make_hypergeo_noisy,
    "neghypergeo_noisy": make_neghypergeo_noisy,
    "betabinomial_noisy": make_betabinomial_noisy,
    "zinb":            make_zinb,
    "zig":             make_zig,
    "mixbinom":        make_mixbinom,
    "zinb_noisy":      make_zinb_noisy,
    "zig_noisy":       make_zig_noisy,
    "mixbinom_noisy":  make_mixbinom_noisy,
}
