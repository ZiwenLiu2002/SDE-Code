import re
import numpy as np

_OP_PATTERNS = {
    "logC":    re.compile(r"\blogC\s*\("),
    "logB":    re.compile(r"\blogB\s*\("),
    "logfac":  re.compile(r"\blogfac\s*\("),
    "nlogfac": re.compile(r"\bnlogfac\s*\("),
    "log":     re.compile(r"\blog\s*\("),
    "abs":     re.compile(r"\babs\s*\("),
    "*":       re.compile(r"\*"),
}


def count_ops(equation: str):
    s = equation
    return {k: len(p.findall(s)) for k, p in _OP_PATTERNS.items()}

def passes_op_limits(equation: str, limits: dict):
    cnt = count_ops(equation)
    for op, lim in limits.items():
        if cnt.get(op, 0) > lim:
            return False, cnt
    return True, cnt

def pmf_checks(lambda_fn, X, sum_tol: float, logp_max_tol: float):
    try:
        y = np.asarray(lambda_fn(X)).reshape(-1)
        if not np.all(np.isfinite(y)):
            return False, {"reason": "nonfinite"}
        m = float(np.max(y))
        Z = float(np.sum(np.exp(y - m)) * np.exp(m))
        norm_dev = abs(Z - 1.0)
        ok = (m <= logp_max_tol) and (norm_dev <= sum_tol)
        return ok, {"norm_dev": norm_dev, "max_logp": m}
    except Exception as e:
        return False, {"reason": f"exception: {e}"}
