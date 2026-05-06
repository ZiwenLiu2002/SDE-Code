DEFAULT_MAXSIZE = 15
DEFAULT_EXPENSIVE_RATIO = 0.5

ALL_OPS = [
    "logfac", "log", "exp", "abs", "sin", "cos",
    "+", "-", "*", "^", "logB", "logC",
]

EXTENSION_OPS = ["logdelta0", "logaddexp"]

COMPLEXITY_PROFILES = [
    {
        "name": "bias_logC",
        "cheap_ops": {"logC", "logB", "+", "-"},
    },
    {
        "name": "bias_gamma",
        "cheap_ops": {"logfac", "log", "*", "+", "-"},
    },
]


def expensive_cost(maxsize: int = DEFAULT_MAXSIZE,
                   expensive_ratio: float = DEFAULT_EXPENSIVE_RATIO) -> int:
    cost = int(round(float(maxsize) * float(expensive_ratio)))
    return max(2, cost)


def resolve_profile(profile: dict,
                    maxsize: int = DEFAULT_MAXSIZE,
                    expensive_ratio: float = DEFAULT_EXPENSIVE_RATIO,
                    extension: bool = False) -> dict:
    import math
    cheap = set(profile["cheap_ops"])
    high = expensive_cost(maxsize, expensive_ratio)
    costs = {op: (1 if op in cheap else high) for op in ALL_OPS}
    if extension:
        # logdelta0: cheap (1); logaddexp: moderate cost between cheap and expensive
        costs["logdelta0"] = 1
        costs["logaddexp"] = 3
    return costs


def get_profile(name: str) -> dict:
    for p in COMPLEXITY_PROFILES:
        if p["name"] == name:
            return p
    raise ValueError(f"Unknown profile: {name}. "
                     f"Available: {[p['name'] for p in COMPLEXITY_PROFILES]}")
