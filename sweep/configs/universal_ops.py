from ops.primitive_ops import LOGFAC_JULIA, LOGB_JULIA, LOGC_JULIA, LOGADDEXP_JULIA, LOGDELTA0_JULIA

UNIVERSAL_UNARY  = [LOGFAC_JULIA, "log", "exp", "abs", "sin", "cos"]
UNIVERSAL_BINARY = ["+", "-", "*", "^", LOGB_JULIA, LOGC_JULIA]
_SPECIALS = ["logfac", "logB", "logC", "log", "exp", "abs", "sin", "cos"]

_INNER_BLOCKED = _SPECIALS + ["*", "^"]

UNIVERSAL_NESTED = {outer: {inner: 0 for inner in _INNER_BLOCKED} for outer in _SPECIALS}

UNIVERSAL_CONSTRAINTS = {
    "^": (-1, 1),
}

_EXT_SPECIALS = ["logaddexp", "logdelta0"]
_EXT_ALL_BLOCKED = _INNER_BLOCKED + _EXT_SPECIALS

EXTENSION_UNARY  = [LOGDELTA0_JULIA]
EXTENSION_BINARY = [LOGADDEXP_JULIA]

# logaddexp args should stay shallow (no nesting of any special, ^, or *)
_LOGADDEXP_NESTED = {inner: 0 for inner in _EXT_ALL_BLOCKED}
# logdelta0 is essentially a leaf: block everything inside it
_LOGDELTA0_NESTED = {inner: 0 for inner in _EXT_ALL_BLOCKED + ["+", "-"]}

EXTENSION_NESTED = {
    "logaddexp": _LOGADDEXP_NESTED,
    "logdelta0": _LOGDELTA0_NESTED,
}

EXTENSION_CONSTRAINTS = {
    "logaddexp": (10, 10),
}


def get_combined_ops(extension: bool = False):
    """Return (unary, binary, nested, constraints) with optional extension ops."""
    unary = list(UNIVERSAL_UNARY)
    binary = list(UNIVERSAL_BINARY)
    nested = {k: dict(v) for k, v in UNIVERSAL_NESTED.items()}
    constraints = dict(UNIVERSAL_CONSTRAINTS)

    if extension:
        unary = unary + EXTENSION_UNARY
        binary = binary + EXTENSION_BINARY
        for outer in nested:
            for ext in _EXT_SPECIALS:
                nested[outer][ext] = 0
        nested.update({k: dict(v) for k, v in EXTENSION_NESTED.items()})
        constraints.update(EXTENSION_CONSTRAINTS)

    return unary, binary, nested, constraints
