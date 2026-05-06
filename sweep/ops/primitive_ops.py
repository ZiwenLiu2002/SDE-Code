import sympy as sp

LOGFAC_JULIA = r"""
logfac(t) = begin
    if t <= -0.9f0
        return NaN32
    end
    try
        if !isdefined(Main, :__SF_LOADED__)
            @eval begin
                using SpecialFunctions
                const __SF_LOADED__ = true
            end
        end
        return Float32(SpecialFunctions.loggamma(Float64(t) + 1.0))
    catch
        z = Float64(t) + 1.0
        val = (z - 0.5)*log(z) - z + 0.9189385332046727 + 1/(12*z) - 1/(360*z^3)
        return Float32(val)
    end
end
""".strip()


LOGB_JULIA = r"""
logB(a, b) = begin
    if a <= 0f0 || b <= 0f0
        return NaN32
    end
    try
        if !isdefined(Main, :__SF_LOADED__)
            @eval begin
                using SpecialFunctions
                const __SF_LOADED__ = true
            end
        end
        return Float32(SpecialFunctions.logbeta(Float64(a), Float64(b)))
    catch
        A = Float64(a); B = Float64(b)
        lg = z -> (z - 0.5)*log(z) - z + 0.9189385332046727 + 1/(12*z) - 1/(360*z^3)
        return Float32(lg(A) + lg(B) - lg(A + B))
    end
end
""".strip()

LOGC_JULIA = r"""
logC(n, k) = begin
    if n < 0 || k < 0 || k > n
        return NaN32
    end
    try
        if !isdefined(Main, :__SF_LOADED__)
            @eval begin
                using SpecialFunctions
                const __SF_LOADED__ = true
            end
        end
        return Float32(loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0))
    catch
        return NaN32
    end
end
""".strip()

LOGADDEXP_JULIA = r"""
logaddexp(x, y) = begin
    m = max(x, y)
    return m + log1p(exp(-abs(x - y)))
end
""".strip()

LOGDELTA0_JULIA = r"""
logdelta0(t) = t == 0f0 ? 0f0 : Float32(-1e6)
""".strip()

BASE_SYMPY_MAP = {
    "logfac":    lambda z: sp.loggamma(z + 1),
    "logB":      lambda a, b: sp.loggamma(a) + sp.loggamma(b) - sp.loggamma(a + b),
    "logC":      lambda n, k: sp.loggamma(n + 1) - sp.loggamma(k + 1) - sp.loggamma(n - k + 1),
    "log":       sp.log,
    "exp":       sp.exp,
    "abs":       sp.Abs,
    "sin":       sp.sin,
    "cos":       sp.cos,
    "logaddexp": lambda a, b: sp.log(sp.exp(a) + sp.exp(b)),
    "logdelta0": lambda t: sp.Piecewise((sp.Integer(0), sp.Eq(t, 0)), (sp.Float(-1e6), True)),
}
