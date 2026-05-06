from . import (
    poisson, geometric, logseries, zipf, zipfian,
    binomial, negbinomial, betabinomial,
    hypergeometric, neghypergeometric, dlaplace, betanegbinomial,
    yulesimon, boltzmann,
    zinb, zig, mixbinom,
)

REGISTRY = {
    "poisson": poisson.recog,
    "geometric": geometric.recog,
    "logseries": logseries.recog,
    "zipf": zipf.recog,
    "zipfian": zipfian.recog,
    "binomial": binomial.recog,
    "negbinomial": negbinomial.recog,
    "betabinomial": betabinomial.recog,
    "hypergeometric": hypergeometric.recog,
    "neghypergeometric": neghypergeometric.recog,
    "dlaplace": dlaplace.recog,
    "betanegbinomial": betanegbinomial.recog_betanegbinomial,
    "beta-negative-binomial": betanegbinomial.recog_betanegbinomial,
    "bnb": betanegbinomial.recog_betanegbinomial,
    "yulesimon": yulesimon.recog_yulesimon,
    "boltzmann": boltzmann.recog_boltzmann,
    "zinb": zinb.recog,
    "zig": zig.recog,
    "mixbinom": mixbinom.recog,
}
