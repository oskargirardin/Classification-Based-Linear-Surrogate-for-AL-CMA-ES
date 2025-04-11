from .BinaryConstraintOptimizationAL import BinaryConstraintOptimizationAL


def optimize(
    f,
    constraint,
    x0,
    sigma0,
    surrogate_opts,
    cma_opts,
    verbose=True,
    **algorithm_kwargs
):

    optimizer = BinaryConstraintOptimizationAL(
        f, constraint, x0, sigma0, surrogate_opts, cma_opts, verbose=verbose, **algorithm_kwargs
    )

    es, cfun = optimizer.run()

    return es, cfun
