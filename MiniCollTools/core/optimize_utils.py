import numpy as np
import scipy
import scipy.optimize
import differential_evolution

# import cmaes
# from cmaes import CMA


class InfException(BaseException):
    pass


def minimize_region(
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    n_regions=None,
    region_bounds=None,
    region_seeds=None,
    **kwargs
):
    x0 = np.atleast_1d(x0)
    # If scanning regions are provided, then set them up for use
    if region_bounds is not None:
        if region_seeds is not None and len(region_bounds) == len(region_seeds):
            pass
        else:
            region_seeds = np.mean(region_bounds, axis=2)
    # Otherwise, check if a number of divisions is specified
    else:
        if n_regions is None:
            n_regions = 1
        if bounds is not None and n_regions > 1:
            regions = [
                np.linspace(bound[0], bound[1], 1 + n_regions) for bound in bounds
            ]
            shape = [len(region) - 1 for region in regions]
            region_bounds = []
            for idx in np.ndindex(*shape):
                region_bounds.append(
                    [(regions[i][j], regions[i][j + 1]) for i, j in enumerate(idx)]
                )
            region_seeds = np.mean(region_bounds, axis=2)

    res = None
    res_fun = None
    # Scan over all regions if more than one is provided, check seed point explicitly
    if region_bounds is not None and region_seeds is not None:
        for r_bounds, r_seed in zip(region_bounds, region_seeds):
            try:
                if method == "BFGS":
                    r_bounds = None
                region_res = scipy.optimize.minimize(
                    fun,
                    r_seed,
                    args=args,
                    method=method,
                    jac=jac,
                    hess=hess,
                    hessp=hessp,
                    bounds=r_bounds,
                    constraints=constraints,
                    tol=tol,
                    callback=callback,
                    options=options,
                    **kwargs
                )
                if res_fun is None:
                    res = region_res
                    res_fun = res.fun
                elif region_res.fun < res_fun:
                    res = region_res
                    res_fun = res.fun
            except InfException as e:
                pass
        try:
            seed_bounds = list(zip(x0, x0))
            if method == "BFGS":
                seed_bounds = None
            region_res = scipy.optimize.minimize(
                fun,
                x0,
                args=args,
                method=method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=seed_bounds,
                constraints=constraints,
                tol=tol,
                callback=callback,
                options=options,
                **kwargs
            )
            if res_fun is None:
                res = region_res
                res_fun = res.fun
            elif region_res.fun < res_fun:
                res = region_res
                res_fun = res.fun
        except InfException as e:
            pass
        return res
    # Otherwise run a single region minimization
    else:
        try:
            if method == "BFGS":
                bounds = None
            res = scipy.optimize.minimize(
                fun,
                x0,
                args=args,
                method=method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                callback=callback,
                options=options,
                **kwargs
            )
        except InfException:
            pass
    return res


def minimize(
    fung,
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    n_regions=None,
    region_bounds=None,
    region_seeds=None,
    differential_evolution_init=None,
    popsize=None,
    mutation=None,
    recombination=None,
    n_diff_ev=None,
    modes_diff_ev=None,
    center=None,
    **kwargs
):
    if popsize is None:
        popsize = 40
    if recombination is None:
        recombination = 0.9
    if mutation is None:
        mutation = 0.7
    if n_regions is None:
        n_regions = 1
    if n_diff_ev is None:
        n_diff_ev = 2
    if method is None:
        method = "L-BFGS-B"
    if options is None:
        options = {
            "ftol": 1e4 * np.finfo(float).eps,
            "gtol": 1e-18,
            "maxls": 20,
            "maxfun": 1000,
        }
    if jac is None:
        jac = True
    if n_regions == 1 and region_seeds is None:
        region_seeds = [x0]

    # Wrap the objective function and callback so that we can halt the minimizer when it encounters an invalid region of the parameter space

    inf_thresh = 10
    reset_thresh = 10
    inf_count = 0
    good_count = 0
    has_inf = False

    def fung_wrapper(x, *args):
        nonlocal has_inf
        v, grad = fung(x, *args)
        if np.isinf(v):
            has_inf = True
        return v, grad

    def fun_wrapper(x, *args):
        nonlocal has_inf
        v = fun(x, *args)
        if np.isinf(v):
            has_inf = True
        return v

    if callback is None:

        def callback_wrapper(x):
            nonlocal inf_thresh
            nonlocal reset_thresh
            nonlocal inf_count
            nonlocal good_count
            nonlocal has_inf
            if has_inf:
                inf_count += 1
                good_count = 0
                has_inf = False
                if inf_count >= inf_thresh:
                    return False
                    # raise InfException("Got " + str(inf_thresh) + " inf values in a row!")
            else:
                good_count += 1
                if good_count >= reset_thresh:
                    inf_count = 0
            return True

    else:

        def callback_wrapper(x):
            nonlocal inf_thresh
            nonlocal reset_thresh
            nonlocal inf_count
            nonlocal good_count
            nonlocal has_inf
            b = callback(x)
            if has_inf:
                inf_count += 1
                good_count = 0
                has_inf = False
                if inf_count >= inf_thresh:
                    return False
                    # raise InfException("Got " + str(inf_thresh) + " inf values in a row!")
            else:
                good_count += 1
                if good_count >= reset_thresh:
                    inf_count = 0
            return b

    # Run the standard gradient minimizer over provided parameter space regions
    print("###########")
    print("Minimizing regions")
    print("###########")
    res = None
    res = minimize_region(
        fung_wrapper,
        x0,
        args=args,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback_wrapper,
        options=options,
        n_regions=n_regions,
        region_bounds=region_bounds,
        region_seeds=region_seeds,
        **kwargs
    )
    print(res.message)
    # res = scipy.optimize.minimize(fung, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback_wrapper, options=options)

    if callback is None:

        def callback_wrapper(x, convergence):
            return False

    else:

        def callback_wrapper(x, convergence):
            b = callback(x)
            return not b

    # Run the differential evolution
    """
    print("###########")
    print("Differential evolution")
    print("###########")

    init = differential_evolution_init(popsize, bounds, res.x)
    d_res = differential_evolution.differential_evolution(fun, bounds, args=args, callback=callback_wrapper, tol=1e-18, atol=1e-5, popsize=popsize, recombination=recombination, mutation=mutation, init=init, strategy="best1bin", center=center, modes=modes_diff_ev, maxiter=40, disp=True)
    print("Differential evolution ran with", d_res.nit, "iterations")
    print(d_res.message)
    if res is None or d_res.fun < res.fun:
        print("###########")
        print("Polishing")
        print("###########")
        res = d_res
        res = scipy.optimize.minimize(fung, res.x, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options, **kwargs)
        print(res.message)
    """
    return res


def optimize(
    fung,
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    n_regions=None,
    region_bounds=None,
    region_seeds=None,
    differential_evolution_init=None,
    popsize=None,
    **kwargs
):
    pass
