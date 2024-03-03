import numpy as np
import math
from numba import njit, vectorize

import autodiff


@vectorize
def lgamma(x):
    return math.lgamma(x)


@njit
def gammaPriorPoissonLikelihood_grad(k, alpha_grad, beta_grad):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.
    Parameters
    ----------
    k : int
        The number of observed events
    alpha_grad : tuple
        Zeroeth element contains Gamma distribution shape parameter
        First element contains respective gradients
    beta_grad : tuple
        Zeroeth element contains Gamma distribution inverse rate parameter
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    val = autodiff.mul_grad(alpha_grad, autodiff.log(beta_grad))
    val = autodiff.plus_grad(val, autodiff.lgamma(autodiff.plus(alpha_grad, k)))
    val = autodiff.minus(val, lgamma(k + 1.0))
    val = autodiff.minus_grad(
        val, autodiff.mul_grad(autodiff.plus(alpha_grad, k), autodiff.log1p(beta_grad))
    )
    val = autodiff.minus_grad(val, autodiff.lgamma(alpha_grad))

    return val


@njit
def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.
    Parameters
    ----------
    k : int
        The number of observed events
    alpha : ndarray
        Gamma distribution shape parameter
    beta : ndarray
        Gamma distribution inverse rate parameter
    Returns
    -------
    ndarray
        the log likelihood
    """
    val = alpha * np.log(beta)
    val += lgamma(alpha + k)
    val -= lgamma(k + 1.0)
    val -= (alpha + k) * np.log1p(beta)
    val -= lgamma(alpha)
    return val


@njit
def poissonLikelihood_grad(k, weight_sum_grad):
    """Computes Log of the Poisson Likelihood.
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum_grad : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """

    logw = autodiff.log(weight_sum_grad)

    klogw = autodiff.mul_r(k, logw)

    klogw_minus_w = autodiff.minus_grad(klogw, weight_sum_grad)

    llh = autodiff.minus(klogw_minus_w, lgamma(k + 1))

    return llh


@njit
def poissonLikelihood(k, weight_sum):
    """Computes Log of the Poisson Likelihood.
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : ndarray
        the sum of the weighted MC event counts
    Returns
    -------
    ndarray
        the log likelihood
    """

    logw = np.log(weight_sum)

    klogw = k * logw

    klogw_minus_w = klogw - weight_sum

    llh = klogw_minus_w - lgamma(k + 1)

    return llh


@njit
def LEff_grad(k, weight_sum_grad, weight_sq_sum_grad, eps=0):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum_grad : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    weight_sq_sum_grad : 2D tuple
        Zeroeth element containsthe sum of the square of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    k = np.asarray(k)
    wsx, wsg = weight_sum_grad
    wsx = np.asarray(weight_sum_grad[0])
    wsg = np.asarray(weight_sum_grad[1])
    wssx = np.asarray(weight_sq_sum_grad[0])

    res = (np.zeros(np.shape(wsx)), np.zeros(np.shape(wsg)))

    # print('wsx:',wsx)
    bad_mask = np.logical_and(np.logical_or(wsx <= eps, wssx < eps), k != 0)
    if np.any(bad_mask):
        res[0][bad_mask] = -np.inf

    poisson_mask = np.logical_and(wssx == 0, wsx > 0)
    if np.any(poisson_mask):
        x0, g0 = poissonLikelihood_grad(
            k[poisson_mask],
            (wsx[poisson_mask], wsg[poisson_mask]),
        )
        res[0][poisson_mask] = x0
        res[1][poisson_mask] = g0

    good_mask = np.logical_and(np.logical_and(~bad_mask, ~poisson_mask), wsx > 0)
    if np.any(good_mask):

        kk = k[good_mask]
        ws = autodiff.slice(weight_sum_grad, good_mask)
        wss = autodiff.slice(weight_sq_sum_grad, good_mask)

        alpha_grad = autodiff.plus(
            autodiff.div_grad(autodiff.pow(ws, np.array(2)), wss), np.array(1.0)
        )
        beta_grad = autodiff.div_grad(ws, wss)
        Lx0, Lg0 = gammaPriorPoissonLikelihood_grad(kk, alpha_grad, beta_grad)
        res[0][good_mask] = Lx0
        res[1][good_mask] = Lg0

    return res


@njit
def LEff(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : ndarray
        the weighted MC event counts
    weight_sq_sum : ndarray
        the square of the weighted MC event counts
    Returns
    -------
    ndarray
        the log likelihood
    """
    k = np.asarray(k)
    wsx = np.asarray(weight_sum)
    wssx = np.asarray(weight_sq_sum)

    res = np.zeros(np.shape(wsx))

    bad_mask = np.logical_and(np.logical_or(wsx <= 0, wssx < 0), k != 0)
    if np.any(bad_mask):
        res[bad_mask] = -np.inf

    poisson_mask = np.logical_and(wssx == 0, wsx > 0)
    if np.any(poisson_mask):
        x0 = poissonLikelihood(
            k[poisson_mask],
            wsx[poisson_mask],
        )
        res[poisson_mask] = x0

    good_mask = np.logical_and(np.logical_and(~bad_mask, ~poisson_mask), wsx > 0)
    if np.any(good_mask):

        kk = k[good_mask]
        ws = weight_sum[good_mask]
        wss = weight_sq_sum[good_mask]

        alpha = ws ** 2 / wss + 1.0
        beta = ws / wss
        Lx0 = gammaPriorPoissonLikelihood(kk, alpha, beta)
        res[good_mask] = Lx0

    return res


@njit
def computeLEff_grad(k, weights):
    """Computes Log of the L_Eff Likelihood from a list of weights.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weights : 2D tuple
        Zeroeth element contains list of the weighted MC events
        First element contains list of respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    weight_sum = autodiff.sum(weights)
    weight_sq_sum = autodiff.sum(autodiff.pow(weights, 2))

    return LEff_grad(k, weight_sum, weight_sq_sum)


@njit
def computeLEff(k, weights):
    """Computes Log of the L_Eff Likelihood from a list of weights.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weights : 2D tuple
        Zeroeth element contains list of the weighted MC events
        First element contains list of respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    weight_sum = np.sum(weights)
    weight_sq_sum = np.sum(weights ** 2)

    return LEff(k, weight_sum, weight_sq_sum)


@njit
def calcEffLLH_grad(data, weights, bin_slices):
    """
    Computes and returns the effective log likelihood
    Parameters
    -----------
    data: array-like
        list of observed events in each analysis bin.
    weights: array-like
        list of sorted weights.
    bin_slices: array-like
        list of bin slices, where each slice picks out the elements in weights
        corresponding to an analysis bin.
    Returns
    --------
    tuple:
        Zeroth element is the effective log likelihood
        First element is the gradient of the effective log likelihood
    """
    llhs = []

    for i, bin_slice in enumerate(bin_slices):
        if bin_slice.stop - bin_slice.start == 0:
            continue
        llhs.append(
            computeLEff_grad(data[i], (weights[0][bin_slice], weights[1][bin_slice]))
        )

    llhs = (np.array([llh[0] for llh in llhs]), np.array([llh[1] for llh in llhs]))
    llh = autodiff.sum(llhs)

    return llh


@njit
def calcEffLLH(data, weights, bin_slices):
    """
    Computes and returns the effective log likelihood
    Parameters
    -----------
    data: array-like
        list of observed events in each analysis bin.
    weights: array-like
        list of sorted weights.
    bin_slices: array-like
        list of bin slices, where each slice picks out the elements in weights
        corresponding to an analysis bin.
    Returns
    --------
    tuple:
        Zeroth element is the effective log likelihood
        First element is the gradient of the effective log likelihood
    """
    llhs = []

    for i, bin_slice in enumerate(bin_slices):
        if bin_slice.stop - bin_slice.start == 0:
            continue
        llhs.append(computeLEff(data[i], weights))

    llh = np.sum(llhs)

    return llh
