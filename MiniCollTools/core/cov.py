import os
import uproot
import numpy as np
from paths import cov_path, dl_cov_path, wc_cov_path
import sample_info
import binning
import scipy
import scipy.linalg
import scipy.special

from numba import njit

# Steps to get the error matrix
# 0) Define the binning
# 1) Load the fractional covariance matrices from the root files
# 2) Correct the matrix statistical error
# 3) Optional: Add statistical error
# 4) Optional: Add intrinsic MC statistical error
# 5) Optional: Add signal MC statistical error
# 6) Rebin the matrices

# UpdateHistograms: Recompute the per component expectations
# BuildEventVector: Copy expectations from each sample to a vector of expectations
# BuildCollapsedEventVector: Compute the expectation by adding the expectations of individual components
# BuildFullErrorMatrix: Multiply the fractional error matrix by the per component expectation from "BuildEventVector" to obtain the error matrix
# AddFullOscStat: Add the statistical error from the osc component to the diagonal of the error matrix
# BuildCollapsedErrorMatrix: Compute the covariance matrix for the observable bins by adding the different components
# InvertCollapsedErrorMatrix: Invert the collapsed error matrix for use in the Chi2
# Compute Chi2: Compute the Chi2 in the standard way

# The "constrained" error matrix is not used in the fit, but is rather used for reporting purposes, since the muon-neutrino data is generally not displayed on plots alongside the electron neutrino data

# Matrix bins
# nue_osc: 18
# nue_intrinsic: 18
# numu_intrinsic: 17
# nuebar_osc: 18
# nuebar_intrinsic: 18
# numubar_intrinsic: 17


def G_IW(z, q, k):
    lplus = ((1 + q) * k + 1 + np.sqrt((2 * k + 1) * (2 * q * k + 1))) / k
    lminus = ((1 + q) * k + 1 - np.sqrt((2 * k + 1) * (2 * q * k + 1))) / k
    return (z * (1 + k) - k * (1 - q) - np.sqrt(z - lplus) * np.sqrt(z - lminus)) / (
        z * (z + 2 * q * k)
    )


def rei(z, q, g):
    return np.real(z) / np.abs(1 - q + q * z * g) ** 2


def denoising_rei(N, q, lambdas):
    sort_idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[sort_idx]
    Lambdas = np.empty(len(lambdas))
    xsis = np.empty(len(lambdas))
    lN = np.amin(lambdas)
    k = 2 * lN / ((1 - q - lN) ** 2 - 4 * q * lN)
    alpha = 1.0 / (1 + 2 * q * k)
    for i in range(N):
        z = lambdas[i] - 1j * (N ** (-0.5))
        g = np.sum([1.0 / (z - l) for j, l in enumerate(lambdas) if j != i]) / (N - 1)
        xsis[i] = rei(z, q, g)
        g = G_IW(z, q, k)
        Lambdas[i] = (1 + alpha * (lambdas[i] - 1)) / rei(z, q, g)
        if Lambdas[i] > 1 and lambdas[i] < 1:
            xsis[i] = Lambdas[i] * xsis[i]
    s = np.sum(lambdas) / np.sum(xsis)
    res = np.empty(len(lambdas))
    res[sort_idx] = np.real(xsis * s)
    return res


def estimate_cov(est_cov, n):
    positive_semidefinite(est_cov)
    lambdas, vectors = scipy.linalg.eig(est_cov)
    mask = lambdas > 0
    N = np.count_nonzero(mask)
    q = n / N
    new_lambdas = np.zeros(len(lambdas))
    new_lambdas[mask] = denoising_rei(N, q, lambdas[mask])
    new_est_cov = np.real(
        np.dot(np.dot(vectors, np.diag(new_lambdas)), scipy.linalg.inv(vectors))
    )
    return positive_semidefinite(new_est_cov)


def error_expect(N, est_error, mean):
    erf = scipy.special.erf((mean - est_error) * np.sqrt(N / (2 * est_error)))
    erfc = scipy.special.erfc((mean - est_error) * np.sqrt(N / (2 * est_error)))
    return (
        np.sqrt(2 * est_error / (N * np.pi))
        * np.exp(-((mean - est_error) ** 2) * N / (2 * est_error))
        + est_error
        - est_error * erf
    ) / (erfc)


@njit(cache=True)
def positive_semidefinite(cov):
    cov = 0.5 * (cov + cov.T)
    lambdas, vectors = np.linalg.eigh(cov)
    lambdas[lambdas < 0] = 0
    return np.real(np.dot(np.dot(vectors, np.diag(lambdas)), vectors.T))


def reform_covariance_matrix(lambdas, components):
    return np.dot(np.dot(components.T, np.diag(lambdas)), components)


def estimate_cov_minus_poisson(est_cov, n, mean):
    print("Original Total Error")
    print(np.sum(est_cov))
    print("Mean")
    print(np.sum(mean))
    print("Original Total Error - Mean")
    print(np.sum(est_cov) - np.sum(mean))

    est_cov = 0.5 * (est_cov + est_cov.T)
    """
    est_cov = est_cov - np.diag(mean)
    rr = np.diag(est_cov)
    for i in range(len(est_cov)):
        if rr[i] <= 0:
            for j in range(len(est_cov)):
                if i == j or rr[j] <= 0:
                    continue
                else:
                    est = np.abs(est_cov[i, j]**2 / est_cov[j, j])
                    if est_cov[i, i] < est:
                        est_cov[i, i] = est
    #est_corr = est_cov / np.sqrt(np.diag(est_cov))[:, None] / np.sqrt(np.diag(est_cov))[None, :]
    """
    lambdas, vectors = scipy.linalg.eigh(est_cov)
    idxs = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idxs]
    vectors = vectors[:, idxs]
    print(lambdas)
    lambdas[lambdas < 0] = 0
    mask = lambdas > 1e-8
    N = len(lambdas)
    N = np.count_nonzero(mask)
    q = n / N
    new_lambdas = np.zeros(len(lambdas))
    new_lambdas[mask] = denoising_rei(N, q, lambdas[mask])
    new_lambdas[new_lambdas < 0] = 0
    idxs = np.argsort(new_lambdas)[::-1]
    new_lambdas = new_lambdas[idxs]
    print(new_lambdas)
    vectors = vectors[:, idxs]
    vec_inv = vectors.T
    diag_lam = np.diag(new_lambdas)
    v_dot_l = np.dot(vectors, diag_lam)
    complex_cov = np.dot(v_dot_l, vec_inv)
    new_est_cov = np.real(complex_cov)
    # new_est_cov = new_est_corr * np.sqrt(np.diag(est_cov))[:, None] * np.sqrt(np.diag(est_cov))[None, :]
    print("New Total Error")
    print(np.sum(new_est_cov))
    print("New Total Error - Mean")
    print(np.sum(new_est_cov) - np.sum(mean))
    res_est_cov = new_est_cov - np.diag(mean)
    rr = np.diag(res_est_cov)
    for i in range(len(res_est_cov)):
        if rr[i] <= 0:
            for j in range(len(res_est_cov)):
                if i == j or rr[j] <= 0:
                    continue
                else:
                    est = np.abs(res_est_cov[i, j] ** 2 / res_est_cov[j, j])
                    if res_est_cov[i, i] < est:
                        res_est_cov[i, i] = est
    res = positive_semidefinite(res_est_cov)
    print("New Total Error after element-wise subtraction")
    print(np.sum(res))
    return positive_semidefinite(new_est_cov)


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def correct_cov(cov, cv, component_bin_masks, null_cv=None):
    cov = np.copy(cov)
    diag = np.copy(np.diagonal(cov))

    intrinsic_mask = component_bin_masks[("raw", "intrinsic")]
    for component, sample in binning.rebin_components:
        if component == "intrinsic":
            sample_mask = component_bin_masks[("raw", sample)]
            mask = np.logical_and(intrinsic_mask, sample_mask)
            unscale = cv[mask] / sample_info.multisim_scale[sample]
            diag[mask] = np.abs(diag[mask] - 1.0 / unscale)
            if null_cv is not None:
                nonzero = np.logical_and(mask, null_cv != 0)
                diag[nonzero] += 1.0 / null_cv[nonzero]

    cov[np.diag_indices(cov.shape[0])] = diag
    rr = np.copy(cov)
    for i in range(len(cov)):
        for j in range(len(cov)):
            if i == j or rr[j, j] <= 0:
                continue
            else:
                est = np.abs(rr[i, j] ** 2 / rr[j, j])
                if cov[i, i] < est:
                    cov[i, i] = est

    return cov


def correct_cov(frac_cov, cv, component_bin_masks, n_vars=3, frac=False):
    frac_cov = np.copy(frac_cov)

    intrinsic_mask = component_bin_masks[("raw", "intrinsic")]
    mask = np.zeros(len(intrinsic_mask))
    scale = np.ones(len(mask))
    for component, sample in binning.rebin_components:
        sample_mask = component_bin_masks[("raw", sample)]
        scale[sample_mask] = sample_info.multisim_scale[sample]
        if component == "intrinsic" and sample != "numu" and sample != "numubar":
            mask = np.logical_or(mask, np.logical_and(intrinsic_mask, sample_mask))

    orig_cv = cv / scale
    masked_orig_cv = np.copy(orig_cv)
    masked_orig_cv[~mask] = 0

    cov = frac_cov * orig_cv[:, None] * orig_cv[None, :]
    if frac:
        cov = frac_cov

    orig_diag = np.diag(cov)
    diag = orig_diag - orig_cv
    diag[diag < 0] = 0
    multiplier = np.sqrt(diag / orig_diag)
    new_cov = positive_semidefinite(cov * multiplier[:, None] * multiplier[None, :])

    U, S, Vt = scipy.linalg.svd(new_cov, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    components = Vt
    components[np.abs(components) < 1e-8] = 0.0
    S[S < 1e-8] = 0.0

    leftovers = np.copy(S)
    leftovers[:n_vars] = 0.0
    l_cov = reform_covariance_matrix(leftovers, components)

    principal = np.copy(S)
    principal[n_vars:] = 0.0
    p_cov = reform_covariance_matrix(principal, components) + np.diag(
        np.abs(np.diag(l_cov))
    )

    p_cov = positive_semidefinite(p_cov)

    if frac:
        return p_cov
    else:
        return p_cov / orig_cv[:, None] / orig_cv[None, :]


# Add data statistical error directly to the collapsed error matrix
# In the fitter code this is only added to the background components
# Not normally used in the anaysis though
# A better way to do this would be to create a separate covariance matrix for this
def add_stat_error_to_cov(cov, data):
    cov = np.copy(cov)
    cov[np.diag_indices(cov.shape[0])] = data
    return cov


# Add MC intrinsic statistical error as an additional fractional covariance matrix
def mc_intrinsic_stat_error(expect_sq, expect, component_bin_masks):
    expect = np.copy(expect)
    expect[expect == 0] = 1
    error = expect_sq / (expect ** 2)
    mask = component_bin_masks[("raw", "intrinsic")]
    error[~mask] = 0
    error[expect == 0] = 0
    return np.diag(error)


# Add MC intrinsic statistical error as an additional fractional covariance matrix
def mc_osc_stat_error(expect_sq, expect, component_bin_masks):
    expect = np.copy(expect)
    expect[expect == 0] = 1
    error = expect_sq / (expect ** 2)
    mask = component_bin_masks[("raw", "osc")]
    error[~mask] = 0
    error[expect == 0] = 0
    return np.diag(error)


def mc_stat_error(expect_sq, expect, component_bin_masks):
    return mc_intrinsic_stat_error(
        expect_sq, expect, component_bin_masks
    ) + mc_osc_stat_error(expect_sq, expect, component_bin_masks)


def est_norm_error(frac_cov, pred):
    cov = frac_cov * pred[:, None] * pred[None, :]
    return np.sqrt(np.sum(cov) / (np.sum(pred) ** 2))


def allowed_norm_error(cov_tot):
    bounds = np.sqrt(np.diag(cov_tot)[np.diag(cov_tot) > 0]).tolist()
    for i in range(len(cov_tot)):
        for j in range(len(cov_tot)):
            if i == j:
                continue
            sii = cov_tot[i, i]
            sjj = cov_tot[j, j]
            sij = cov_tot[i, j]
            if sii == 0 or sjj == 0:
                continue
            if abs(sij / np.sqrt(sii * sjj)) > 1 + 1e-6:
                raise ValueError(
                    "Input covariance matrix has no valid correlation matrix!"
                )
            if sii == sij and sii == sjj:
                continue
            elif sii == sij:
                bound = sij
            elif sjj == sij:
                bound = sij
            elif sii == sjj:
                bound = (sij + sii) / 2.0
            elif -1e-6 <= (sii + sjj - 2 * sij) <= 0:
                continue
            elif -1e-6 <= (sii * sjj - sij ** 2) <= 0:
                continue
            else:
                bound = (sii + sjj - 2 * sij) / (sii * sjj - sij ** 2)
            if bound <= 0:
                print(sii, sjj, sij)
                print(sii + sjj - 2 * sij)
                print(sii * sjj - sij ** 2)
                raise ValueError("")
            bound = np.sqrt(bound)
            bounds.append(bound)
    bounds = np.array(bounds)
    return np.amin(bounds)


def load_cov(
    null_expect, null_expect_sq, transform_slices, orig_idx_order, component_bin_masks
):
    matrix_files = [
        ("pipprodrawaboutsw", "pipprodrawaboutsw_matrix.root"),
        ("pimprodrawaboutsw", "pimprodrawaboutsw_matrix.root"),
        ("kpprod", "kpprod_matrix.root"),
        ("kmprod", "kmprod_matrix.root"),
        ("k0prod", "k0prod_matrix.root"),
        ("beamunisims", "beamunisims_matrix.root"),
        ("ccpiq2rewt", "ccpiq2rewt_matrix.root"),
        ("dirt", "dirt_matrix_apr18.root"),
        ("hadronic", "hadronic_matrix.root"),
        ("mcunisims_smoothed", "mcunisims_smoothed_matrix.root"),
        ("pi0yield_common_det", "pi0yield_common_det_matrix_apr18.root"),
        ("pi0yield_statbkg", "pi0yield_statbkg_matrix_apr18.root"),
        ("xsec", "xsec_matrix.root"),
        ("opticalmodel", "opticalmodel_nuScaled2.902.root"),
    ]

    all_cov = {}

    for matrix_name, fname in matrix_files:
        with uproot.open(os.path.join(cov_path, fname)) as f:
            cov = np.array(f[matrix_name].values())
            cv = np.array(f["bigcv"].values())
        all_cov[matrix_name] = (
            positive_semidefinite(cov[orig_idx_order, :][:, orig_idx_order]),
            cv[orig_idx_order],
        )

    matrix_name = "opticalmodel"
    cov, cv = all_cov[matrix_name]
    cov = correct_cov(cov, cv, component_bin_masks)
    all_cov[matrix_name] = (cov, cv)

    norms_to_keep = ["dirt", "opticalmodel", "hadronic"]

    for matrix_name in all_cov.keys():
        print(matrix_name)
        print(
            matrix_name,
            est_norm_error(all_cov[matrix_name][0], null_expect),
            allowed_norm_error(all_cov[matrix_name][0]),
        )

    non_det_frac_cov = sum(
        [
            cov
            for matrix_name, (cov, cv) in all_cov.items()
            if matrix_name not in norms_to_keep
        ]
    )
    # non_det_cov = null_expect[:, None] * null_expect[None, :] * non_det_frac_cov

    # Add MC statistical error for the intrinsic components (everything except nue/nubar osc)
    all_cov["mc_instrinsic_stat"] = (
        positive_semidefinite(
            mc_intrinsic_stat_error(null_expect_sq, null_expect, component_bin_masks)
        ),
        null_expect,
    )
    all_cov["mc_osc_stat"] = (
        positive_semidefinite(
            mc_osc_stat_error(null_expect_sq, null_expect, component_bin_masks)
        ),
        null_expect,
    )

    tot_frac_cov = sum([cov for matrix_name, (cov, cv) in all_cov.items()])

    print(
        "Non det:",
        est_norm_error(non_det_frac_cov, null_expect),
        allowed_norm_error(non_det_frac_cov),
    )
    print(
        "Total",
        est_norm_error(tot_frac_cov, null_expect),
        allowed_norm_error(tot_frac_cov),
    )

    norm_var = (
        min(allowed_norm_error(tot_frac_cov), allowed_norm_error(non_det_frac_cov)) ** 2
    )
    print("Normalization error:", np.sqrt(norm_var))
    # norm_var = 0.0812**2

    # sub_norm_frac_cov = np.full_like(non_det_cov, -norm_var)

    # all_cov["sub_norm"] = (sub_norm_frac_cov, null_expect)

    return all_cov


def load_dl_cov(null_NuE_expect, alt_NuE_expect):
    all_cov = {}
    for cov_name in ["cov", "cov_nom", "mc_frac_error"]:
        data = np.loadtxt(os.path.join(dl_cov_path, "nue_1e1p_" + cov_name + ".txt"))
        all_cov[cov_name] = data

    def ensure_pos(frac_cov, expect):
        cov = frac_cov * expect[:, None] * expect[None, :]
        cov = positive_semidefinite(cov)
        frac_cov = cov / (expect[:, None] * expect[None, :])
        return frac_cov

    all_cov["cov_nom"] = all_cov["cov_nom"] - np.diag(all_cov["mc_frac_error"] ** 2)
    all_cov["cov"] = all_cov["cov"] - np.diag(all_cov["mc_frac_error"] ** 2)
    for i in range(len(all_cov["cov"])):
        if all_cov["cov_nom"][i, i] < 0:
            all_cov["cov_nom"][i, i] = 0
        if all_cov["cov"][i, i] < 0:
            all_cov["cov"][i, i] = 0

    norm_var = (
        min(
            est_norm_error(all_cov["cov"], null_NuE_expect),
            allowed_norm_error(all_cov["cov"]),
        )
        ** 2
    )
    print(
        "DL norm error:",
        est_norm_error(all_cov["cov"], null_NuE_expect),
        allowed_norm_error(all_cov["cov"]),
    )
    # all_cov["cov"] = all_cov["cov"] - norm_var
    all_cov["cov"] = ensure_pos(all_cov["cov"], np.ones(len(all_cov["cov"])))
    all_cov["norm_var"] = norm_var

    return all_cov


def load_wc_cov(null_expect):
    all_cov = {}
    for cov_name in ["frac_cov", "mc_frac_error"]:
        data = np.loadtxt(os.path.join(wc_cov_path, "nue_numu_" + cov_name + ".txt"))
        all_cov[cov_name] = data

    def ensure_pos(frac_cov, expect):
        cov = frac_cov * expect[:, None] * expect[None, :]
        cov = positive_semidefinite(cov)
        frac_cov = cov / (expect[:, None] * expect[None, :])
        return frac_cov

    all_cov["frac_cov"] = all_cov["frac_cov"] - np.diag(all_cov["mc_frac_error"] ** 2)
    for i in range(len(all_cov["frac_cov"])):
        if all_cov["frac_cov"][i, i] < 0:
            all_cov["frac_cov"][i, i] = 0

    norm_var = (
        min(
            est_norm_error(all_cov["frac_cov"], null_expect),
            allowed_norm_error(all_cov["frac_cov"]),
        )
        ** 2
    )
    print(
        "WC norm error:",
        est_norm_error(all_cov["frac_cov"], null_expect),
        allowed_norm_error(all_cov["frac_cov"]),
    )
    # all_cov["cov"] = all_cov["cov"] - norm_var
    all_cov["frac_cov"] = ensure_pos(
        all_cov["frac_cov"], np.ones(len(all_cov["frac_cov"]))
    )
    all_cov["norm_var"] = norm_var

    return all_cov


if __name__ == "__main__":
    import load_sample

    mc = load_sample.load_mc()

    data = load_sample.load_data()

    bin_transforms = binning.bin_transforms()
    data_bin_transforms = binning.bin_transforms(is_data=True)
    (
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
        mc_raw_ids,
    ) = bin_transforms
    (
        data_transform_slices,
        data_orig_idx_order,
        data_component_bin_masks,
        data_raw_ids,
    ) = bin_transforms
    split_mc = functions.split_mc(mc, mc_component_bin_masks)

    sorted_data, data_slices = binning.sort_events(
        data, data_bin_transforms, is_data=True
    )
    sorted_mc, mc_slices = binning.sort_events(mc, bin_transforms, is_data=False)

    null_expect = np.array(
        [np.sum(mc[mask]["cv_weight"]) for mask in mc_slices["collapsed"]]
    )
    null_expect_sq = np.array(
        [np.sum(mc[mask]["cv_weight"] ** 2) for mask in mc_slices["collapsed"]]
    )

    cov = load_cov(null_expect, null_expect_sq, mc_transform_slices, mc_orig_idx_order)

    rebin_orig_idx = [
        tuple(mc_orig_idx_order[slc]) for slc in mc_transform_slices["raw_to_rebin"]
    ]
    sorted_rebin_orig_idx = sorted(rebin_orig_idx, key=lambda x: x[0])
    rebin_orig_map = dict(zip(sorted_rebin_orig_idx, range(len(sorted_rebin_orig_idx))))
    rebin_old_idx = np.array([rebin_orig_map[idx] for idx in rebin_orig_idx])
    print(rebin_old_idx)

    released_cov_load = np.loadtxt("../released_cov.txt")
    released_orig_cov = np.zeros([x + 1 for x in np.shape(released_cov_load)])
    released_orig_cov[:-1, :-1] = released_cov_load
    released_cov = released_orig_cov[rebin_old_idx, :][:, rebin_old_idx]
    print(released_cov)

    all_cov = 0
    for k, (c, cv) in cov.items():
        all_cov += c
    for x in all_cov:
        print(" ".join([str(xx) for xx in x]))
