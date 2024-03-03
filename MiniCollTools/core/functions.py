import sys

core_path = "./core/"
sys.path.insert(0, core_path)
import numpy as np
import likelihood_grad
import autodiff as ad
import sample_info
import load_sample
import binning
import cov as covariance
import scipy
import scipy.linalg
from numba import njit


@njit(cache=True)
def data_diff_grad(expect_grad, binned_data):
    res = -expect_grad
    res[:, 0] += binned_data
    return res


@njit(cache=True)
def gauss_prefactor(cov):
    cond = cov != 0
    nonzero = np.full(cov.shape[0], False)
    for i in range(cov.shape[0]):
        nonzero |= cond[i]
    idx = np.arange(len(nonzero))[nonzero]
    small_cov = np.empty((len(idx), len(idx)))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            small_cov[i, j] = cov[ii, jj]
    det_sign, log_small_cov_det = np.linalg.slogdet(small_cov)
    res = -0.5 * (small_cov.shape[0] * np.log(2 * np.pi) + log_small_cov_det)
    return res


def get_det(cov_grad):
    cov = cov_grad[:, :, 0]
    cond = cov != 0
    nonzero = np.full(cov.shape[0], False)
    for i in range(cov.shape[0]):
        nonzero |= cond[i]
    idx = np.arange(len(nonzero))[nonzero]
    small_cov = np.empty((len(idx), len(idx)))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            small_cov[i, j] = cov[ii, jj]
    det_sign, log_small_cov_det = np.linalg.slogdet(small_cov)
    return log_small_cov_det


# read this: https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density
@njit(cache=True)
def gauss_prefactor_grad(cov_grad, cov_inv):
    # dimensions of cov_grad:
    # 0, 1: the matrix dimensions
    # 2: the value/derivative dimension
    # cov_grad[:,:,0] is the covariance matrix
    # cov_grad[:,:,1:] is the gradient of the covariance matrix
    # cov_inv is just the 2d matrix
    cov = cov_grad[:, :, 0]
    assert np.all(np.isfinite(cov))
    cond = cov != 0
    nonzero = np.full(cov.shape[0], False)
    for i in range(cov.shape[0]):
        nonzero |= cond[i]
    idx = np.arange(len(nonzero))[nonzero]
    small_cov = np.empty((len(idx), len(idx)))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            small_cov[i, j] = cov[ii, jj]
    det_sign, log_small_cov_det = np.linalg.slogdet(small_cov)
    if det_sign <= 0:
        print(np.linalg.eig(cov))
        print("[")
        for r in cov:
            print(r)
        print("]")
        print(np.diag(cov))
        print(np.diag(small_cov))
        print(det_sign)
        print(log_small_cov_det)
        raise ValueError("Covaraince matrix has a non-positive determinant!")
        res = -np.inf
    else:
        res = -0.5 * (small_cov.shape[0] * np.log(2 * np.pi) + log_small_cov_det)
    cov_product = np.expand_dims(cov_inv, 2) * cov_grad[:, :, 1:]
    cov_p = np.empty(
        (np.shape(cov_product)[0] * np.shape(cov_product)[1], np.shape(cov_product)[2])
    )
    idx = 0
    for i in range(np.shape(cov_grad)[0]):
        for j in range(np.shape(cov_grad)[1]):
            cov_p[idx] = cov_product[i, j]
            idx += 1
    grad = np.sum(cov_p, axis=0)
    grad *= -0.5
    ret = np.empty(len(grad) + 1)
    ret[0] = res
    ret[1:] = grad
    return ret


# The exponential term in the gaussian prior
@njit(cache=True)
def gauss_exponent(diff, cov_inv):
    diff = np.asarray(diff)
    res = -0.5 * np.sum(np.expand_dims(diff, 0) * np.expand_dims(diff, 1) * cov_inv)
    return res


@njit(cache=True)
def gauss_exponent_grad(diff_grad, cov, cov_inv):
    # dimensions of diff:
    # 0: the parameters
    # 1: the value/derivative dimension
    # dimensions of cov:
    # 0, 1: the matrix dimensions
    # 2: the value/derivative dimension
    # cov[:,:,0] is the covariance matrix
    # cov[:,:,1:] is the gradient of the covariance matrix
    # cov_inv is just the 2d matrix
    diff = diff_grad
    diff = np.asarray(diff)
    ddiff = diff[:, 1:]
    diff = diff[:, 0]
    res = -0.5 * np.sum(np.expand_dims(diff, 1) * np.expand_dims(diff, 0) * cov_inv)

    # i   : indexes the 1d array "diff"
    # i, l: index the inverse covariance matrix
    # l, m: index the covaraince matrix
    # m, j: index the inverse covariance matrix
    # j   : indexes the 1d array "diff"
    # k   : indexes the derivative

    # fmt: off
    # derivative of exponent with respect to diff
    #cov_product = -0.5 * (
        #((
        ##            i,    j,    k
        #    ddiff[   :, None,    :]
        #  +  diff[   :, None, None])
        #* cov_inv[   :,    :, None] *
        #(    diff[None,    :, None]
        #  + ddiff[None,    :,    :]
        #)),
        #((
        #    np.expand_dims(ddiff, 1)
        #  + np.expand_dims(np.expand_dims(diff, 1), 2))
        #*   np.expand_dims(cov_inv, 2) *
        #(   np.expand_dims(np.expand_dims(diff, 0), 2)
        #  + np.expand_dims(ddiff, 0)
        #))
    #)

    cov_product = -0.5 * (
        (
            np.expand_dims(ddiff, 0) *
            np.expand_dims(np.expand_dims(diff, 1), 2) * 
            np.expand_dims(cov_inv, 2)
        ) +
        (
            np.expand_dims(ddiff, 1) *
            np.expand_dims(np.expand_dims(diff, 0), 2) * 
            np.expand_dims(cov_inv, 2)
        )
    )

    p_shape = np.shape(cov_product)
    cov_p = np.empty((p_shape[0] * p_shape[1], p_shape[2]))
    idx = 0
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            cov_p[idx] = cov_product[i, j]
            idx += 1
    res_d = np.sum(cov_p, axis=0)

    middle = np.empty((np.shape(cov_inv)[0], np.shape(cov_inv)[1], np.shape(cov)[2] - 1))
    for i in range(np.shape(cov)[2] - 1):
        middle[:, :, i] = np.dot(np.dot(cov_inv, cov[:, :, i+1]), cov_inv)

    # derivative of exponent with respect to cov
    """
    cov_product = 0.5 * (
        ##            i,    l,    m,    j,    k
        #     diff[   :, None, None, None, None]
        #* cov_inv[   :,    :, None, None, None]
        #*     cov[None,    :,    :, None,   1:]
        #* cov_inv[None, None,    :,    :, None]
        #*    diff[None, None, None,    :, None],
        np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(diff, 1), 2), 3), 4)
        * np.expand_dims(np.expand_dims(np.expand_dims(cov_inv, 2), 3), 4)
        * np.expand_dims(np.expand_dims(cov, 0), 3)[:, :, :, :, 1:]
        * np.expand_dims(np.expand_dims(np.expand_dims(cov_inv, 0), 1), 4)
        * np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(diff, 0), 1), 2), 4)
    )
    """
    cov_product = 0.5 * (
        ##            i,    m,    k
        #     diff[   :, None, None]
        #*  middle[   :,    :,   1:]
        #*    diff[None,    :, None],
        np.expand_dims(np.expand_dims(diff, 1), 2)
        * middle
        * np.expand_dims(np.expand_dims(diff, 0), 2)
    )
    #axis=(0, 1, 2, 3),

    """
    p_shape = np.shape(cov_product)
    cov_p = np.empty((p_shape[0] * p_shape[1] * p_shape[2] * p_shape[3], p_shape[4]))
    idx = 0
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            for k in range(p_shape[2]):
                for l in range(p_shape[3]):
                    cov_p[idx] = cov_product[i, j, k, l]
                    idx += 1
    """
    p_shape = np.shape(cov_product)
    cov_p = np.empty((p_shape[0] * p_shape[1], p_shape[2]))
    idx = 0
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            cov_p[idx] = cov_product[i, j]
            idx += 1
    res_d += np.sum(cov_p, axis=0)

    ret = np.empty(np.shape(res_d)[-1] + 1)
    ret[0] = res
    ret[1:] = res_d
    # fmt: on
    return ret


# @njit(cache=True)
# def frac_gauss_exponent_grad(diff_grad, cov_inv):
#    # dimensions of diff:
#    # 0: the parameters
#    # 1: the value/derivative dimension
#    # dimensions of cov:
#    # 0, 1: the matrix dimensions
#    # 2: the value/derivative dimension
#    # cov_inv is just the 2d matrix
#    diff = diff_grad
#    diff = np.asarray(diff)
#    ddiff = diff[:, 1:]
#    diff = diff[:, 0]
#    #res = -0.5 * np.sum(np.expand_dims(diff, 1) * np.expand_dims(diff, 0) * cov_inv)
#    res = -0.5 * np.sum(diff * np.sum(np.expand_dims(diff, 1) * cov_inv, axis=0))
#
#    cov_p = np.sum(np.expand_dims(diff, 1) * cov_inv, axis=0)
#    res_d = -np.sum(ddiff * np.expand_dims(cov_p, 1), axis=0)
#
#    ret = np.empty(np.shape(res_d)[-1] + 1)
#    ret[0] = res
#    ret[1:] = res_d
#    return ret


@njit(cache=True)
def frac_gauss_exponent_grad(diff_grad, cov_inv):
    # dimensions of diff:
    # 0: the parameters
    # 1: the value/derivative dimension
    # dimensions of cov:
    # 0, 1: the matrix dimensions
    # 2: the value/derivative dimension
    # cov_inv is just the 2d matrix
    diff = diff_grad
    diff = np.asarray(diff)
    ddiff = diff[:, 1:]
    diff = diff[:, 0]
    res = -0.5 * np.sum(np.expand_dims(diff, 1) * np.expand_dims(diff, 0) * cov_inv)

    # fmt: off

    cov_product = -0.5 * (
        (
            np.expand_dims(ddiff, 0) *
            np.expand_dims(np.expand_dims(diff, 1), 2) * 
            np.expand_dims(cov_inv, 2)
        ) +
        (
            np.expand_dims(ddiff, 1) *
            np.expand_dims(np.expand_dims(diff, 0), 2) * 
            np.expand_dims(cov_inv, 2)
        )
    )

    p_shape = np.shape(cov_product)
    cov_p = np.empty((p_shape[0] * p_shape[1], p_shape[2]))
    idx = 0
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            cov_p[idx] = cov_product[i, j]
            idx += 1
    res_d = np.sum(cov_p, axis=0)

    ret = np.empty(np.shape(res_d)[-1] + 1)
    ret[0] = res
    ret[1:] = res_d
    # fmt: on
    return ret


@njit(cache=True)
def frac_gauss_exponent(diff, cov_inv):
    return -0.5 * np.sum(np.expand_dims(diff, 1) * np.expand_dims(diff, 0) * cov_inv)


@njit(cache=True)
def gauss(gauss_exponent, gauss_prefactor):
    return gauss_exponent + gauss_prefactor


@njit(cache=True)
def gauss_grad(gauss_exponent_grad, gauss_prefactor_grad):
    return gauss_exponent_grad + gauss_prefactor_grad


def leff(the_store, param_defs):
    def likelihood(data, params):
        # Get the MC expectation and error terms with their gradients
        expect = the_store.get_prop("expect", params)
        expect_sq = the_store.get_prop("expect_sq", params)

        # Compute the likelihood and its gradient (requires a transformation between gradient representations via ad.unpack)
        # The sum is over the likelihood in each bin (the use of ad.sum is essential for treading the gradient representation correctly)
        say_likelihood = ad.sum(
            likelihood_grad.LEff(data, ad.unpack(expect), ad.unpack(expect_sq))
        )
        assert np.all(say_likelihood[0] <= 0)

        # Compute the gaussian prior with its gradient
        gauss = the_store.get_prop("gauss", params)

        # Add the two terms
        like = ad.plus_grad(say_likelihood, ad.unpack(gauss[None, :]))

        return ad.mul(like, -1.0)

    pass


def data():
    return load_sample.load_data()


def data_sorting_info(data, bin_transforms):
    (
        sorted_data,
        data_slices,
    ) = binning.sort_events(data, bin_transforms, is_data=True)
    return (sorted_data, data_slices)


def mc_sorting_info(mc, bin_transforms):
    (
        sorted_mc,
        mc_slices,
    ) = binning.sort_events(mc, bin_transforms, is_data=False)
    return (sorted_mc, mc_slices)


def binned_data(data_slices):
    return np.array([slc.stop - slc.start for slc in data_slices])[:-1]


def mc():
    return load_sample.load_mc()


def split_mc(mc, mc_masks_by_component):
    nue_masks = binning.masks_by_component(mc["nue"], is_data=False)
    nuebar_masks = binning.masks_by_component(mc["nuebar"], is_data=False)

    numu_to_nue_mc = mc["nue"][nue_masks["osc"]]
    numubar_to_nuebar_mc = mc["nuebar"][nuebar_masks["osc"]]

    numu_to_numu_mc = np.concatenate(
        [
            mc["numu"][mc["numu"]["ntuple_inno"] == sample_info.neutrino_ids["numu"]],
            mc["numubar"][
                mc["numubar"]["ntuple_inno"] == sample_info.neutrino_ids["numu"]
            ],
            mc["nue"][
                np.logical_and(
                    mc["nue"]["ntuple_inno"] == sample_info.neutrino_ids["numu"],
                    ~nue_masks["osc"],
                )
            ],
            mc["nuebar"][
                np.logical_and(
                    mc["nuebar"]["ntuple_inno"] == sample_info.neutrino_ids["numu"],
                    ~nuebar_masks["osc"],
                )
            ],
        ]
    )

    numubar_to_numubar_mc = np.concatenate(
        [
            mc["numu"][
                mc["numu"]["ntuple_inno"] == sample_info.neutrino_ids["numubar"]
            ],
            mc["numubar"][
                mc["numubar"]["ntuple_inno"] == sample_info.neutrino_ids["numubar"]
            ],
            mc["nue"][
                np.logical_and(
                    mc["nue"]["ntuple_inno"] == sample_info.neutrino_ids["numubar"],
                    ~nue_masks["osc"],
                )
            ],
            mc["nuebar"][
                np.logical_and(
                    mc["nuebar"]["ntuple_inno"] == sample_info.neutrino_ids["numubar"],
                    ~nuebar_masks["osc"],
                )
            ],
        ]
    )

    nue_to_nue_mc = np.concatenate(
        [
            mc["numu"][mc["numu"]["ntuple_inno"] == sample_info.neutrino_ids["nue"]],
            mc["numubar"][
                mc["numubar"]["ntuple_inno"] == sample_info.neutrino_ids["nue"]
            ],
            mc["nue"][
                np.logical_and(
                    mc["nue"]["ntuple_inno"] == sample_info.neutrino_ids["nue"],
                    ~nue_masks["osc"],
                )
            ],
            mc["nuebar"][
                np.logical_and(
                    mc["nuebar"]["ntuple_inno"] == sample_info.neutrino_ids["nue"],
                    ~nuebar_masks["osc"],
                )
            ],
        ]
    )

    nuebar_to_nuebar_mc = np.concatenate(
        [
            mc["numu"][mc["numu"]["ntuple_inno"] == sample_info.neutrino_ids["nuebar"]],
            mc["numubar"][
                mc["numubar"]["ntuple_inno"] == sample_info.neutrino_ids["nuebar"]
            ],
            mc["nue"][
                np.logical_and(
                    mc["nue"]["ntuple_inno"] == sample_info.neutrino_ids["nuebar"],
                    ~nue_masks["osc"],
                )
            ],
            mc["nuebar"][
                np.logical_and(
                    mc["nuebar"]["ntuple_inno"] == sample_info.neutrino_ids["nuebar"],
                    ~nuebar_masks["osc"],
                )
            ],
        ]
    )

    split_mc = {
        "numu_to_nue": numu_to_nue_mc,
        "numubar_to_nuebar": numubar_to_nuebar_mc,
        "numu_to_numu": numu_to_numu_mc,
        "numubar_to_numubar": numubar_to_numubar_mc,
        "nue_to_nue": nue_to_nue_mc,
        "nuebar_to_nuebar": nuebar_to_nuebar_mc,
    }
    return split_mc


def sorted_mc(mc_sorting_info):
    (
        sorted_mc,
        mc_slices,
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
    ) = mc_sorting_info
    return sorted_mc


def mc_slices(mc_sorting_info):
    (
        sorted_mc,
        mc_slices,
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
    ) = mc_sorting_info
    return tuple(mc_slices)


def get_item(k):
    def item(mc_dict):
        return mc_dict[k]

    return item


def mc_transform_slices(mc_sorting_info):
    (
        sorted_mc,
        mc_slices,
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
    ) = mc_sorting_info
    return mc_transform_slices


def mc_component_bin_masks(mc_sorting_info):
    (
        sorted_mc,
        mc_slices,
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
    ) = mc_sorting_info
    return mc_component_bin_masks


def mc_orig_idx_order(mc_sorting_info):
    (
        sorted_mc,
        mc_slices,
        mc_transform_slices,
        mc_orig_idx_order,
        mc_component_bin_masks,
    ) = mc_sorting_info
    return mc_orig_idx_order


def mc_osc_weight_masks(sorted_mc):
    nue_osc_mask = np.logical_and(
        np.logical_or(
            sorted_mc["ntuple_inno"] == sample_info.neutrino_ids["nue"],
            sorted_mc["ntuple_inno"] == sample_info.neutrino_ids["nuebar"],
        ),
        sorted_mc["ntuple_iflux"] == 11,
    )
    numu_osc_mask = np.logical_and(
        np.logical_or(
            sorted_mc["ntuple_inno"] == sample_info.neutrino_ids["numu"],
            sorted_mc["ntuple_inno"] == sample_info.neutrino_ids["numubar"],
        ),
        sorted_mc["ntuple_iflux"] == 16,
    )
    osc_mask = np.logical_or(nue_osc_mask, numu_osc_mask)
    nue_sub_osc_mask = nue_osc_mask[osc_mask]
    numu_sub_osc_mask = numu_osc_mask[osc_mask]
    return (
        osc_mask,
        nue_osc_mask,
        numu_osc_mask,
        nue_sub_osc_mask,
        numu_sub_osc_mask,
    )


@njit(cache=True)
def mc_osc_factor(sorted_mc):
    const = 1.266932679039099
    L = sorted_mc["ntuple_nuleng"]
    E = sorted_mc["ntuple_enugen"]
    mask = E != 0
    res = np.empty(len(mask))
    res[~mask] = np.pi / 4.0
    res[mask] = const * L[mask] / E[mask]
    return res


####################################
# Nick to work on


@njit(cache=True)
def compute_constrained_cv_ratio(constrained_expect, cv_expect):
    cv_expect = np.copy(cv_expect)
    mask = cv_expect > 0
    res = np.empty(np.shape(cv_expect))
    res[~mask] = 1.0
    res[mask] = constrained_expect[mask] / cv_expect[mask]
    return res


# for wire-cell: data-release to smearing matrix ratio
@njit(cache=True)
def compute_DR_SM_ratio(data_release_expect, smearing_matrix):
    smearing_matrix_expect = np.sum(np.copy(smearing_matrix), axis=1)
    mask = smearing_matrix_expect > 0
    res = np.empty(np.shape(smearing_matrix_expect))
    res[~mask] = 1.0
    res[mask] = data_release_expect[mask] / smearing_matrix_expect[mask]
    return res


# uses mb simulation, ub baseline
@njit(cache=True)
def mc_osc_factor_ubbaseline(sorted_mc):
    const = 1.266932679039099
    dist = 0.541 - 0.4685
    L = sorted_mc["ntuple_nuleng"]
    E = sorted_mc["ntuple_enugen"]
    finite_L = np.isfinite(L)
    finite_E = np.isfinite(E)
    L[finite_L] = L[finite_L] - dist
    negative_L = L < 0
    negative_E = E < 0
    zero_L = L == 0
    zero_E = E == 0
    far_mask = L > 1

    bad_mask = np.logical_or(~finite_L, ~finite_E)
    negative_mask = np.logical_or(negative_L, negative_E)
    bad_mask = np.logical_or(bad_mask, negative_mask)
    bad_mask = np.logical_or(bad_mask, far_mask)
    zero_mask = np.logical_and(zero_E, ~zero_L)
    zero_mask = np.logical_and(zero_mask, ~bad_mask)
    zero_mask = np.logical_or(zero_mask, far_mask)
    mask = np.logical_and(~bad_mask, ~zero_mask)
    res = np.empty(len(mask))
    res[zero_mask] = np.pi / 4.0
    res[mask] = const * L[mask] / E[mask]
    return res, bad_mask


# preliminary ub numu->nue osc weights
# bins of true energy
# comes from oscillated rate / intrinsic rate in MB
@njit(cache=True)
def ub_osc_weight_map_grad(cv_expect, osc_expect_grad):
    cv_expect = np.copy(cv_expect)
    weights = np.empty(np.shape(osc_expect_grad))  # full
    mask = cv_expect > 0
    nmask = ~mask
    weights[:, 0][mask] = osc_expect_grad[:, 0][mask] / cv_expect[mask]
    weights[:, 0][nmask] = 1.0
    weights[:, 1:][mask] = osc_expect_grad[:, 1:][mask] / np.expand_dims(
        cv_expect[mask], 1
    )
    weights[:, 1:][nmask] = 0
    return weights


@njit(cache=True)
def ub_osc_weight_map(cv_expect, osc_expect):
    cv_expect = np.copy(cv_expect)
    weights = np.ones(np.shape(osc_expect))  # full
    mask = cv_expect > 0
    weights[mask] = osc_expect[mask] / cv_expect[mask]
    return weights


@njit(cache=True)
def dl_build_smearing_matrix(mc, slices, true_energy_edges, reco_energy_edges):
    expect_flat = raw_center(mc["cv_weight"], slices)
    shape = (len(reco_energy_edges) - 1, len(true_energy_edges) - 1)
    expect = expect_flat.reshape(shape)
    return expect


@njit(cache=True)
def ub_binned_osc_weights(ub_NuE_bin_edges, ub_NuE_nue_mc, ub_osc_weight_map):
    cv_weights = ub_NuE_nue_mc["cv_weight"]
    reco_energy = ub_NuE_nue_mc["reco_energy"]
    true_energy = ub_NuE_nue_mc["true_energy"]
    osc_weights = (ub_osc_weight_map[:, 1])[
        np.searchsorted(ub_osc_weight_map[:, 0], 1e-3 * true_energy) - 1
    ]
    n_osc, _ = np.histogram(
        reco_energy, bins=ub_NuE_bin_edges, weights=osc_weights * cv_weights
    )
    n_cv, _ = np.histogram(reco_energy, bins=ub_NuE_bin_edges, weights=cv_weights)
    return n_osc / n_cv


@njit(cache=True)
def ub_expect_nue_grad(nue_constrained_bkg_grad, numu_expect, osc_expect_grad):
    res = nue_constrained_bkg_grad + osc_expect_grad
    res[:, 0] += numu_expect
    return res


@njit(cache=True)
def ub_expect_nue(nue_constrained_bkg, numu_expect, osc_expect):
    return nue_constrained_bkg + numu_expect + osc_expect


@njit(cache=True)
def ub_expect_numu_grad(numu_constrained_bkg_grad, bkg_expect):
    res = numu_constrained_bkg_grad
    res[:, 0] += bkg_expect
    return res


@njit(cache=True)
def ub_expect_numu(numu_constrained_bkg, bkg_expect):
    return numu_constrained_bkg + bkg_expect


@njit(cache=True)
def ub_mc_var_from_frac_error_grad(expect_grad, frac_error):
    res = np.empty(np.shape(expect_grad))
    res[:, 0] = (expect_grad[:, 0] * frac_error) ** 2
    res[:, 1:] = (
        np.expand_dims(2 * expect_grad[:, 0] * frac_error ** 2, 1) * expect_grad[:, 1:]
    )
    return res


@njit(cache=True)
def ub_mc_var_from_frac_error(expect, frac_error):
    return (expect * frac_error) ** 2


@njit(cache=True)
def ub_apply_sys_norms(nominal_expect, sys_norms):
    return nominal_expect * sys_norms


@njit(cache=True)
def ub_apply_sys_norms_grad(nominal_expect_grad, sys_norms_grad):
    n_expect_grad = np.shape(nominal_expect_grad)[1] - 1
    n_sys_grad = np.shape(sys_norms_grad)[1] - 1
    tot_shape = (len(nominal_expect_grad), 1 + n_expect_grad + n_sys_grad)
    ub_sys_expect_grad = np.zeros(tot_shape)
    ub_sys_expect_grad[:, 0] = sys_norms_grad[:, 0] * nominal_expect_grad[:, 0]
    ub_sys_expect_grad[:, 1 : n_expect_grad + 1] = (
        np.expand_dims(sys_norms_grad[:, 0], 1) * nominal_expect_grad[:, 1:]
    )
    ub_sys_expect_grad[:, n_expect_grad + 1 :] = sys_norms_grad[:, 1:] * np.expand_dims(
        nominal_expect_grad[:, 0], 1
    )
    return ub_sys_expect_grad


# Nick todo: make sure these functions are working properly
@njit(cache=True)
def wc_expect_grad(
    nue_FC_expect_grad, nue_PC_expect_grad, numu_FC_expect_grad, numu_PC_expect_grad
):
    return np.concatenate(
        (
            nue_FC_expect_grad,
            nue_PC_expect_grad,
            numu_FC_expect_grad,
            numu_PC_expect_grad,
        )
    )


@njit(cache=True)
def wc_expect(nue_FC_expect, nue_PC_expect, numu_FC_expect, numu_PC_expect):
    return np.concatenate(
        (nue_FC_expect, nue_PC_expect, numu_FC_expect, numu_PC_expect)
    )


@njit(cache=True)
def ub_expect_mask(ub_expect, eps=1e-6):
    return np.where(ub_expect < eps)


@njit(cache=True)
def ub_mask_modes(ub_modes, ub_mask):
    # print('\n\n Before mask')
    # for mode in ub_modes: print(mode)
    ub_modes[ub_mask] = 0
    # print('\n\n After mask')
    # for mode in ub_modes: print(mode)
    # print('\n\n')
    return ub_modes


####################################


@njit(cache=True)
def mc_osc_sin2_grad(mc_osc_factor, dm2):
    alpha = dm2 * mc_osc_factor  # sub
    sin = np.sin(alpha)  # sub
    sin2 = sin ** 2  # sub
    d_sin2 = 2 * mc_osc_factor * np.cos(alpha) * sin  # sub
    return sin2, d_sin2


@njit(cache=True)
def mc_osc_sin2(mc_osc_factor, dm2):
    alpha = dm2 * mc_osc_factor  # sub
    sin = np.sin(alpha)  # sub
    sin2 = sin ** 2  # sub
    return sin2


@njit(cache=True)
def mc_osc_sin2_check_grad(mc_osc_factor, dm2):
    osc_factor, zero_mask = mc_osc_factor
    alpha = dm2 * osc_factor  # sub
    sin = np.sin(alpha)  # sub
    sin2 = sin ** 2  # sub
    d_sin2 = 2 * osc_factor * np.cos(alpha) * sin  # sub
    sin2[zero_mask] = 0
    d_sin2[zero_mask] = 0
    return sin2, d_sin2


@njit(cache=True)
def mc_osc_sin2_check(mc_osc_factor, dm2):
    osc_factor, zero_mask = mc_osc_factor
    alpha = dm2 * osc_factor  # sub
    sin = np.sin(alpha)  # sub
    sin2 = sin ** 2  # sub
    sin2[zero_mask] = 0
    return sin2


@njit(cache=True)
def numu_to_nue_osc_weights_grad(
    mc_osc_factor, mc_osc_sin2_grad, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2, d_sin2 = mc_osc_sin2_grad
    weights = np.empty((len(sin2), 4))  # full
    if effective_radius != -1:
        val = r ** 2 * np.sin(2 * theta) ** 2
        val_r = 2 * r * np.sin(2 * theta) ** 2
        val_theta = 2 * r ** 2 * np.sin(4 * theta)
        weights[:, 0] = sin2 * val  # value
        weights[:, 1] = d_sin2 * val  # d_dm2
        weights[:, 2] = sin2 * val_r  # d_r
        weights[:, 3] = sin2 * val_theta  # d_theta
    else:
        weights[:, 0] = 1
        weights[:, 1:] = 0
    return weights


@njit(cache=True)
def numu_to_nue_osc_weights(
    mc_osc_factor, mc_osc_sin2, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2 = mc_osc_sin2
    if effective_radius != -1:
        weights = sin2 * r ** 2 * np.sin(2 * theta) ** 2
    else:
        weights = np.ones(len(sin2))
    return weights


@njit(cache=True)
def nue_to_nue_osc_weights_grad(
    mc_osc_factor, mc_osc_sin2_grad, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2, d_sin2 = mc_osc_sin2_grad
    weights = np.empty((len(sin2), 4))  # full
    if effective_radius != -1:
        val = -4 * r * np.sin(theta) ** 2 * (1 - r * np.sin(theta) ** 2)
        val_r = -4 * np.sin(theta) ** 2 - 8 * r * np.sin(theta) ** 4
        val_theta = (
            -8 * r * np.cos(theta) * np.sin(theta) * (1 - 2 * r * np.sin(theta) ** 2)
        )
        weights[:, 0] = 1 + sin2 * val  # value
        weights[:, 1] = d_sin2 * val  # d_dm2
        weights[:, 2] = sin2 * val_r  # d_r
        weights[:, 3] = sin2 * val_theta  # d_theta
    else:
        weights[:, 0] = 1
        weights[:, 1:] = 0
    return weights


@njit(cache=True)
def nue_to_nue_osc_weights(
    mc_osc_factor, mc_osc_sin2, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2 = mc_osc_sin2
    if effective_radius != -1:
        weights = 1 - sin2 * 4 * r * np.sin(theta) ** 2 * (1 - r * np.sin(theta) ** 2)
    else:
        weights = np.ones(len(sin2))
    return weights


@njit(cache=True)
def numu_to_numu_osc_weights_grad(
    mc_osc_factor, mc_osc_sin2_grad, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2, d_sin2 = mc_osc_sin2_grad
    weights = np.empty((len(sin2), 4))  # full
    if effective_radius != -1:
        val = -4 * r * np.cos(theta) ** 2 * (1 - r * np.cos(theta) ** 2)
        val_r = -4 * np.cos(theta) ** 2 + 8 * r * np.cos(theta) ** 4
        val_theta = (
            -8 * r * np.cos(theta) * np.sin(theta) * (-1 + 2 * r * np.cos(theta) ** 2)
        )
        weights[:, 0] = 1 + sin2 * val  # value
        weights[:, 1] = d_sin2 * val  # d_dm2
        weights[:, 2] = sin2 * val_r  # d_r
        weights[:, 3] = sin2 * val_theta  # d_theta
    else:
        weights[:, 0] = 1
        weights[:, 1:] = 0
    return weights


@njit(cache=True)
def numu_to_numu_osc_weights(
    mc_osc_factor, mc_osc_sin2, effective_radius, effective_theta
):
    r, theta = effective_radius, effective_theta
    sin2 = mc_osc_sin2
    if effective_radius != -1:
        weights = 1 - sin2 * 4 * r * np.cos(theta) ** 2 * (1 - r * np.cos(theta) ** 2)
    else:
        weights = np.ones(len(sin2))
    return weights


def cv_weight(sorted_mc):
    return sorted_mc["cv_weight"]


@njit(cache=True)
def mc_weights_grad(cv_weight, mc_osc_weights_grad):
    return np.expand_dims(cv_weight, 1) * mc_osc_weights_grad


@njit(cache=True)
def mc_weights(cv_weight, mc_osc_weights):
    res = cv_weight * mc_osc_weights
    return res


@njit(cache=True)
def raw_center_grad(mc_weights_grad, mc_raw_slices):
    shape = (len(mc_raw_slices),) + np.shape(mc_weights_grad)[1:]
    res = np.empty(shape)
    for i, slc in enumerate(mc_raw_slices):
        res[i] = np.sum(mc_weights_grad[slc], axis=0)
    return res


@njit(cache=True)
def raw_center(mc_weights, mc_raw_slices):
    shape = (len(mc_raw_slices),)
    res = np.empty(shape)
    for i, slc in enumerate(mc_raw_slices):
        res[i] = np.sum(mc_weights[slc], axis=0)
    return res


@njit(cache=True)
def raw_center_sq_grad(mc_weights_grad, mc_raw_slices):
    shape = (len(mc_raw_slices),) + np.shape(mc_weights_grad)[1:]
    res = np.empty(shape)
    for i, slc in enumerate(mc_raw_slices):
        res[i, 0] = np.sum(mc_weights_grad[slc, 0] ** 2, axis=0)
        res[i, 1:] = np.sum(
            2 * mc_weights_grad[:, 0:1] * mc_weights_grad[:, 1:], axis=0
        )
    return res


@njit(cache=True)
def raw_center_sq(mc_weights, mc_raw_slices):
    shape = (len(mc_raw_slices),)
    res = np.empty(shape)
    for i, slc in enumerate(mc_raw_slices):
        res[i] = np.sum(mc_weights[slc] ** 2, axis=0)
    return res


@njit(cache=True)
def rebinned_center(raw_center, mc_raw_to_rebin_slices):
    res = binning.rebin_cv(raw_center, mc_raw_to_rebin_slices)
    return res


def raw_frac_cov_list(cov_cv_map):
    covs = [item[0] for item in cov_cv_map.values()]
    return np.array(covs)


def ub_raw_frac_cov_list():
    frac_covs = covariance.load_ub_cov()
    return frac_covs["cov"]


@njit(cache=True)
def raw_sys_frac_cov(raw_frac_cov_list):
    return covariance.positive_semidefinite(np.sum(raw_frac_cov_list, axis=0))


@njit(cache=True)
def raw_sys_cov_grad(raw_sys_frac_cov, raw_expect_grad):
    res = (
        raw_sys_frac_cov
        * raw_expect_grad[:, 0][:, None]
        * raw_expect_grad[:, 0][None, :]
    )
    res_d = raw_sys_frac_cov[:, :, None] * (
        raw_expect_grad[:, None, 1:] * raw_expect_grad[:, 0][None, :, None]
        + raw_expect_grad[:, 0][:, None, None] * raw_expect_grad[None, :, 1:]
    )
    return np.concatenate([res[:, :, None], res_d], axis=-1)


@njit(cache=True)
def mul_cov_grad(cov_grad, mu_grad):
    cov = cov_grad[:, :, 0]
    cov_d = cov_grad[:, :, 1:]
    mu, mu_d = mu_grad[:, 0], mu_grad[:, 1:]
    res = cov * np.expand_dims(mu, 0) * np.expand_dims(mu, 1)
    res_d = np.zeros((np.shape(cov_d)))
    res_d += (
        cov_d
        * np.expand_dims(np.expand_dims(mu, 0), 2)
        * np.expand_dims(np.expand_dims(mu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(mu_d, 0)
        * np.expand_dims(np.expand_dims(mu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(np.expand_dims(mu, 0), 2)
        * np.expand_dims(mu_d, 1)
    )
    return np.concatenate((np.expand_dims(res, 2), res_d), axis=2)


@njit(cache=True)
def mul_const_cov_grad(cov, mu_grad):
    mu, mu_d = mu_grad[:, 0], mu_grad[:, 1:]
    res = cov * np.expand_dims(mu, 0) * np.expand_dims(mu, 1)
    res_d = np.zeros((np.shape(cov)[0], np.shape(cov)[1], np.shape(mu_grad)[1] - 1))
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(mu_d, 0)
        * np.expand_dims(np.expand_dims(mu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(np.expand_dims(mu, 0), 2)
        * np.expand_dims(mu_d, 1)
    )
    return np.concatenate((np.expand_dims(res, 2), res_d), axis=2)


@njit(cache=True)
def mul_cov(cov, mu):
    return cov * np.expand_dims(mu, 0) * np.expand_dims(mu, 1)


@njit(cache=True)
def div_cov_grad(cov_grad, mu_grad):
    cov = cov_grad[:, :, 0]
    cov_d = cov_grad[:, :, 1:]
    mu, mu_d = mu_grad[:, 0], mu_grad[:, 1:]
    invmu = 1.0 / mu
    invmu_d = np.expand_dims(-(invmu ** 2), 1) * mu_d
    res = cov / np.expand_dims(mu, 0) / np.expand_dims(mu, 1)
    res_d = np.zeros((np.shape(cov_d)))
    res_d += (
        cov_d
        * np.expand_dims(np.expand_dims(invmu, 0), 2)
        * np.expand_dims(np.expand_dims(invmu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(invmu_d, 0)
        * np.expand_dims(np.expand_dims(invmu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(np.expand_dims(invmu, 0), 2)
        * np.expand_dims(invmu_d, 1)
    )
    return np.concatenate((np.expand_dims(res, 2), res_d), axis=2)


@njit(cache=True)
def div_const_cov_grad(cov_grad, mu_grad):
    mu, mu_d = mu_grad[:, 0], mu_grad[:, 1:]
    invmu = 1.0 / mu
    invmu_d = np.expand_dims(-(invmu ** 2), 1) * mu_d
    res = cov / np.expand_dims(mu, 0) / np.expand_dims(mu, 1)
    res_d = np.zeros((np.shape(cov)[0], np.shape(cov)[1], np.shape(mu_grad)[1] - 1))
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(invmu_d, 0)
        * np.expand_dims(np.expand_dims(invmu, 1), 2)
    )
    res_d += (
        np.expand_dims(cov, 2)
        * np.expand_dims(np.expand_dims(invmu, 0), 2)
        * np.expand_dims(invmu_d, 1)
    )
    return np.concatenate((np.expand_dims(res, 2), res_d), axis=2)


@njit(cache=True)
def div_cov(cov, mu):
    return cov / np.expand_dims(mu, 0) / np.expand_dims(mu, 1)


@njit(cache=True)
def rebinned_sys_frac_cov(raw_sys_cov, rebinned_null_center, mc_raw_to_rebin_slices):
    rebinned_cov = binning.rebin_cov(raw_sys_cov, mc_raw_to_rebin_slices)
    return div_cov(rebinned_cov, rebinned_null_center)


@njit(cache=True)
def identity(x):
    return x


@njit(cache=True)
def rebinned_osc_stat_error_grad(rebinned_expect_grad, osc_bin_mask):
    rebinned_error = np.zeros(np.shape(rebinned_expect_grad))
    rebinned_error[osc_bin_mask] = rebinned_expect_grad[osc_bin_mask]
    return rebinned_error


@njit(cache=True)
def rebinned_osc_stat_error(rebinned_expect, osc_bin_mask):
    rebinned_error = np.zeros(np.shape(rebinned_expect))
    rebinned_error[osc_bin_mask] = rebinned_expect[osc_bin_mask]
    return rebinned_error


# This not actually used in the fitter
@njit(cache=True)
def rebinned_intrinsic_stat_error_grad(rebinned_center_grad, intrinsic_bin_mask):
    rebinned_error = np.zeros(np.shape(rebinned_center_grad))
    rebinned_error[intrinsic_bin_mask] = rebinned_center_grad[intrinsic_bin_mask]
    return rebinned_error


@njit(cache=True)
def rebinned_intrinsic_stat_error(rebinned_center, intrinsic_bin_mask):
    rebinned_error = np.zeros(np.shape(rebinned_center))
    rebinned_error[intrinsic_bin_mask] = rebinned_center[intrinsic_bin_mask]
    return rebinned_error


@njit(cache=True)
def apply_norm_grad(raw_center_grad, norm):
    shape = np.shape(raw_center_grad)[:-1] + (np.shape(raw_center_grad)[-1] + 1,)
    res = np.empty(shape)
    res[:, :-1] = raw_center_grad * norm
    res[:, -1] = raw_center_grad[:, 0]
    return res


@njit(cache=True)
def apply_norm_sq_grad(raw_center_sq_grad, norm):
    shape = np.shape(raw_center_grad)[:-1] + (np.shape(raw_center_grad)[-1] + 1,)
    res = np.empty(shape)
    res[:, :-1] = raw_center_grad * norm ** 2
    res[:, -1] = 2 * raw_center_grad[:, 0] * norm
    return res


@njit(cache=True)
def apply_norm(raw_center, norm):
    return raw_center * norm


@njit(cache=True)
def apply_norm_sq(raw_center_sq, norm):
    return raw_center_sq * norm ** 2


@njit(cache=True)
def expect(rebinned_expect, slices):
    res = binning.rebin_cv(rebinned_expect, slices)[:-1]
    return res


# @njit(cache=True)
@njit
def sys_cov(rebinned_sys_cov, mc_rebin_to_collapsed_slices):
    return binning.rebin_cov(rebinned_sys_cov, mc_rebin_to_collapsed_slices)


@njit(cache=True)
def osc_stat_error(rebinned_osc_stat_error, slices):
    error = binning.rebin_cv(rebinned_osc_stat_error, slices)
    return np.diag(error)


@njit(cache=True)
def stat_error(expect_grad):
    dim = len(expect_grad)
    shape = (dim, dim)
    matrix = np.zeros(shape)
    for i in range(dim):
        matrix[i, i] = expect_grad[i]
    return matrix

@njit(cache=True)
def stat_error_CNP(data,expect):
    dim = len(expect)
    shape = (dim, dim)
    matrix = np.zeros(shape)
    for i in range(dim):
        if expect[i] <= 0:
            matrix[i, i] = 0
        elif data[i] <= 0:
            matrix[i, i] = expect[i]/2.
        else:
            matrix[i, i] = 3. / (1. / data[i] + 2. / expect[i])
    return matrix


@njit(cache=True)
def osc_stat_error_grad(rebinned_osc_stat_error_grad, slices):
    error = binning.rebin_cv(rebinned_osc_stat_error_grad, slices)
    dim = len(error)
    shape = (dim, dim, np.shape(error)[1])
    matrix = np.zeros(shape)
    for i in range(dim):
        matrix[i, i, :] = error[i]
    return matrix


@njit(cache=True)
def stat_error_grad(expect_grad):
    dim = len(expect_grad)
    shape = (dim, dim, np.shape(expect_grad)[1])
    matrix = np.zeros(shape)
    for i in range(dim):
        matrix[i, i, :] = expect_grad[i]
    return matrix

@njit(cache=True)
def stat_error_CNP_grad(data,expect_grad):
    dim = len(expect_grad)
    shape = (dim, dim, np.shape(expect_grad)[1])
    matrix = np.zeros(shape)
    for i in range(dim):
        if expect_grad[i,0] <= 0:
            matrix[i, i, :] = 0
        elif data[i] <= 0:
            matrix[i, i, :] = expect_grad[i,:]/2.
        else:
            matrix[i, i, 0] = 3. / (1. / data[i] + 2. / expect_grad[i,0])
            matrix[i, i, 1:] = (6. / (expect_grad[i,0] / data[i] + 2.)**2) * expect_grad[i,1:]
    return matrix


@njit(cache=True)
def frac_cov(sys_cov, osc_stat_error, expect):
    return (sys_cov + osc_stat_error)[:-1, :-1] / expect[:, None] / expect[None, :]


# def cov_grad(sys_cov_grad, osc_stat_error_grad, intrinsic_stat_error_grad):
#    return sys_cov_grad + osc_stat_error_grad + intrinsic_stat_error_grad
@njit(cache=True)
def cov(sys_cov, osc_stat_error):
    return (sys_cov + osc_stat_error)[:-1, :-1]


@njit(cache=True)
def cov_add_error(sys_cov, stat_error, uB=False):
    if uB: return sys_cov + stat_error
    else: return (sys_cov)[:-1, :-1] + stat_error


# The inverse of the covariance matrix
@njit(cache=True)
def cov_inv(cov):
    cond = cov != 0
    nonzero = np.full(cov.shape[0], False)
    for i in range(cov.shape[0]):
        nonzero |= cond[i]
    idx = np.arange(len(nonzero))[nonzero]
    small_cov = np.empty((len(idx), len(idx)))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            small_cov[i, j] = cov[ii, jj]
    small_cov_inv = np.linalg.inv(small_cov)
    cov_inv = np.zeros(np.shape(cov))
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            cov_inv[ii, jj] = small_cov_inv[i, j]
    return cov_inv


@njit(cache=True)
def cov_inv_grad(cov_grad):
    cov = cov_grad[:, :, 0]
    return cov_inv(cov)


## Normalization prior


def norm_var_from_covs(cov_cv_map):
    frac_cov, cv = cov_cv_map["sub_norm"]
    return -frac_cov[0, 0]


@njit(cache=True)
def normal_prior_grad(x, mu, var):
    prefactor = -np.log(var) - 0.5 * (np.log(2) + np.log(np.pi))
    prefactor_d = 0.0
    exponent = -0.5 * (x - mu) ** 2 / var
    exponent_d = -(x - mu) / var
    res = np.empty((2,))
    res[0] = prefactor + exponent
    res[1] = prefactor_d + exponent_d
    return res


@njit(cache=True)
def normal_prior(x, mu, var):
    prefactor = -np.log(var) - 0.5 * (np.log(2) + np.log(np.pi))
    exponent = -0.5 * (x - mu) ** 2 / var
    return prefactor + exponent


@njit(cache=True)
def normal_priors_grad(x, mu, var):
    n = len(var)
    prefactor = np.sum(-np.log(var) - 0.5 * (np.log(2) + np.log(np.pi)))
    prefactor_d = 0.0
    exponent = np.sum(-0.5 * (x - mu) ** 2 / var)
    exponent_d = -(x - mu) / var
    res = np.empty((1 + n))
    res[0] = prefactor + exponent
    res[1:] = prefactor_d + exponent_d
    return res


@njit(cache=True)
def normal_priors(x, mu, var):
    prefactor = np.sum(-np.log(var) - 0.5 * (np.log(2) + np.log(np.pi)))
    exponent = np.sum(-0.5 * (x - mu) ** 2 / var)
    return prefactor + exponent


@njit(cache=True)
def eigen_decomp(matrix):
    lambdas, vectors = scipy.linalg.eigh(matrix)
    return lambdas, vectors


@njit(cache=True)
def eigen_nonzero_mask(lambdas):
    lambda_threshold = 1e-8
    return lambdas > lambda_threshold


@njit(cache=True)
def eigen_nonzero_lambdas(lambdas, nonzero_mask):
    return lambdas[nonzero_mask]


@njit(cache=True)
def eigen_nonzero_vectors(vectors, nonzero_mask):
    return vectors[:, nonzero_mask]


@njit(cache=True)
def eigen_zero_vector(vectors, nonzero_mask):
    return vectors[:, ~nonzero_mask][:, 0]


@njit(cache=True)
def eigen_count_nonzero(nonzero_mask):
    return np.count_nonzero(nonzero_mask)


@njit(cache=True)
def eigen_gauss_prefactor(nonzero_lambdas):
    n_params = len(nonzero_lambdas)
    cov_det = np.sum(np.log(nonzero_lambdas))
    res = -0.5 * (n_params * np.log(2 * np.pi) + cov_det)
    return res
