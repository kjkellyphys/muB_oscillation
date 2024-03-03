import numpy as np
from numba import njit

import analysis
import functions as f
import likelihood_grad
import autodiff as ad

import logging

import scipy

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.DEBUG)


@njit(cache=True)
def uB_params_from_subspace(sub_params, uB_zero_mode):
    res = np.zeros(len(sub_params) + 1)
    res[:-1] = sub_params
    res[-1] = -np.sum(sub_params * uB_zero_mode[:-1]) / uB_zero_mode[-1]
    return res


@njit(cache=True)
def uB_params_from_subspace_grad(sub_params_grad, uB_zero_mode):
    res = np.zeros((len(sub_params_grad) + 1,) + (np.shape(sub_params_grad)[1],))
    res[:-1] = sub_params_grad
    res[-1] = (
        -np.sum(sub_params_grad * np.expand_dims(uB_zero_mode[:-1], 1), axis=0)
        / uB_zero_mode[-1]
    )
    return res


@njit(cache=True)
def dl_unscaled_expect(
    r,
    theta,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_nue_mc_osc_sin2_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_ub_NuE_nue_constrained_cv_ratio,
):
    # ub stuff
    v_numu_to_nue_mc_osc_weights_ubbaseline = f.numu_to_nue_osc_weights(
        v_numu_to_nue_mc_osc_factor_ubbaseline,
        v_numu_to_nue_mc_osc_sin2_ubbaseline,
        r,
        theta,
    )
    v_nue_to_nue_mc_osc_weights_ubbaseline = f.nue_to_nue_osc_weights(
        v_nue_to_nue_mc_osc_factor_ubbaseline,
        v_nue_to_nue_mc_osc_sin2_ubbaseline,
        r,
        theta,
    )

    v_ub_numu_to_nue_mb_mc_weights_ubbaseline = f.mc_weights(
        v_ub_numu_to_nue_mb_mc_cv_weights, v_numu_to_nue_mc_osc_weights_ubbaseline
    )
    v_ub_nue_to_nue_mb_mc_weights_ubbaseline = f.mc_weights(
        v_ub_nue_to_nue_mb_mc_cv_weights, v_nue_to_nue_mc_osc_weights_ubbaseline
    )
    v_ub_numu_to_nue_mb_osc_expect = f.raw_center(
        v_ub_numu_to_nue_mb_mc_weights_ubbaseline, v_ub_numu_to_nue_mb_mc_slices
    )
    v_ub_nue_to_nue_mb_osc_expect = f.raw_center(
        v_ub_nue_to_nue_mb_mc_weights_ubbaseline, v_ub_nue_to_nue_mb_mc_slices
    )
    v_ub_numu_to_nue_osc_weight_map = f.ub_osc_weight_map(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_numu_to_nue_mb_osc_expect
    )
    v_ub_nue_to_nue_osc_weight_map = f.ub_osc_weight_map(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_nue_to_nue_mb_osc_expect
    )

    v_ub_NuE_numu_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * v_dl_NuE_nue_smearing_matrix,
        axis=1,
    )
    v_ub_NuE_nue_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0)
        * v_dl_NuE_nue_smearing_matrix,
        axis=1,
    )
    v_ub_NuE_numu_to_nue_constrained_expect = (
        v_ub_NuE_numu_to_nue_cv_expect * v_ub_NuE_nue_constrained_cv_ratio
    )
    v_ub_NuE_nue_to_nue_constrained_expect = (
        v_ub_NuE_nue_to_nue_cv_expect * v_ub_NuE_nue_constrained_cv_ratio
    )
    v_ub_unscaled_expect = f.ub_expect_nue(
        v_ub_NuE_nue_to_nue_constrained_expect,
        v_numu_fitted_bkg_dl_template,
        v_ub_NuE_numu_to_nue_constrained_expect,
    )
    return v_ub_unscaled_expect


@njit(cache=True)
def wc_unscaled_expect(
    r,
    theta,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_nue_mc_osc_sin2_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
):
    # ub stuff
    v_numu_to_nue_mc_osc_weights_ubbaseline = f.numu_to_nue_osc_weights(
        v_numu_to_nue_mc_osc_factor_ubbaseline,
        v_numu_to_nue_mc_osc_sin2_ubbaseline,
        r,
        theta,
    )
    v_nue_to_nue_mc_osc_weights_ubbaseline = f.nue_to_nue_osc_weights(
        v_nue_to_nue_mc_osc_factor_ubbaseline,
        v_nue_to_nue_mc_osc_sin2_ubbaseline,
        r,
        theta,
    )
    v_numu_to_numu_mc_osc_weights_ubbaseline = f.numu_to_numu_osc_weights(
        v_numu_to_numu_mc_osc_factor_ubbaseline,
        v_numu_to_numu_mc_osc_sin2_ubbaseline,
        r,
        theta,
    )

    v_ub_numu_to_nue_mb_mc_weights_ubbaseline = f.mc_weights(
        v_ub_numu_to_nue_mb_mc_cv_weights, v_numu_to_nue_mc_osc_weights_ubbaseline
    )
    v_ub_nue_to_nue_mb_mc_weights_ubbaseline = f.mc_weights(
        v_ub_nue_to_nue_mb_mc_cv_weights, v_nue_to_nue_mc_osc_weights_ubbaseline
    )
    v_ub_numu_to_numu_mb_mc_weights_ubbaseline = f.mc_weights(
        v_ub_numu_to_numu_mb_mc_cv_weights, v_numu_to_numu_mc_osc_weights_ubbaseline
    )
    v_ub_numu_to_nue_mb_osc_expect = f.raw_center(
        v_ub_numu_to_nue_mb_mc_weights_ubbaseline, v_ub_numu_to_nue_mb_mc_slices
    )
    v_ub_nue_to_nue_mb_osc_expect = f.raw_center(
        v_ub_nue_to_nue_mb_mc_weights_ubbaseline, v_ub_nue_to_nue_mb_mc_slices
    )
    v_ub_numu_to_numu_mb_osc_expect = f.raw_center(
        v_ub_numu_to_numu_mb_mc_weights_ubbaseline, v_ub_numu_to_numu_mb_mc_slices
    )
    v_ub_numu_to_nue_osc_weight_map = f.ub_osc_weight_map(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_numu_to_nue_mb_osc_expect
    )
    v_ub_nue_to_nue_osc_weight_map = f.ub_osc_weight_map(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_nue_to_nue_mb_osc_expect
    )
    v_ub_numu_to_numu_osc_weight_map = f.ub_osc_weight_map(
        v_ub_numu_to_numu_mb_cv_expect, v_ub_numu_to_numu_mb_osc_expect
    )

    # nue FC
    v_wc_nue_FC_numu_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * v_wc_nue_FC_smearing_matrix,
        axis=1,
    )
    v_wc_nue_FC_nue_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0) * v_wc_nue_FC_smearing_matrix,
        axis=1,
    )
    v_wc_nue_FC_numu_to_nue_constrained_expect = (
        v_wc_nue_FC_numu_to_nue_cv_expect * v_wc_nue_FC_DR_SM_ratio
    )
    v_wc_nue_FC_nue_to_nue_constrained_expect = (
        v_wc_nue_FC_nue_to_nue_cv_expect * v_wc_nue_FC_DR_SM_ratio
    )
    v_wc_nue_FC_unscaled_expect = f.ub_expect_nue(
        v_wc_nue_FC_nue_to_nue_constrained_expect,
        v_nue_FC_bkg_wc_template,
        v_wc_nue_FC_numu_to_nue_constrained_expect,
    )
    # nue PC
    v_wc_nue_PC_numu_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * v_wc_nue_PC_smearing_matrix,
        axis=1,
    )
    v_wc_nue_PC_nue_to_nue_cv_expect = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0) * v_wc_nue_PC_smearing_matrix,
        axis=1,
    )
    v_wc_nue_PC_numu_to_nue_constrained_expect = (
        v_wc_nue_PC_numu_to_nue_cv_expect * v_wc_nue_PC_DR_SM_ratio
    )
    v_wc_nue_PC_nue_to_nue_constrained_expect = (
        v_wc_nue_PC_nue_to_nue_cv_expect * v_wc_nue_PC_DR_SM_ratio
    )
    v_wc_nue_PC_unscaled_expect = f.ub_expect_nue(
        v_wc_nue_PC_nue_to_nue_constrained_expect,
        v_nue_PC_bkg_wc_template,
        v_wc_nue_PC_numu_to_nue_constrained_expect,
    )
    # numu FC
    v_wc_numu_FC_numu_to_numu_cv_expect = np.sum(
        np.expand_dims(v_ub_numu_to_numu_osc_weight_map, 0)
        * v_wc_numu_FC_smearing_matrix,
        axis=1,
    )
    v_wc_numu_FC_numu_to_numu_constrained_expect = (
        v_wc_numu_FC_numu_to_numu_cv_expect * v_wc_numu_FC_DR_SM_ratio
    )
    v_wc_numu_FC_unscaled_expect = f.ub_expect_numu(
        v_wc_numu_FC_numu_to_numu_constrained_expect,
        v_numu_FC_bkg_wc_template,
    )
    # numu PC
    v_wc_numu_PC_numu_to_numu_cv_expect = np.sum(
        np.expand_dims(v_ub_numu_to_numu_osc_weight_map, 0)
        * v_wc_numu_PC_smearing_matrix,
        axis=1,
    )
    v_wc_numu_PC_numu_to_numu_constrained_expect = (
        v_wc_numu_PC_numu_to_numu_cv_expect * v_wc_numu_PC_DR_SM_ratio
    )
    v_wc_numu_PC_unscaled_expect = f.ub_expect_numu(
        v_wc_numu_PC_numu_to_numu_constrained_expect,
        v_numu_PC_bkg_wc_template,
    )

    return f.wc_expect(
        v_wc_nue_FC_unscaled_expect,
        v_wc_nue_PC_unscaled_expect,
        v_wc_numu_FC_unscaled_expect,
        v_wc_numu_PC_unscaled_expect,
    )


@njit(cache=True)
def ub_likelihood(
    ub_data,
    # ub_norm,
    ub_params,
    ub_modes,
    ub_nonzero_eigen_values,
    v_ub_unscaled_expect,
    v_ub_gauss_likelihood_prefactor,
    v_ub_expect_frac_mc_error,
    ub_zero_eigen_vector,
):

    # v_ub_nominal_expect = f.apply_norm(v_ub_unscaled_expect, ub_norm)
    v_ub_nominal_expect = v_ub_unscaled_expect

    ub_scale = ub_params * ub_nonzero_eigen_values
    full_ub_diff_ratio = np.sum(np.expand_dims(ub_scale, 0) * ub_modes, axis=1)

    full_ub_ratio = full_ub_diff_ratio + 1.0

    ub_exponent = f.normal_priors(
        ub_scale,
        0.0,
        ub_nonzero_eigen_values,
    )

    ub_sys_expect = f.ub_apply_sys_norms(full_ub_ratio, v_ub_nominal_expect)

    alpha = 0.01
    boundary = alpha * v_ub_nominal_expect
    modified_expect = boundary * np.exp(ub_sys_expect / boundary - 1)
    ub_sys_expect = np.where(
        full_ub_ratio < alpha, modified_expect, ub_sys_expect
    )

    ub_expect_mc_error = f.ub_mc_var_from_frac_error(
        ub_sys_expect, v_ub_expect_frac_mc_error
    )
    ub_leff = likelihood_grad.LEff(ub_data, ub_sys_expect, ub_expect_mc_error)

    ub_like = -(v_ub_gauss_likelihood_prefactor + ub_exponent + np.sum(ub_leff))
    return ub_like


@njit(cache=True)
def mb_likelihood(
    data,
    # norm,
    v_rebinned_center,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
):
    # v_rebinned_expect = f.apply_norm(v_rebinned_center, norm)
    v_rebinned_expect = v_rebinned_center

    v_expect = f.expect(v_rebinned_expect, v_rebin_to_collapsed_slices)
    v_rebinned_sys_cov = f.mul_cov(v_rebinned_sys_frac_cov, v_rebinned_expect)
    v_sys_cov = f.sys_cov(v_rebinned_sys_cov, v_rebin_to_collapsed_slices)

    v_stat_error = f.stat_error(v_expect)
    v_cov = f.cov_add_error(v_sys_cov, v_stat_error)
    v_cov_inv = f.cov_inv(v_cov)
    v_gauss_likelihood_prefactor = f.gauss_prefactor(v_cov)
    diff = data - v_expect
    exponent = f.gauss_exponent(diff, v_cov_inv)

    like = -(v_gauss_likelihood_prefactor + exponent)

    return like

@njit(cache=True)
def ub_covCNP(
    data,
    v_expect,
    v_sys_frac_cov,
):
    v_sys_cov = f.mul_cov(
        v_sys_frac_cov, v_expect
    )
    v_stat_error = f.stat_error_CNP(data,v_expect)
    v_cov = f.cov_add_error(v_sys_cov, v_stat_error, uB=True)
    v_cov_inv = f.cov_inv(v_cov)
    v_gauss_likelihood_prefactor = f.gauss_prefactor(v_cov)
    diff = data-v_expect
    exponent = f.gauss_exponent(diff,v_cov_inv)
    print(v_gauss_likelihood_prefactor,exponent)
    like = -(v_gauss_likelihood_prefactor + exponent)

    return like


@njit(cache=True)
def mb_2nu_prereqs(
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2,
    v_numubar_to_nuebar_mc_osc_sin2,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
):
    v_numu_to_nue_mc_osc_weights = f.numu_to_nue_osc_weights(
        v_numu_to_nue_mc_osc_factor, v_numu_to_nue_mc_osc_sin2, r, theta
    )
    v_numu_to_nue_mc_weights = f.mc_weights(
        v_numu_to_nue_mc_cv_weights, v_numu_to_nue_mc_osc_weights
    )
    v_numu_to_nue_mc_raw_center = f.raw_center(
        v_numu_to_nue_mc_weights, v_numu_to_nue_mc_slices
    )

    v_nue_to_nue_mc_raw_center = f.raw_center(
        v_nue_to_nue_mc_cv_weights, v_nue_to_nue_mc_slices
    )

    v_numu_to_numu_mc_raw_center = f.raw_center(
        v_numu_to_numu_mc_cv_weights, v_numu_to_numu_mc_slices
    )

    v_numubar_to_nuebar_mc_osc_weights = f.numu_to_nue_osc_weights(
        v_numubar_to_nuebar_mc_osc_factor, v_numubar_to_nuebar_mc_osc_sin2, r, theta
    )
    v_numubar_to_nuebar_mc_weights = f.mc_weights(
        v_numubar_to_nuebar_mc_cv_weights, v_numubar_to_nuebar_mc_osc_weights
    )
    v_numubar_to_nuebar_mc_raw_center = f.raw_center(
        v_numubar_to_nuebar_mc_weights, v_numubar_to_nuebar_mc_slices
    )

    v_nuebar_to_nuebar_mc_raw_center = f.raw_center(
        v_nuebar_to_nuebar_mc_cv_weights, v_nuebar_to_nuebar_mc_slices
    )

    v_numubar_to_numubar_mc_raw_center = f.raw_center(
        v_numubar_to_numubar_mc_cv_weights, v_numubar_to_numubar_mc_slices
    )

    v_raw_center = (
        v_numu_to_nue_mc_raw_center
        + v_nue_to_nue_mc_raw_center
        + v_numu_to_numu_mc_raw_center
        + v_numubar_to_nuebar_mc_raw_center
        + v_nuebar_to_nuebar_mc_raw_center
        + v_numubar_to_numubar_mc_raw_center
    )

    v_rebinned_center = f.rebinned_center(v_raw_center, v_raw_to_rebin_slices)

    return v_rebinned_center, v_rebin_to_collapsed_slices, v_rebinned_sys_frac_cov


@njit(cache=True)
def mb_prereqs(
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2,
    v_nue_to_nue_mc_osc_sin2,
    v_numu_to_numu_mc_osc_sin2,
    v_numubar_to_nuebar_mc_osc_sin2,
    v_nuebar_to_nuebar_mc_osc_sin2,
    v_numubar_to_numubar_mc_osc_sin2,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
):
    v_numu_to_nue_mc_osc_weights = f.numu_to_nue_osc_weights(
        v_numu_to_nue_mc_osc_factor, v_numu_to_nue_mc_osc_sin2, r, theta
    )
    v_numu_to_nue_mc_weights = f.mc_weights(
        v_numu_to_nue_mc_cv_weights, v_numu_to_nue_mc_osc_weights
    )
    v_numu_to_nue_mc_raw_center = f.raw_center(
        v_numu_to_nue_mc_weights, v_numu_to_nue_mc_slices
    )

    v_nue_to_nue_mc_osc_weights = f.nue_to_nue_osc_weights(
        v_nue_to_nue_mc_osc_factor, v_nue_to_nue_mc_osc_sin2, r, theta
    )
    v_nue_to_nue_mc_weights = f.mc_weights(
        v_nue_to_nue_mc_cv_weights, v_nue_to_nue_mc_osc_weights
    )
    v_nue_to_nue_mc_raw_center = f.raw_center(
        v_nue_to_nue_mc_weights, v_nue_to_nue_mc_slices
    )

    v_numu_to_numu_mc_osc_weights = f.numu_to_numu_osc_weights(
        v_numu_to_numu_mc_osc_factor, v_numu_to_numu_mc_osc_sin2, r, theta
    )
    v_numu_to_numu_mc_weights = f.mc_weights(
        v_numu_to_numu_mc_cv_weights, v_numu_to_numu_mc_osc_weights
    )
    v_numu_to_numu_mc_raw_center = f.raw_center(
        v_numu_to_numu_mc_weights, v_numu_to_numu_mc_slices
    )

    v_numubar_to_nuebar_mc_osc_weights = f.numu_to_nue_osc_weights(
        v_numubar_to_nuebar_mc_osc_factor, v_numubar_to_nuebar_mc_osc_sin2, r, theta
    )
    v_numubar_to_nuebar_mc_weights = f.mc_weights(
        v_numubar_to_nuebar_mc_cv_weights, v_numubar_to_nuebar_mc_osc_weights
    )
    v_numubar_to_nuebar_mc_raw_center = f.raw_center(
        v_numubar_to_nuebar_mc_weights, v_numubar_to_nuebar_mc_slices
    )

    v_nuebar_to_nuebar_mc_osc_weights = f.nue_to_nue_osc_weights(
        v_nuebar_to_nuebar_mc_osc_factor, v_nuebar_to_nuebar_mc_osc_sin2, r, theta
    )
    v_nuebar_to_nuebar_mc_weights = f.mc_weights(
        v_nuebar_to_nuebar_mc_cv_weights, v_nuebar_to_nuebar_mc_osc_weights
    )
    v_nuebar_to_nuebar_mc_raw_center = f.raw_center(
        v_nuebar_to_nuebar_mc_weights, v_nuebar_to_nuebar_mc_slices
    )

    v_numubar_to_numubar_mc_osc_weights = f.numu_to_numu_osc_weights(
        v_numubar_to_numubar_mc_osc_factor, v_numubar_to_numubar_mc_osc_sin2, r, theta
    )
    v_numubar_to_numubar_mc_weights = f.mc_weights(
        v_numubar_to_numubar_mc_cv_weights, v_numubar_to_numubar_mc_osc_weights
    )
    v_numubar_to_numubar_mc_raw_center = f.raw_center(
        v_numubar_to_numubar_mc_weights, v_numubar_to_numubar_mc_slices
    )

    v_raw_center = (
        v_numu_to_nue_mc_raw_center
        + v_nue_to_nue_mc_raw_center
        + v_numu_to_numu_mc_raw_center
        + v_numubar_to_nuebar_mc_raw_center
        + v_nuebar_to_nuebar_mc_raw_center
        + v_numubar_to_numubar_mc_raw_center
    )

    v_rebinned_center = f.rebinned_center(v_raw_center, v_raw_to_rebin_slices)

    return v_rebinned_center, v_rebin_to_collapsed_slices, v_rebinned_sys_frac_cov


@njit(cache=True)
def joint_nllh(
    data,
    ub_data,
    ub_mode,
    ub_fit_mode,
    ub_sys_frac_cov,
    r,
    theta,
    # norm,
    # ub_norm,
    ub_params,
    dl_modes,
    dl_nonzero_eigen_values,
    wc_modes,
    wc_nonzero_eigen_values,
    v_numu_to_nue_mc_osc_sin2,
    v_nue_to_nue_mc_osc_sin2,
    v_numu_to_numu_mc_osc_sin2,
    v_numubar_to_nuebar_mc_osc_sin2,
    v_nuebar_to_nuebar_mc_osc_sin2,
    v_numubar_to_numubar_mc_osc_sin2,
    v_nue_to_nue_mc_osc_sin2_ubbaseline,
    v_numu_to_nue_mc_osc_sin2_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_ubbaseline,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_nue_constrained_bkg_dl_template,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_dl_NuE_nue_constrained_cv_ratio,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
    v_dl_gauss_likelihood_prefactor,
    v_dl_expect_frac_mc_error,
    v_wc_gauss_likelihood_prefactor,
    v_wc_expect_frac_mc_error,
    # v_norm_var,
    # v_ub_norm_var,
    dl_zero_eigen_vector,
    wc_zero_eigen_vector,
):

    (
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_prereqs(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2,
        v_nue_to_nue_mc_osc_sin2,
        v_numu_to_numu_mc_osc_sin2,
        v_numubar_to_nuebar_mc_osc_sin2,
        v_nuebar_to_nuebar_mc_osc_sin2,
        v_numubar_to_numubar_mc_osc_sin2,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_osc_factor,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_osc_factor,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_osc_factor,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_osc_factor,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood(
        data,
        # norm,
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    if ub_mode == "DL":
        v_ub_unscaled_expect = dl_unscaled_expect(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_dl_NuE_nue_smearing_matrix,
            v_numu_fitted_bkg_dl_template,
            v_dl_NuE_nue_constrained_cv_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like = ub_likelihood(
                ub_data,
                ub_params,
                dl_modes,
                dl_nonzero_eigen_values,
                v_ub_unscaled_expect,
                v_dl_gauss_likelihood_prefactor,
                v_dl_expect_frac_mc_error,
                dl_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like = ub_covCNP(
                ub_data,
                v_ub_unscaled_expect,
                ub_sys_frac_cov,
            )
    elif ub_mode == "WC":
        v_ub_unscaled_expect = wc_unscaled_expect(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_ubbaseline,
            v_numu_to_numu_mc_osc_factor_ubbaseline,
            v_numu_to_numu_mc_osc_sin2_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_numu_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_numu_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_ub_numu_to_numu_mb_cv_expect,
            v_wc_nue_FC_smearing_matrix,
            v_wc_nue_PC_smearing_matrix,
            v_wc_numu_FC_smearing_matrix,
            v_wc_numu_PC_smearing_matrix,
            v_nue_FC_bkg_wc_template,
            v_nue_PC_bkg_wc_template,
            v_numu_FC_bkg_wc_template,
            v_numu_PC_bkg_wc_template,
            v_wc_nue_FC_DR_SM_ratio,
            v_wc_nue_PC_DR_SM_ratio,
            v_wc_numu_FC_DR_SM_ratio,
            v_wc_numu_PC_DR_SM_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like = ub_likelihood(
                ub_data,
                ub_params,
                wc_modes,
                wc_nonzero_eigen_values,
                v_ub_unscaled_expect,
                v_wc_gauss_likelihood_prefactor,
                v_wc_expect_frac_mc_error,
                wc_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like = ub_covCNP(
                ub_data,
                v_ub_unscaled_expect,
                ub_sys_frac_cov,
            )

    # norm_prior = -f.normal_prior(norm, 1.0, v_norm_var)
    # ub_norm_prior = -f.normal_prior(ub_norm, 1.0, v_ub_norm_var)

    joint_like = mb_like + ub_like  # + norm_prior + ub_norm_prior
    return joint_like


@njit(cache=True)
def mb_2nu_nllh(
    data,
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2,
    v_numubar_to_nuebar_mc_osc_sin2,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    # v_norm_var,
):

    (
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_2nu_prereqs(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2,
        v_numubar_to_nuebar_mc_osc_sin2,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood(
        data,
        # norm,
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    # norm_prior = -f.normal_prior(norm, 1.0, v_norm_var)

    like_like = mb_like  # + norm_prior
    return like_like


@njit(cache=True)
def mb_nllh(
    data,
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2,
    v_nue_to_nue_mc_osc_sin2,
    v_numu_to_numu_mc_osc_sin2,
    v_numubar_to_nuebar_mc_osc_sin2,
    v_nuebar_to_nuebar_mc_osc_sin2,
    v_numubar_to_numubar_mc_osc_sin2,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    # v_norm_var,
):

    (
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_prereqs(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2,
        v_nue_to_nue_mc_osc_sin2,
        v_numu_to_numu_mc_osc_sin2,
        v_numubar_to_nuebar_mc_osc_sin2,
        v_nuebar_to_nuebar_mc_osc_sin2,
        v_numubar_to_numubar_mc_osc_sin2,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_osc_factor,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_osc_factor,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_osc_factor,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_osc_factor,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood(
        data,
        # norm,
        v_rebinned_center,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    # norm_prior = -f.normal_prior(norm, 1.0, v_norm_var)

    like_like = mb_like  # + norm_prior
    return like_like


@njit(cache=True)
def ub_nllh(
    ub_data,
    ub_mode,
    ub_fit_mode,
    ub_sys_frac_cov,
    r,
    theta,
    # ub_norm,
    ub_params,
    dl_modes,
    dl_nonzero_eigen_values,
    wc_modes,
    wc_nonzero_eigen_values,
    v_numu_to_nue_mc_osc_sin2_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_ubbaseline,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_nue_constrained_bkg_dl_template,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_dl_NuE_nue_constrained_cv_ratio,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
    v_dl_gauss_likelihood_prefactor,
    v_dl_expect_frac_mc_error,
    v_wc_gauss_likelihood_prefactor,
    v_wc_expect_frac_mc_error,
    # v_norm_var,
    # v_ub_norm_var,
    dl_zero_eigen_vector,
    wc_zero_eigen_vector,
):

    if ub_mode == "DL":
        v_dl_unscaled_expect = dl_unscaled_expect(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_dl_NuE_nue_smearing_matrix,
            v_numu_fitted_bkg_dl_template,
            v_dl_NuE_nue_constrained_cv_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like = ub_likelihood(
                ub_data,
                ub_params,
                dl_modes,
                dl_nonzero_eigen_values,
                v_dl_unscaled_expect,
                v_dl_gauss_likelihood_prefactor,
                v_dl_expect_frac_mc_error,
                dl_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like = ub_covCNP(
                ub_data,
                v_dl_unscaled_expect,
                ub_sys_frac_cov,
            )
    elif ub_mode == "WC":
        v_wc_unscaled_expect = wc_unscaled_expect(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_ubbaseline,
            v_numu_to_numu_mc_osc_factor_ubbaseline,
            v_numu_to_numu_mc_osc_sin2_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_numu_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_numu_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_ub_numu_to_numu_mb_cv_expect,
            v_wc_nue_FC_smearing_matrix,
            v_wc_nue_PC_smearing_matrix,
            v_wc_numu_FC_smearing_matrix,
            v_wc_numu_PC_smearing_matrix,
            v_nue_FC_bkg_wc_template,
            v_nue_PC_bkg_wc_template,
            v_numu_FC_bkg_wc_template,
            v_numu_PC_bkg_wc_template,
            v_wc_nue_FC_DR_SM_ratio,
            v_wc_nue_PC_DR_SM_ratio,
            v_wc_numu_FC_DR_SM_ratio,
            v_wc_numu_PC_DR_SM_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like = ub_likelihood(
                ub_data,
                ub_params,
                wc_modes,
                wc_nonzero_eigen_values,
                v_wc_unscaled_expect,
                v_wc_gauss_likelihood_prefactor,
                v_wc_expect_frac_mc_error,
                wc_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like = ub_covCNP(
                ub_data,
                v_wc_unscaled_expect,
                ub_sys_frac_cov,
            )

    like_like = ub_like  # + ub_norm_prior
    return like_like


@njit(cache=True)
def dl_unscaled_expect_grad(
    r,
    theta,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_ub_NuE_nue_constrained_cv_ratio,
):
    # ub stuff
    v_numu_to_nue_mc_osc_weights_grad_ubbaseline = f.numu_to_nue_osc_weights_grad(
        v_numu_to_nue_mc_osc_factor_ubbaseline,
        v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
        r,
        theta,
    )
    v_nue_to_nue_mc_osc_weights_grad_ubbaseline = f.nue_to_nue_osc_weights_grad(
        v_nue_to_nue_mc_osc_factor_ubbaseline,
        v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
        r,
        theta,
    )
    v_ub_numu_to_nue_mb_mc_weights_grad_ubbaseline = f.mc_weights_grad(
        v_ub_numu_to_nue_mb_mc_cv_weights, v_numu_to_nue_mc_osc_weights_grad_ubbaseline
    )
    v_ub_nue_to_nue_mb_mc_weights_grad_ubbaseline = f.mc_weights_grad(
        v_ub_nue_to_nue_mb_mc_cv_weights, v_nue_to_nue_mc_osc_weights_grad_ubbaseline
    )
    v_ub_numu_to_nue_mb_osc_expect_grad = f.raw_center_grad(
        v_ub_numu_to_nue_mb_mc_weights_grad_ubbaseline, v_ub_numu_to_nue_mb_mc_slices
    )
    v_ub_nue_to_nue_mb_osc_expect_grad = f.raw_center_grad(
        v_ub_nue_to_nue_mb_mc_weights_grad_ubbaseline, v_ub_nue_to_nue_mb_mc_slices
    )
    v_ub_numu_to_nue_osc_weight_map = f.ub_osc_weight_map_grad(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_numu_to_nue_mb_osc_expect_grad
    )
    v_ub_nue_to_nue_osc_weight_map = f.ub_osc_weight_map_grad(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_nue_to_nue_mb_osc_expect_grad
    )

    v_ub_NuE_numu_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_dl_NuE_nue_smearing_matrix, 2),
        axis=1,
    )
    v_ub_NuE_nue_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_dl_NuE_nue_smearing_matrix, 2),
        axis=1,
    )
    v_ub_NuE_numu_to_nue_constrained_expect_grad = (
        v_ub_NuE_numu_to_nue_cv_expect_grad
        * np.expand_dims(v_ub_NuE_nue_constrained_cv_ratio, 1)
    )
    v_ub_NuE_nue_to_nue_constrained_expect_grad = (
        v_ub_NuE_nue_to_nue_cv_expect_grad
        * np.expand_dims(v_ub_NuE_nue_constrained_cv_ratio, 1)
    )
    v_ub_unscaled_expect_grad = f.ub_expect_nue_grad(
        v_ub_NuE_nue_to_nue_constrained_expect_grad,
        v_numu_fitted_bkg_dl_template,
        v_ub_NuE_numu_to_nue_constrained_expect_grad,
    )

    return v_ub_unscaled_expect_grad


@njit(cache=True)
def wc_unscaled_expect_grad(
    r,
    theta,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
):
    # ub stuff
    v_numu_to_nue_mc_osc_weights_grad_ubbaseline = f.numu_to_nue_osc_weights_grad(
        v_numu_to_nue_mc_osc_factor_ubbaseline,
        v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
        r,
        theta,
    )
    v_nue_to_nue_mc_osc_weights_grad_ubbaseline = f.nue_to_nue_osc_weights_grad(
        v_nue_to_nue_mc_osc_factor_ubbaseline,
        v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
        r,
        theta,
    )
    v_numu_to_numu_mc_osc_weights_grad_ubbaseline = f.numu_to_numu_osc_weights_grad(
        v_numu_to_numu_mc_osc_factor_ubbaseline,
        v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
        r,
        theta,
    )
    v_ub_numu_to_nue_mb_mc_weights_grad_ubbaseline = f.mc_weights_grad(
        v_ub_numu_to_nue_mb_mc_cv_weights, v_numu_to_nue_mc_osc_weights_grad_ubbaseline
    )
    v_ub_nue_to_nue_mb_mc_weights_grad_ubbaseline = f.mc_weights_grad(
        v_ub_nue_to_nue_mb_mc_cv_weights, v_nue_to_nue_mc_osc_weights_grad_ubbaseline
    )
    v_ub_numu_to_numu_mb_mc_weights_grad_ubbaseline = f.mc_weights_grad(
        v_ub_numu_to_numu_mb_mc_cv_weights,
        v_numu_to_numu_mc_osc_weights_grad_ubbaseline,
    )
    v_ub_numu_to_nue_mb_osc_expect_grad = f.raw_center_grad(
        v_ub_numu_to_nue_mb_mc_weights_grad_ubbaseline, v_ub_numu_to_nue_mb_mc_slices
    )
    v_ub_nue_to_nue_mb_osc_expect_grad = f.raw_center_grad(
        v_ub_nue_to_nue_mb_mc_weights_grad_ubbaseline, v_ub_nue_to_nue_mb_mc_slices
    )
    v_ub_numu_to_numu_mb_osc_expect_grad = f.raw_center_grad(
        v_ub_numu_to_numu_mb_mc_weights_grad_ubbaseline, v_ub_numu_to_numu_mb_mc_slices
    )
    v_ub_numu_to_nue_osc_weight_map = f.ub_osc_weight_map_grad(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_numu_to_nue_mb_osc_expect_grad
    )
    v_ub_nue_to_nue_osc_weight_map = f.ub_osc_weight_map_grad(
        v_ub_nue_to_nue_mb_cv_expect, v_ub_nue_to_nue_mb_osc_expect_grad
    )
    v_ub_numu_to_numu_osc_weight_map = f.ub_osc_weight_map_grad(
        v_ub_numu_to_numu_mb_cv_expect, v_ub_numu_to_numu_mb_osc_expect_grad
    )

    # nue FC
    v_wc_nue_FC_numu_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_wc_nue_FC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_nue_FC_nue_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_wc_nue_FC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_nue_FC_numu_to_nue_constrained_expect_grad = (
        v_wc_nue_FC_numu_to_nue_cv_expect_grad
        * np.expand_dims(v_wc_nue_FC_DR_SM_ratio, 1)
    )
    v_wc_nue_FC_nue_to_nue_constrained_expect_grad = (
        v_wc_nue_FC_nue_to_nue_cv_expect_grad
        * np.expand_dims(v_wc_nue_FC_DR_SM_ratio, 1)
    )
    v_wc_nue_FC_unscaled_expect_grad = f.ub_expect_nue_grad(
        v_wc_nue_FC_nue_to_nue_constrained_expect_grad,
        v_nue_FC_bkg_wc_template,
        v_wc_nue_FC_numu_to_nue_constrained_expect_grad,
    )
    # nue PC
    v_wc_nue_PC_numu_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_numu_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_wc_nue_PC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_nue_PC_nue_to_nue_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_nue_to_nue_osc_weight_map, 0)
        * np.expand_dims(v_wc_nue_PC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_nue_PC_numu_to_nue_constrained_expect_grad = (
        v_wc_nue_PC_numu_to_nue_cv_expect_grad
        * np.expand_dims(v_wc_nue_PC_DR_SM_ratio, 1)
    )
    v_wc_nue_PC_nue_to_nue_constrained_expect_grad = (
        v_wc_nue_PC_nue_to_nue_cv_expect_grad
        * np.expand_dims(v_wc_nue_PC_DR_SM_ratio, 1)
    )
    v_wc_nue_PC_unscaled_expect_grad = f.ub_expect_nue_grad(
        v_wc_nue_PC_nue_to_nue_constrained_expect_grad,
        v_nue_PC_bkg_wc_template,
        v_wc_nue_PC_numu_to_nue_constrained_expect_grad,
    )
    # numu FC
    v_wc_numu_FC_numu_to_numu_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_numu_to_numu_osc_weight_map, 0)
        * np.expand_dims(v_wc_numu_FC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_numu_FC_numu_to_numu_constrained_expect_grad = (
        v_wc_numu_FC_numu_to_numu_cv_expect_grad
        * np.expand_dims(v_wc_numu_FC_DR_SM_ratio, 1)
    )
    v_wc_numu_FC_unscaled_expect_grad = f.ub_expect_numu_grad(
        v_wc_numu_FC_numu_to_numu_constrained_expect_grad,
        v_numu_FC_bkg_wc_template,
    )
    # numu PC
    v_wc_numu_PC_numu_to_numu_cv_expect_grad = np.sum(
        np.expand_dims(v_ub_numu_to_numu_osc_weight_map, 0)
        * np.expand_dims(v_wc_numu_PC_smearing_matrix, 2),
        axis=1,
    )
    v_wc_numu_PC_numu_to_numu_constrained_expect_grad = (
        v_wc_numu_PC_numu_to_numu_cv_expect_grad
        * np.expand_dims(v_wc_numu_PC_DR_SM_ratio, 1)
    )
    v_wc_numu_PC_unscaled_expect_grad = f.ub_expect_numu_grad(
        v_wc_numu_PC_numu_to_numu_constrained_expect_grad,
        v_numu_PC_bkg_wc_template,
    )

    return f.wc_expect_grad(
        v_wc_nue_FC_unscaled_expect_grad,
        v_wc_nue_PC_unscaled_expect_grad,
        v_wc_numu_FC_unscaled_expect_grad,
        v_wc_numu_PC_unscaled_expect_grad,
    )


@njit(cache=True)
def ub_likelihood_grad(
    ub_data,
    # ub_norm,
    ub_params,
    ub_modes,
    ub_nonzero_eigen_values,
    v_ub_unscaled_expect_grad,
    v_ub_gauss_likelihood_prefactor,
    v_ub_expect_frac_mc_error,
    ub_zero_eigen_vector,
):
    # v_ub_nominal_expect_grad = f.apply_norm_grad(v_ub_unscaled_expect_grad, ub_norm)
    v_ub_nominal_expect_grad = v_ub_unscaled_expect_grad

    # full_ub_diff_ratio_grad = np.zeros((len(ub_params) + 1, 1 + len(ub_params)))
    full_ub_diff_ratio_grad = np.zeros((len(ub_data), len(ub_params) + 1))
    ub_scale = ub_params * ub_nonzero_eigen_values
    ub_diff = np.sum(np.expand_dims(ub_scale, 0) * ub_modes, axis=1)
    full_ub_diff_ratio_grad[:, 0] = ub_diff
    full_ub_diff_ratio_grad[:, 1:] = ub_modes

    full_ub_ratio_grad = np.copy(full_ub_diff_ratio_grad)
    full_ub_ratio_grad[:, 0] += 1.0

    expect_shape = np.shape(v_ub_nominal_expect_grad)
    tot_shape = expect_shape[:-1] + (expect_shape[-1] + len(ub_params),)

    ub_exponent = f.normal_priors_grad(
        ub_scale,
        0.0,
        ub_nonzero_eigen_values,
    )
    ub_exponent_extended = np.zeros(tot_shape[-1])
    ub_exponent_extended[0] = ub_exponent[0]
    ub_exponent_extended[expect_shape[-1] :] = ub_exponent[1:]

    ub_sys_expect_grad = f.ub_apply_sys_norms_grad(
        v_ub_nominal_expect_grad, full_ub_ratio_grad
    )
    alpha = 0.01
    boundary = alpha * v_ub_nominal_expect_grad[:, 0]
    modified_expect = boundary * np.exp(ub_sys_expect_grad[:, 0] / boundary - 1)
    modified_grad = np.exp(ub_sys_expect_grad[:, 0] / boundary - 1)
    ub_sys_expect_grad[:, 0] = np.where(
        full_ub_ratio_grad[:, 0] < alpha, modified_expect, ub_sys_expect_grad[:, 0]
    )
    n_expect_grad = np.shape(v_ub_nominal_expect_grad)[1] - 1
    boundary_mask = np.empty((len(v_ub_nominal_expect_grad[:, 0]), len(ub_params)))
    boundary_mask[:, :] = np.expand_dims(full_ub_ratio_grad[:, 0], 1) < alpha
    ub_sys_expect_grad[:, n_expect_grad + 1 :] = np.where(
        boundary_mask,
        ub_sys_expect_grad[:, n_expect_grad + 1 :] * np.expand_dims(modified_grad, 1),
        ub_sys_expect_grad[:, n_expect_grad + 1 :],
    )

    ub_expect_mc_error_grad = f.ub_mc_var_from_frac_error_grad(
        ub_sys_expect_grad, v_ub_expect_frac_mc_error
    )
    ub_leff, ub_leff_d = likelihood_grad.LEff_grad(
        ub_data, ad.unpack(ub_sys_expect_grad), ad.unpack(ub_expect_mc_error_grad)
    )
    ub_leff_grad = np.empty((len(ub_leff), np.shape(ub_leff_d)[-1] + 1))
    ub_leff_grad[:, 0] = ub_leff
    ub_leff_grad[:, 1:] = ub_leff_d

    ub_like = -(ub_exponent_extended + np.sum(ub_leff_grad, axis=0))
    ub_like[0] -= v_ub_gauss_likelihood_prefactor
    return ub_like


@njit(cache=True)
def mb_likelihood_grad(
    data,
    # norm,
    v_rebinned_center_grad,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
):
    # v_rebinned_expect_grad = f.apply_norm_grad(v_rebinned_center_grad, norm)
    v_rebinned_expect_grad = v_rebinned_center_grad
    v_expect_grad = f.expect(v_rebinned_expect_grad, v_rebin_to_collapsed_slices)
    v_rebinned_sys_cov_grad = f.mul_const_cov_grad(
        v_rebinned_sys_frac_cov, v_rebinned_expect_grad
    )
    v_sys_cov_grad = f.sys_cov(v_rebinned_sys_cov_grad, v_rebin_to_collapsed_slices)
    v_stat_error_grad = f.stat_error_grad(v_expect_grad)
    v_cov_grad = f.cov_add_error(v_sys_cov_grad, v_stat_error_grad)
    v_cov_inv = f.cov_inv_grad(v_cov_grad)
    v_gauss_likelihood_prefactor = f.gauss_prefactor_grad(v_cov_grad, v_cov_inv)
    diff_grad = -v_expect_grad
    diff_grad[:, 0] += data
    exponent = f.gauss_exponent_grad(diff_grad, v_cov_grad, v_cov_inv)

    like = -(v_gauss_likelihood_prefactor + exponent)

    return like

@njit(cache=True)
def ub_covCNP_grad(
    data,
    v_expect_grad,
    v_sys_frac_cov,
):
    v_sys_cov_grad = f.mul_const_cov_grad(
        v_sys_frac_cov, v_expect_grad
    )
    v_stat_error_grad = f.stat_error_CNP_grad(data,v_expect_grad)
    v_cov_grad = f.cov_add_error(v_sys_cov_grad, v_stat_error_grad, uB=True)
    v_cov_inv = f.cov_inv_grad(v_cov_grad)
    v_gauss_likelihood_prefactor = f.gauss_prefactor_grad(v_cov_grad, v_cov_inv)
    diff_grad = -v_expect_grad
    diff_grad[:, 0] += data
    exponent = f.gauss_exponent_grad(diff_grad, v_cov_grad, v_cov_inv)
    like = -(v_gauss_likelihood_prefactor + exponent)

    return like


@njit(cache=True)
def mb_2nu_prereqs_grad(
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2_grad,
    v_numubar_to_nuebar_mc_osc_sin2_grad,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
):
    v_numu_to_nue_mc_osc_weights_grad = f.numu_to_nue_osc_weights_grad(
        v_numu_to_nue_mc_osc_factor, v_numu_to_nue_mc_osc_sin2_grad, r, theta
    )
    v_numu_to_nue_mc_weights_grad = f.mc_weights_grad(
        v_numu_to_nue_mc_cv_weights, v_numu_to_nue_mc_osc_weights_grad
    )
    v_numu_to_nue_mc_raw_center_grad = f.raw_center_grad(
        v_numu_to_nue_mc_weights_grad, v_numu_to_nue_mc_slices
    )

    v_nue_to_nue_mc_raw_center = f.raw_center(
        v_nue_to_nue_mc_cv_weights, v_nue_to_nue_mc_slices
    )

    v_numu_to_numu_mc_raw_center = f.raw_center_grad(
        v_numu_to_numu_mc_cv_weights, v_numu_to_numu_mc_slices
    )

    v_numubar_to_nuebar_mc_osc_weights_grad = f.numu_to_nue_osc_weights_grad(
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_osc_sin2_grad,
        r,
        theta,
    )
    v_numubar_to_nuebar_mc_weights_grad = f.mc_weights_grad(
        v_numubar_to_nuebar_mc_cv_weights, v_numubar_to_nuebar_mc_osc_weights_grad
    )
    v_numubar_to_nuebar_mc_raw_center_grad = f.raw_center_grad(
        v_numubar_to_nuebar_mc_weights_grad, v_numubar_to_nuebar_mc_slices
    )

    v_nuebar_to_nuebar_mc_raw_center = f.raw_center_grad(
        v_nuebar_to_nuebar_mc_cv_weights, v_nuebar_to_nuebar_mc_slices
    )

    v_numubar_to_numubar_mc_raw_center = f.raw_center_grad(
        v_numubar_to_numubar_mc_cv_weights, v_numubar_to_numubar_mc_slices
    )

    v_raw_center = (
        v_nue_to_nue_mc_raw_center
        + v_numu_to_numu_mc_raw_center
        + v_nuebar_to_nuebar_mc_raw_center
        + v_numubar_to_numubar_mc_raw_center
    )
    v_raw_center_grad = (
        v_numu_to_nue_mc_raw_center_grad + v_numubar_to_nuebar_mc_raw_center_grad
    )
    v_raw_center_grad[:, 0] += v_raw_center

    v_rebinned_center_grad = f.rebinned_center(v_raw_center_grad, v_raw_to_rebin_slices)

    return (
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )


@njit(cache=True)
def mb_prereqs_grad(
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2_grad,
    v_nue_to_nue_mc_osc_sin2_grad,
    v_numu_to_numu_mc_osc_sin2_grad,
    v_numubar_to_nuebar_mc_osc_sin2_grad,
    v_nuebar_to_nuebar_mc_osc_sin2_grad,
    v_numubar_to_numubar_mc_osc_sin2_grad,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
):
    v_numu_to_nue_mc_osc_weights_grad = f.numu_to_nue_osc_weights_grad(
        v_numu_to_nue_mc_osc_factor, v_numu_to_nue_mc_osc_sin2_grad, r, theta
    )
    v_numu_to_nue_mc_weights_grad = f.mc_weights_grad(
        v_numu_to_nue_mc_cv_weights, v_numu_to_nue_mc_osc_weights_grad
    )
    v_numu_to_nue_mc_raw_center_grad = f.raw_center_grad(
        v_numu_to_nue_mc_weights_grad, v_numu_to_nue_mc_slices
    )

    v_nue_to_nue_mc_osc_weights_grad = f.nue_to_nue_osc_weights_grad(
        v_nue_to_nue_mc_osc_factor, v_nue_to_nue_mc_osc_sin2_grad, r, theta
    )
    v_nue_to_nue_mc_weights_grad = f.mc_weights_grad(
        v_nue_to_nue_mc_cv_weights, v_nue_to_nue_mc_osc_weights_grad
    )
    v_nue_to_nue_mc_raw_center_grad = f.raw_center_grad(
        v_nue_to_nue_mc_weights_grad, v_nue_to_nue_mc_slices
    )

    v_numu_to_numu_mc_osc_weights_grad = f.numu_to_numu_osc_weights_grad(
        v_numu_to_numu_mc_osc_factor, v_numu_to_numu_mc_osc_sin2_grad, r, theta
    )
    v_numu_to_numu_mc_weights_grad = f.mc_weights_grad(
        v_numu_to_numu_mc_cv_weights, v_numu_to_numu_mc_osc_weights_grad
    )
    v_numu_to_numu_mc_raw_center_grad = f.raw_center_grad(
        v_numu_to_numu_mc_weights_grad, v_numu_to_numu_mc_slices
    )

    v_numubar_to_nuebar_mc_osc_weights_grad = f.numu_to_nue_osc_weights_grad(
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_osc_sin2_grad,
        r,
        theta,
    )
    v_numubar_to_nuebar_mc_weights_grad = f.mc_weights_grad(
        v_numubar_to_nuebar_mc_cv_weights, v_numubar_to_nuebar_mc_osc_weights_grad
    )
    v_numubar_to_nuebar_mc_raw_center_grad = f.raw_center_grad(
        v_numubar_to_nuebar_mc_weights_grad, v_numubar_to_nuebar_mc_slices
    )

    v_nuebar_to_nuebar_mc_osc_weights_grad = f.nue_to_nue_osc_weights_grad(
        v_nuebar_to_nuebar_mc_osc_factor, v_nuebar_to_nuebar_mc_osc_sin2_grad, r, theta
    )
    v_nuebar_to_nuebar_mc_weights_grad = f.mc_weights_grad(
        v_nuebar_to_nuebar_mc_cv_weights, v_nuebar_to_nuebar_mc_osc_weights_grad
    )
    v_nuebar_to_nuebar_mc_raw_center_grad = f.raw_center_grad(
        v_nuebar_to_nuebar_mc_weights_grad, v_nuebar_to_nuebar_mc_slices
    )

    v_numubar_to_numubar_mc_osc_weights_grad = f.numu_to_numu_osc_weights_grad(
        v_numubar_to_numubar_mc_osc_factor,
        v_numubar_to_numubar_mc_osc_sin2_grad,
        r,
        theta,
    )
    v_numubar_to_numubar_mc_weights_grad = f.mc_weights_grad(
        v_numubar_to_numubar_mc_cv_weights, v_numubar_to_numubar_mc_osc_weights_grad
    )
    v_numubar_to_numubar_mc_raw_center_grad = f.raw_center_grad(
        v_numubar_to_numubar_mc_weights_grad, v_numubar_to_numubar_mc_slices
    )

    v_raw_center_grad = (
        v_numu_to_nue_mc_raw_center_grad
        + v_nue_to_nue_mc_raw_center_grad
        + v_numu_to_numu_mc_raw_center_grad
        + v_numubar_to_nuebar_mc_raw_center_grad
        + v_nuebar_to_nuebar_mc_raw_center_grad
        + v_numubar_to_numubar_mc_raw_center_grad
    )

    v_rebinned_center_grad = f.rebinned_center(v_raw_center_grad, v_raw_to_rebin_slices)

    return (
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )


@njit(cache=True)
def joint_nllh_grad(
    data,
    ub_data,
    ub_mode,
    ub_fit_mode,
    ub_sys_frac_cov,
    r,
    theta,
    # norm,
    # ub_norm,
    ub_ratios,
    dl_modes,
    dl_nonzero_eigen_values,
    wc_modes,
    wc_nonzero_eigen_values,
    v_numu_to_nue_mc_osc_sin2_grad,
    v_nue_to_nue_mc_osc_sin2_grad,
    v_numu_to_numu_mc_osc_sin2_grad,
    v_numubar_to_nuebar_mc_osc_sin2_grad,
    v_nuebar_to_nuebar_mc_osc_sin2_grad,
    v_numubar_to_numubar_mc_osc_sin2_grad,
    v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_nue_constrained_bkg_dl_template,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_dl_NuE_nue_constrained_cv_ratio,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
    v_dl_gauss_likelihood_prefactor,
    v_dl_expect_frac_mc_error,
    v_wc_gauss_likelihood_prefactor,
    v_wc_expect_frac_mc_error,
    # v_norm_var,
    # v_ub_norm_var,
    dl_zero_eigen_vector,
    wc_zero_eigen_vector,
):

    (
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_prereqs_grad(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2_grad,
        v_nue_to_nue_mc_osc_sin2_grad,
        v_numu_to_numu_mc_osc_sin2_grad,
        v_numubar_to_nuebar_mc_osc_sin2_grad,
        v_nuebar_to_nuebar_mc_osc_sin2_grad,
        v_numubar_to_numubar_mc_osc_sin2_grad,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_osc_factor,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_osc_factor,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_osc_factor,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_osc_factor,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood_grad(
        data,
        # norm,
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    # norm_prior = f.normal_prior_grad(norm, 1.0, v_norm_var)
    # ub_norm_prior = f.normal_prior_grad(ub_norm, 1.0, v_ub_norm_var)

    # ub stuff
    if ub_mode == "DL":
        v_dl_unscaled_expect_grad = dl_unscaled_expect_grad(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_dl_NuE_nue_smearing_matrix,
            v_numu_fitted_bkg_dl_template,
            v_dl_NuE_nue_constrained_cv_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like_grad = ub_likelihood_grad(
                ub_data,
                ub_ratios,
                dl_modes,
                dl_nonzero_eigen_values,
                v_dl_unscaled_expect_grad,
                v_dl_gauss_likelihood_prefactor,
                v_dl_expect_frac_mc_error,
                dl_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like_grad = ub_covCNP_grad(
                ub_data,
                v_dl_unscaled_expect_grad,
                ub_sys_frac_cov,
            )
    elif ub_mode == "WC":
        v_wc_unscaled_expect_grad = wc_unscaled_expect_grad(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_numu_to_numu_mc_osc_factor_ubbaseline,
            v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_numu_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_numu_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_ub_numu_to_numu_mb_cv_expect,
            v_wc_nue_FC_smearing_matrix,
            v_wc_nue_PC_smearing_matrix,
            v_wc_numu_FC_smearing_matrix,
            v_wc_numu_PC_smearing_matrix,
            v_nue_FC_bkg_wc_template,
            v_nue_PC_bkg_wc_template,
            v_numu_FC_bkg_wc_template,
            v_numu_PC_bkg_wc_template,
            v_wc_nue_FC_DR_SM_ratio,
            v_wc_nue_PC_DR_SM_ratio,
            v_wc_numu_FC_DR_SM_ratio,
            v_wc_numu_PC_DR_SM_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like_grad = ub_likelihood_grad(
                ub_data,
                ub_ratios,
                wc_modes,
                wc_nonzero_eigen_values,
                v_wc_unscaled_expect_grad,
                v_wc_gauss_likelihood_prefactor,
                v_wc_expect_frac_mc_error,
                wc_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like_grad = ub_covCNP_grad(
                ub_data,
                v_wc_unscaled_expect_grad,
                ub_sys_frac_cov,
            )

    n_mb_phys_params = np.shape(mb_like)[-1] - 1
    n_joint_phys_params = n_mb_phys_params

    joint_like = np.zeros(1 + n_joint_phys_params + len(ub_ratios))
    joint_like[0] = (
        ub_like_grad[0] + mb_like[0]
    )  # - (norm_prior[0] + ub_norm_prior[0])  # value
    # joint_like[1: np.shape(mb_like)[-1]] += mb_like[1:]  # MB phys params with MB norm
    joint_like[1 : np.shape(mb_like)[-1]] += mb_like[1:]  # MB phys params
    # joint_like[np.shape(mb_like)[-1] - 1] -= norm_prior[1]
    # joint_like[1: (np.shape(mb_like)[-1] - 1)] += ub_like_grad[1:(np.shape(mb_like)[-1] - 1)]  # uB phys params minus uB norm
    joint_like[1 : (np.shape(mb_like)[-1])] += ub_like_grad[
        1 : (np.shape(mb_like)[-1])
    ]  # UB phys params
    # joint_like[np.shape(mb_like)[-1]] += ub_like_grad[(np.shape(mb_like)[-1] - 1)] - ub_norm_prior[1]  # uB norm
    if ub_fit_mode=="LEff": joint_like[-(len(ub_ratios)) :] += ub_like_grad[-len(ub_ratios) :]  # UB pull terms
    return joint_like[0], joint_like[1:]


@njit(cache=True)
def mb_2nu_nllh_grad(
    data,
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2_grad,
    v_numubar_to_nuebar_mc_osc_sin2_grad,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    # v_norm_var,
):
    (
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_2nu_prereqs_grad(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2_grad,
        v_numubar_to_nuebar_mc_osc_sin2_grad,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood_grad(
        data,
        # norm,
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    # norm_prior = f.normal_prior_grad(norm, 1.0, v_norm_var)

    # n_phys_params = np.shape(mb_like)[-1] - 1
    # norm_prior_extended = np.zeros(n_phys_params + 1)
    # norm_prior_extended[0] = norm_prior[0]
    # norm_prior_extended[-1] = norm_prior[1]

    like = mb_like  # - norm_prior_extended
    return like[0], like[1:]


@njit(cache=True)
def mb_nllh_grad(
    data,
    r,
    theta,
    # norm,
    v_numu_to_nue_mc_osc_sin2_grad,
    v_nue_to_nue_mc_osc_sin2_grad,
    v_numu_to_numu_mc_osc_sin2_grad,
    v_numubar_to_nuebar_mc_osc_sin2_grad,
    v_nuebar_to_nuebar_mc_osc_sin2_grad,
    v_numubar_to_numubar_mc_osc_sin2_grad,
    v_rebinned_osc_mask,
    v_raw_to_rebin_slices,
    v_rebin_to_collapsed_slices,
    v_rebinned_sys_frac_cov,
    v_numu_to_nue_sorted_mc,
    v_numu_to_nue_mc_osc_factor,
    v_numu_to_nue_mc_cv_weights,
    v_numu_to_nue_mc_slices,
    v_nue_to_nue_sorted_mc,
    v_nue_to_nue_mc_osc_factor,
    v_nue_to_nue_mc_cv_weights,
    v_nue_to_nue_mc_slices,
    v_numu_to_numu_sorted_mc,
    v_numu_to_numu_mc_osc_factor,
    v_numu_to_numu_mc_cv_weights,
    v_numu_to_numu_mc_slices,
    v_numubar_to_nuebar_sorted_mc,
    v_numubar_to_nuebar_mc_osc_factor,
    v_numubar_to_nuebar_mc_cv_weights,
    v_numubar_to_nuebar_mc_slices,
    v_nuebar_to_nuebar_sorted_mc,
    v_nuebar_to_nuebar_mc_osc_factor,
    v_nuebar_to_nuebar_mc_cv_weights,
    v_nuebar_to_nuebar_mc_slices,
    v_numubar_to_numubar_sorted_mc,
    v_numubar_to_numubar_mc_osc_factor,
    v_numubar_to_numubar_mc_cv_weights,
    v_numubar_to_numubar_mc_slices,
    # v_norm_var,
):
    (
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    ) = mb_prereqs_grad(
        r,
        theta,
        # norm,
        v_numu_to_nue_mc_osc_sin2_grad,
        v_nue_to_nue_mc_osc_sin2_grad,
        v_numu_to_numu_mc_osc_sin2_grad,
        v_numubar_to_nuebar_mc_osc_sin2_grad,
        v_nuebar_to_nuebar_mc_osc_sin2_grad,
        v_numubar_to_numubar_mc_osc_sin2_grad,
        v_rebinned_osc_mask,
        v_raw_to_rebin_slices,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
        v_numu_to_nue_sorted_mc,
        v_numu_to_nue_mc_osc_factor,
        v_numu_to_nue_mc_cv_weights,
        v_numu_to_nue_mc_slices,
        v_nue_to_nue_sorted_mc,
        v_nue_to_nue_mc_osc_factor,
        v_nue_to_nue_mc_cv_weights,
        v_nue_to_nue_mc_slices,
        v_numu_to_numu_sorted_mc,
        v_numu_to_numu_mc_osc_factor,
        v_numu_to_numu_mc_cv_weights,
        v_numu_to_numu_mc_slices,
        v_numubar_to_nuebar_sorted_mc,
        v_numubar_to_nuebar_mc_osc_factor,
        v_numubar_to_nuebar_mc_cv_weights,
        v_numubar_to_nuebar_mc_slices,
        v_nuebar_to_nuebar_sorted_mc,
        v_nuebar_to_nuebar_mc_osc_factor,
        v_nuebar_to_nuebar_mc_cv_weights,
        v_nuebar_to_nuebar_mc_slices,
        v_numubar_to_numubar_sorted_mc,
        v_numubar_to_numubar_mc_osc_factor,
        v_numubar_to_numubar_mc_cv_weights,
        v_numubar_to_numubar_mc_slices,
    )

    mb_like = mb_likelihood_grad(
        data,
        # norm,
        v_rebinned_center_grad,
        v_rebin_to_collapsed_slices,
        v_rebinned_sys_frac_cov,
    )

    # norm_prior = f.normal_prior_grad(norm, 1.0, v_norm_var)

    # n_phys_params = np.shape(mb_like)[-1] - 1
    # norm_prior_extended = np.zeros(n_phys_params + 1)
    # norm_prior_extended[0] = norm_prior[0]
    # norm_prior_extended[-1] = norm_prior[1]

    like = mb_like  # - norm_prior_extended
    return like[0], like[1:]


@njit(cache=True)
def ub_nllh_grad(
    ub_data,
    ub_mode,
    ub_fit_mode,
    ub_sys_frac_cov,
    r,
    theta,
    # ub_norm,
    ub_ratios,
    dl_modes,
    dl_nonzero_eigen_values,
    wc_modes,
    wc_nonzero_eigen_values,
    v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
    v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
    v_numu_to_nue_mc_osc_factor_ubbaseline,
    v_nue_to_nue_mc_osc_factor_ubbaseline,
    v_numu_to_numu_mc_osc_factor_ubbaseline,
    v_ub_numu_to_nue_mb_mc_cv_weights,
    v_ub_nue_to_nue_mb_mc_cv_weights,
    v_ub_numu_to_numu_mb_mc_cv_weights,
    v_ub_numu_to_nue_mb_mc_slices,
    v_ub_nue_to_nue_mb_mc_slices,
    v_ub_numu_to_numu_mb_mc_slices,
    v_ub_numu_to_nue_mb_cv_expect,
    v_ub_nue_to_nue_mb_cv_expect,
    v_ub_numu_to_numu_mb_cv_expect,
    v_nue_constrained_bkg_dl_template,
    v_dl_NuE_nue_smearing_matrix,
    v_numu_fitted_bkg_dl_template,
    v_dl_NuE_nue_constrained_cv_ratio,
    v_wc_nue_FC_smearing_matrix,
    v_wc_nue_PC_smearing_matrix,
    v_wc_numu_FC_smearing_matrix,
    v_wc_numu_PC_smearing_matrix,
    v_nue_FC_bkg_wc_template,
    v_nue_PC_bkg_wc_template,
    v_numu_FC_bkg_wc_template,
    v_numu_PC_bkg_wc_template,
    v_wc_nue_FC_DR_SM_ratio,
    v_wc_nue_PC_DR_SM_ratio,
    v_wc_numu_FC_DR_SM_ratio,
    v_wc_numu_PC_DR_SM_ratio,
    v_dl_gauss_likelihood_prefactor,
    v_dl_expect_frac_mc_error,
    v_wc_gauss_likelihood_prefactor,
    v_wc_expect_frac_mc_error,
    # v_norm_var,
    # v_ub_norm_var,
    dl_zero_eigen_vector,
    wc_zero_eigen_vector,
):
    # ub_norm_prior = f.normal_prior_grad(ub_norm, 1.0, v_ub_norm_var)

    # ub stuff
    if ub_mode == "DL":
        v_dl_unscaled_expect_grad = dl_unscaled_expect_grad(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_dl_NuE_nue_smearing_matrix,
            v_numu_fitted_bkg_dl_template,
            v_dl_NuE_nue_constrained_cv_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like_grad = ub_likelihood_grad(
                ub_data,
                ub_ratios,
                dl_modes,
                dl_nonzero_eigen_values,
                v_dl_unscaled_expect_grad,
                v_dl_gauss_likelihood_prefactor,
                v_dl_expect_frac_mc_error,
                dl_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like_grad = ub_covCNP_grad(
                ub_data,
                v_dl_unscaled_expect_grad,
                ub_sys_frac_cov,
            )
    elif ub_mode == "WC":
        v_wc_unscaled_expect_grad = wc_unscaled_expect_grad(
            r,
            theta,
            v_numu_to_nue_mc_osc_factor_ubbaseline,
            v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_nue_to_nue_mc_osc_factor_ubbaseline,
            v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            v_numu_to_numu_mc_osc_factor_ubbaseline,
            v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
            v_ub_numu_to_nue_mb_mc_cv_weights,
            v_ub_nue_to_nue_mb_mc_cv_weights,
            v_ub_numu_to_numu_mb_mc_cv_weights,
            v_ub_numu_to_nue_mb_mc_slices,
            v_ub_nue_to_nue_mb_mc_slices,
            v_ub_numu_to_numu_mb_mc_slices,
            v_ub_numu_to_nue_mb_cv_expect,
            v_ub_nue_to_nue_mb_cv_expect,
            v_ub_numu_to_numu_mb_cv_expect,
            v_wc_nue_FC_smearing_matrix,
            v_wc_nue_PC_smearing_matrix,
            v_wc_numu_FC_smearing_matrix,
            v_wc_numu_PC_smearing_matrix,
            v_nue_FC_bkg_wc_template,
            v_nue_PC_bkg_wc_template,
            v_numu_FC_bkg_wc_template,
            v_numu_PC_bkg_wc_template,
            v_wc_nue_FC_DR_SM_ratio,
            v_wc_nue_PC_DR_SM_ratio,
            v_wc_numu_FC_DR_SM_ratio,
            v_wc_numu_PC_DR_SM_ratio,
        )
        if ub_fit_mode=="LEff":
            ub_like_grad = ub_likelihood_grad(
                ub_data,
                ub_ratios,
                wc_modes,
                wc_nonzero_eigen_values,
                v_wc_unscaled_expect_grad,
                v_wc_gauss_likelihood_prefactor,
                v_wc_expect_frac_mc_error,
                wc_zero_eigen_vector,
            )
        elif ub_fit_mode=="CNP":
            ub_like_grad = ub_covCNP_grad(
                ub_data,
                v_wc_unscaled_expect_grad,
                ub_sys_frac_cov,
            )

    return ub_like_grad[0], ub_like_grad[1:]


class problem:
    def __init__(self, the_store=None, dm2=None, sin22th=None, ub_mode="DL", ub_fit_mode="LEff"):

        if the_store is None:
            the_store = analysis.setup_analysis()
        else:
            the_store = the_store

        self.the_store = the_store

        # ub stuff
        self.ub_mode = ub_mode
        self.ub_fit_mode = ub_fit_mode
        self.v_dl_binned_data = the_store.get_prop("dl_NuE_binned_data")
        self.v_dl_expect_frac_mc_error = the_store.get_prop("dl_expect_frac_mc_error")
        self.v_wc_binned_data = the_store.get_prop("nue_numu_data_wc_template")
        self.v_wc_expect_frac_mc_error = the_store.get_prop("wc_expect_frac_mc_error")
        if self.ub_mode == "DL":
            ubsamp = "dl"
        elif self.ub_mode == "WC":
            ubsamp = "wc"
        self.v_numu_to_nue_mc_osc_factor_ubbaseline = the_store.get_prop(
            ubsamp + "_numu_to_nue_mc_osc_factor_ubbaseline"
        )
        self.v_nue_to_nue_mc_osc_factor_ubbaseline = the_store.get_prop(
            ubsamp + "_nue_to_nue_mc_osc_factor_ubbaseline"
        )
        self.v_ub_numu_to_nue_mb_mc_cv_weights = the_store.get_prop(
            ubsamp + "_numu_to_nue_mb_mc_cv_weights"
        )
        self.v_ub_nue_to_nue_mb_mc_cv_weights = the_store.get_prop(
            ubsamp + "_nue_to_nue_mb_mc_cv_weights"
        )
        self.v_ub_numu_to_nue_mb_mc_slices = the_store.get_prop(
            ubsamp + "_numu_to_nue_mb_mc_slices"
        )
        self.v_ub_nue_to_nue_mb_mc_slices = the_store.get_prop(
            ubsamp + "_nue_to_nue_mb_mc_slices"
        )
        self.v_ub_numu_to_nue_mb_cv_expect = the_store.get_prop(
            ubsamp + "_numu_to_nue_mb_cv_expect"
        )
        self.v_ub_nue_to_nue_mb_cv_expect = the_store.get_prop(
            ubsamp + "_nue_to_nue_mb_cv_expect"
        )
        # Temporarily set numu->numu osc weights to WC calculation (not used in DL): required for numba
        self.v_numu_to_numu_mc_osc_factor_ubbaseline = the_store.get_prop(
            "wc_numu_to_numu_mc_osc_factor_ubbaseline"
        )
        self.v_ub_numu_to_numu_mb_mc_cv_weights = the_store.get_prop(
            "wc_numu_to_numu_mb_mc_cv_weights"
        )
        self.v_ub_numu_to_numu_mb_mc_slices = the_store.get_prop(
            "wc_numu_to_numu_mb_mc_slices"
        )
        self.v_ub_numu_to_numu_mb_cv_expect = the_store.get_prop(
            "wc_numu_to_numu_mb_cv_expect"
        )

        self.v_dl_NuE_nue_cv_expect = the_store.get_prop("dl_NuE_nue_cv_expect")
        self.v_nue_constrained_bkg_dl_template = the_store.get_prop(
            "nue_constrained_bkg_dl_template"
        )
        self.v_numu_fitted_bkg_dl_template = the_store.get_prop(
            "numu_fitted_bkg_dl_template"
        )
        self.v_nue_FC_sig_wc_template = the_store.get_prop("nue_FC_sig_wc_template")
        self.v_nue_FC_bkg_wc_template = the_store.get_prop("nue_FC_bkg_wc_template")
        self.v_nue_PC_sig_wc_template = the_store.get_prop("nue_PC_sig_wc_template")
        self.v_nue_PC_bkg_wc_template = the_store.get_prop("nue_PC_bkg_wc_template")
        self.v_numu_FC_sig_wc_template = the_store.get_prop("numu_FC_sig_wc_template")
        self.v_numu_FC_bkg_wc_template = the_store.get_prop("numu_FC_bkg_wc_template")
        self.v_numu_PC_sig_wc_template = the_store.get_prop("numu_PC_sig_wc_template")
        self.v_numu_PC_bkg_wc_template = the_store.get_prop("numu_PC_bkg_wc_template")
        self.v_dl_NuE_nue_smearing_matrix = the_store.get_prop(
            "dl_NuE_nue_smearing_matrix"
        )
        self.v_wc_nue_FC_smearing_matrix = the_store.get_prop(
            "wc_nue_FC_smearing_matrix"
        )
        self.v_wc_nue_PC_smearing_matrix = the_store.get_prop(
            "wc_nue_PC_smearing_matrix"
        )
        self.v_wc_numu_FC_smearing_matrix = the_store.get_prop(
            "wc_numu_FC_smearing_matrix"
        )
        self.v_wc_numu_PC_smearing_matrix = the_store.get_prop(
            "wc_numu_PC_smearing_matrix"
        )
        self.v_dl_NuE_nue_constrained_cv_ratio = the_store.get_prop(
            "dl_NuE_nue_constrained_cv_ratio"
        )
        self.v_wc_nue_FC_DR_SM_ratio = the_store.get_prop("wc_nue_FC_DR_SM_ratio")
        self.v_wc_nue_PC_DR_SM_ratio = the_store.get_prop("wc_nue_PC_DR_SM_ratio")
        self.v_wc_numu_FC_DR_SM_ratio = the_store.get_prop("wc_numu_FC_DR_SM_ratio")
        self.v_wc_numu_PC_DR_SM_ratio = the_store.get_prop("wc_numu_PC_DR_SM_ratio")
        self.v_dl_NuE_nominal_constrained_frac_cov = the_store.get_prop(
            "dl_NuE_nominal_constrained_frac_cov"
        )
        self.v_dl_gauss_likelihood_prefactor = the_store.get_prop(
            "dl_gauss_likelihood_prefactor"
        )
        self.v_wc_nue_numu_nominal_frac_cov = the_store.get_prop(
            "wc_nue_numu_nominal_frac_cov"
        )
        self.v_dl_gauss_likelihood_prefactor = the_store.get_prop(
            "dl_gauss_likelihood_prefactor"
        )
        self.v_wc_gauss_likelihood_prefactor = the_store.get_prop(
            "wc_gauss_likelihood_prefactor"
        )

        # MB stuff
        self.v_mb_binned_data = the_store.get_prop("binned_data")
        self.v_rebinned_osc_mask = the_store.get_prop("rebinned_osc_mask")

        self.v_raw_to_rebin_slices = the_store.get_prop("raw_to_rebin_slices")
        self.v_rebin_to_collapsed_slices = the_store.get_prop(
            "rebin_to_collapsed_slices"
        )
        self.v_rebinned_sys_frac_cov = the_store.get_prop("rebinned_sys_frac_cov")

        self.v_numu_to_nue_sorted_mc = the_store.get_prop("numu_to_nue_sorted_mc")
        self.v_numu_to_nue_mc_osc_factor = the_store.get_prop(
            "numu_to_nue_mc_osc_factor"
        )
        self.v_numu_to_nue_mc_cv_weights = the_store.get_prop(
            "numu_to_nue_mc_cv_weights"
        )
        self.v_numu_to_nue_mc_slices = the_store.get_prop("numu_to_nue_mc_slices")

        self.v_nue_to_nue_sorted_mc = the_store.get_prop("nue_to_nue_sorted_mc")
        self.v_nue_to_nue_mc_osc_factor = the_store.get_prop("nue_to_nue_mc_osc_factor")
        self.v_nue_to_nue_mc_cv_weights = the_store.get_prop("nue_to_nue_mc_cv_weights")
        self.v_nue_to_nue_mc_slices = the_store.get_prop("nue_to_nue_mc_slices")

        self.v_numu_to_numu_sorted_mc = the_store.get_prop("numu_to_numu_sorted_mc")
        self.v_numu_to_numu_mc_osc_factor = the_store.get_prop(
            "numu_to_numu_mc_osc_factor"
        )
        self.v_numu_to_numu_mc_cv_weights = the_store.get_prop(
            "numu_to_numu_mc_cv_weights"
        )
        self.v_numu_to_numu_mc_slices = the_store.get_prop("numu_to_numu_mc_slices")

        self.v_numubar_to_nuebar_sorted_mc = the_store.get_prop(
            "numubar_to_nuebar_sorted_mc"
        )
        self.v_numubar_to_nuebar_mc_osc_factor = the_store.get_prop(
            "numubar_to_nuebar_mc_osc_factor"
        )
        self.v_numubar_to_nuebar_mc_cv_weights = the_store.get_prop(
            "numubar_to_nuebar_mc_cv_weights"
        )
        self.v_numubar_to_nuebar_mc_slices = the_store.get_prop(
            "numubar_to_nuebar_mc_slices"
        )

        self.v_nuebar_to_nuebar_sorted_mc = the_store.get_prop(
            "nuebar_to_nuebar_sorted_mc"
        )
        self.v_nuebar_to_nuebar_mc_osc_factor = the_store.get_prop(
            "nuebar_to_nuebar_mc_osc_factor"
        )
        self.v_nuebar_to_nuebar_mc_cv_weights = the_store.get_prop(
            "nuebar_to_nuebar_mc_cv_weights"
        )
        self.v_nuebar_to_nuebar_mc_slices = the_store.get_prop(
            "nuebar_to_nuebar_mc_slices"
        )

        self.v_numubar_to_numubar_sorted_mc = the_store.get_prop(
            "numubar_to_numubar_sorted_mc"
        )
        self.v_numubar_to_numubar_mc_osc_factor = the_store.get_prop(
            "numubar_to_numubar_mc_osc_factor"
        )
        self.v_numubar_to_numubar_mc_cv_weights = the_store.get_prop(
            "numubar_to_numubar_mc_cv_weights"
        )
        self.v_numubar_to_numubar_mc_slices = the_store.get_prop(
            "numubar_to_numubar_mc_slices"
        )

        # self.v_norm_var = the_store.get_prop("norm_var")

        # self.v_ub_norm_var = the_store.get_prop("ub_norm_var")

        dl_eigen_values, dl_eigen_vectors = scipy.linalg.eig(
            self.v_dl_NuE_nominal_constrained_frac_cov
        )
        wc_eigen_values, wc_eigen_vectors = scipy.linalg.eig(
            self.v_wc_nue_numu_nominal_frac_cov
        )

        self.dl_n_modes = the_store.get_prop("dl_n_modes")
        # assert n_uB_zero_modes == 1
        # self.dl_zero_eigen_vector = the_store.get_prop("dl_zero_eigen_vector")
        self.dl_zero_eigen_vector = None
        self.dl_modes = the_store.get_prop("dl_modes")
        self.dl_eigen_values = the_store.get_prop("dl_eigen_values")
        self.dl_eigen_nonzero_mask = the_store.get_prop("dl_eigen_nonzero_mask")
        self.dl_nonzero_eigen_values = the_store.get_prop("dl_nonzero_eigen_values")

        self.wc_n_modes = the_store.get_prop("wc_n_modes")
        self.wc_zero_eigen_vector = None
        self.wc_modes = the_store.get_prop("wc_modes_masked")
        self.wc_eigen_values = the_store.get_prop("wc_eigen_values")
        self.wc_eigen_nonzero_mask = the_store.get_prop("wc_eigen_nonzero_mask")
        self.wc_nonzero_eigen_values = the_store.get_prop("wc_nonzero_eigen_values")

        if self.ub_mode == "DL":
            self.v_ub_binned_data = self.v_dl_binned_data
            self.ub_n_modes = the_store.get_prop("dl_n_modes")
            self.ub_zero_eigen_vector = None
            self.ub_modes = the_store.get_prop("dl_modes")
            self.ub_eigen_values = the_store.get_prop("dl_eigen_values")
            self.ub_eigen_nonzero_mask = the_store.get_prop("dl_eigen_nonzero_mask")
            self.ub_nonzero_eigen_values = the_store.get_prop("dl_nonzero_eigen_values")
            self.ub_cov = the_store.get_prop("dl_NuE_nominal_constrained_frac_cov")
            self.ub_cov_inv = the_store.get_prop(
                "dl_NuE_nominal_constrained_frac_cov_inv"
            )
        elif self.ub_mode == "WC":
            self.v_ub_binned_data = self.v_wc_binned_data
            self.ub_n_modes = the_store.get_prop("wc_n_modes")
            self.ub_zero_eigen_vector = None
            self.ub_modes = the_store.get_prop("wc_modes_masked")
            self.ub_eigen_values = the_store.get_prop("wc_eigen_values")
            self.ub_eigen_nonzero_mask = the_store.get_prop("wc_eigen_nonzero_mask")
            self.ub_nonzero_eigen_values = the_store.get_prop("wc_nonzero_eigen_values")
            self.ub_cov = the_store.get_prop("wc_nue_numu_nominal_frac_cov")
            self.ub_cov_inv = the_store.get_prop("wc_nue_numu_nominal_frac_cov_inv")

        self.last_grad_dm2 = None
        self.last_grad_dm2_ub = None
        self.last_grad_dm2_mb = None
        self.v_numu_to_nue_mc_osc_sin2_grad = None
        self.v_nue_to_nue_mc_osc_sin2_grad = None
        self.v_numu_to_numu_mc_osc_sin2_grad = None
        self.v_numubar_to_nuebar_mc_osc_sin2_grad = None
        self.v_nuebar_to_nuebar_mc_osc_sin2_grad = None
        self.v_numubar_to_numubar_mc_osc_sin2_grad = None

        self.v_nue_to_nue_mc_osc_sin2_grad_ubbaseline = None
        self.v_numu_to_nue_mc_osc_sin2_grad_ubbaseline = None

        self.last_dm2 = None
        self.last_dm2_ub = None
        self.last_dm2_mb = None
        self.v_numu_to_nue_mc_osc_sin2 = None
        self.v_nue_to_nue_mc_osc_sin2 = None
        self.v_numu_to_numu_mc_osc_sin2 = None
        self.v_numubar_to_nuebar_mc_osc_sin2 = None
        self.v_nuebar_to_nuebar_mc_osc_sin2 = None
        self.v_numubar_to_numubar_mc_osc_sin2 = None

        self.v_nue_to_nue_mc_osc_sin2_ubbaseline = None
        self.v_numu_to_nue_mc_osc_sin2_ubbaseline = None

    def cached_dm2_grad(self, dm2):
        if self.v_numu_to_nue_mc_osc_sin2_grad is None or dm2 != self.last_grad_dm2:
            self.last_grad_dm2 = dm2
            self.last_grad_dm2_ub = None
            self.last_grad_dm2_mb = None
            self.v_numu_to_nue_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numu_to_nue_mc_osc_factor, dm2
            )
            self.v_nue_to_nue_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_nue_to_nue_mc_osc_factor, dm2
            )
            self.v_numu_to_numu_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numu_to_numu_mc_osc_factor, dm2
            )
            self.v_numubar_to_nuebar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numubar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_nuebar_to_nuebar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_nuebar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_numubar_to_numubar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numubar_to_numubar_mc_osc_factor, dm2
            )

            self.v_nue_to_nue_mc_osc_sin2_grad_ubbaseline = f.mc_osc_sin2_check_grad(
                self.v_nue_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            self.v_numu_to_nue_mc_osc_sin2_grad_ubbaseline = f.mc_osc_sin2_check_grad(
                self.v_numu_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            if True:  # self.ub_mode=='WC'):
                self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline = (
                    f.mc_osc_sin2_check_grad(
                        self.v_numu_to_numu_mc_osc_factor_ubbaseline, dm2
                    )
                )
            else:
                self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline = None

    def cached_dm2_grad_ub(self, dm2):
        if (
            self.v_nue_to_nue_mc_osc_factor_ubbaseline is None
            or dm2 != self.last_grad_dm2_ub
        ):
            self.last_grad_dm2 = None
            self.last_grad_dm2_ub = dm2
            self.last_grad_dm2_mb = None

            self.v_nue_to_nue_mc_osc_sin2_grad_ubbaseline = f.mc_osc_sin2_check_grad(
                self.v_nue_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            self.v_numu_to_nue_mc_osc_sin2_grad_ubbaseline = f.mc_osc_sin2_check_grad(
                self.v_numu_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            if True:  # self.ub_mode=='WC'):
                self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline = (
                    f.mc_osc_sin2_check_grad(
                        self.v_numu_to_numu_mc_osc_factor_ubbaseline, dm2
                    )
                )
            else:
                self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline = None

    def cached_dm2_grad_mb(self, dm2):
        if self.v_numu_to_nue_mc_osc_sin2_grad is None or dm2 != self.last_grad_dm2_mb:
            self.last_grad_dm2 = None
            self.last_grad_dm2_ub = None
            self.last_grad_dm2_mb = dm2
            self.v_numu_to_nue_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numu_to_nue_mc_osc_factor, dm2
            )
            self.v_nue_to_nue_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_nue_to_nue_mc_osc_factor, dm2
            )
            self.v_numu_to_numu_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numu_to_numu_mc_osc_factor, dm2
            )
            self.v_numubar_to_nuebar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numubar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_nuebar_to_nuebar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_nuebar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_numubar_to_numubar_mc_osc_sin2_grad = f.mc_osc_sin2_grad(
                self.v_numubar_to_numubar_mc_osc_factor, dm2
            )

    def cached_dm2(self, dm2):
        if self.v_numu_to_nue_mc_osc_sin2 is None or dm2 != self.last_dm2:
            self.last_dm2 = dm2
            self.last_dm2_ub = None
            self.last_dm2_mb = None
            self.v_numu_to_nue_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numu_to_nue_mc_osc_factor, dm2
            )
            self.v_nue_to_nue_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_nue_to_nue_mc_osc_factor, dm2
            )
            self.v_numu_to_numu_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numu_to_numu_mc_osc_factor, dm2
            )
            self.v_numubar_to_nuebar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numubar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_nuebar_to_nuebar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_nuebar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_numubar_to_numubar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numubar_to_numubar_mc_osc_factor, dm2
            )

            self.v_nue_to_nue_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                self.v_nue_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            self.v_numu_to_nue_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                self.v_numu_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            if True:  # self.ub_mode=='WC'):
                self.v_numu_to_numu_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                    self.v_numu_to_numu_mc_osc_factor_ubbaseline, dm2
                )
            else:
                self.v_numu_to_numu_mc_osc_sin2_ubbaseline = None

    def cached_dm2_ub(self, dm2):
        if (
            self.v_nue_to_nue_mc_osc_factor_ubbaseline is None
            or dm2 != self.last_dm2_ub
        ):
            self.last_dm2 = None
            self.last_dm2_ub = dm2
            self.last_dm2_mb = None

            self.v_nue_to_nue_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                self.v_nue_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            self.v_numu_to_nue_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                self.v_numu_to_nue_mc_osc_factor_ubbaseline, dm2
            )
            if True:  # self.ub_mode=='WC'):
                self.v_numu_to_numu_mc_osc_sin2_ubbaseline = f.mc_osc_sin2_check(
                    self.v_numu_to_numu_mc_osc_factor_ubbaseline, dm2
                )

    def cached_dm2_mb(self, dm2):
        if self.v_numu_to_nue_mc_osc_sin2 is None or dm2 != self.last_dm2_mb:
            self.last_dm2 = None
            self.last_dm2_ub = None
            self.last_dm2_mb = dm2
            self.v_numu_to_nue_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numu_to_nue_mc_osc_factor, dm2
            )
            self.v_nue_to_nue_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_nue_to_nue_mc_osc_factor, dm2
            )
            self.v_numu_to_numu_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numu_to_numu_mc_osc_factor, dm2
            )
            self.v_numubar_to_nuebar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numubar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_nuebar_to_nuebar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_nuebar_to_nuebar_mc_osc_factor, dm2
            )
            self.v_numubar_to_numubar_mc_osc_sin2 = f.mc_osc_sin2(
                self.v_numubar_to_numubar_mc_osc_factor, dm2
            )

    # def joint_nllh_grad_cached_dm2(self, data, ub_data, dm2, r, theta, norm, ub_norm, ub_ratios):
    def joint_nllh_grad_cached_dm2(self, data, ub_data, dm2, r, theta, ub_ratios):
        self.cached_dm2_grad(dm2)
        res = joint_nllh_grad(
            data,
            ub_data,
            self.ub_mode,
            self.ub_fit_mode,
            self.ub_cov,
            r,
            theta,
            # norm,
            # ub_norm,
            ub_ratios,
            self.dl_modes,
            self.dl_nonzero_eigen_values,
            self.wc_modes,
            self.wc_nonzero_eigen_values,
            self.v_numu_to_nue_mc_osc_sin2_grad,
            self.v_nue_to_nue_mc_osc_sin2_grad,
            self.v_numu_to_numu_mc_osc_sin2_grad,
            self.v_numubar_to_nuebar_mc_osc_sin2_grad,
            self.v_nuebar_to_nuebar_mc_osc_sin2_grad,
            self.v_numubar_to_numubar_mc_osc_sin2_grad,
            self.v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            self.v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_osc_factor,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_osc_factor,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_osc_factor,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_osc_factor,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            self.v_numu_to_nue_mc_osc_factor_ubbaseline,
            self.v_nue_to_nue_mc_osc_factor_ubbaseline,
            self.v_numu_to_numu_mc_osc_factor_ubbaseline,
            self.v_ub_numu_to_nue_mb_mc_cv_weights,
            self.v_ub_nue_to_nue_mb_mc_cv_weights,
            self.v_ub_numu_to_numu_mb_mc_cv_weights,
            self.v_ub_numu_to_nue_mb_mc_slices,
            self.v_ub_nue_to_nue_mb_mc_slices,
            self.v_ub_numu_to_numu_mb_mc_slices,
            self.v_ub_numu_to_nue_mb_cv_expect,
            self.v_ub_nue_to_nue_mb_cv_expect,
            self.v_ub_numu_to_numu_mb_cv_expect,
            self.v_nue_constrained_bkg_dl_template,
            self.v_dl_NuE_nue_smearing_matrix,
            self.v_numu_fitted_bkg_dl_template,
            self.v_dl_NuE_nue_constrained_cv_ratio,
            self.v_wc_nue_FC_smearing_matrix,
            self.v_wc_nue_PC_smearing_matrix,
            self.v_wc_numu_FC_smearing_matrix,
            self.v_wc_numu_PC_smearing_matrix,
            self.v_nue_FC_bkg_wc_template,
            self.v_nue_PC_bkg_wc_template,
            self.v_numu_FC_bkg_wc_template,
            self.v_numu_PC_bkg_wc_template,
            self.v_wc_nue_FC_DR_SM_ratio,
            self.v_wc_nue_PC_DR_SM_ratio,
            self.v_wc_numu_FC_DR_SM_ratio,
            self.v_wc_numu_PC_DR_SM_ratio,
            self.v_dl_gauss_likelihood_prefactor,
            self.v_dl_expect_frac_mc_error,
            self.v_wc_gauss_likelihood_prefactor,
            self.v_wc_expect_frac_mc_error,
            self.dl_zero_eigen_vector,
            self.wc_zero_eigen_vector,
        )
        return res

    # def ub_nllh_grad_cached_dm2(self, ub_data, dm2, r, theta, ub_norm, ub_ratios):
    def ub_nllh_grad_cached_dm2(self, ub_data, dm2, r, theta, ub_ratios):
        self.cached_dm2_grad_ub(dm2)
        res = ub_nllh_grad(
            ub_data,
            self.ub_mode,
            self.ub_fit_mode,
            self.ub_cov,
            r,
            theta,
            # ub_norm,
            ub_ratios,
            self.dl_modes,
            self.dl_nonzero_eigen_values,
            self.wc_modes,
            self.wc_nonzero_eigen_values,
            self.v_numu_to_nue_mc_osc_sin2_grad_ubbaseline,
            self.v_nue_to_nue_mc_osc_sin2_grad_ubbaseline,
            self.v_numu_to_numu_mc_osc_sin2_grad_ubbaseline,
            self.v_numu_to_nue_mc_osc_factor_ubbaseline,
            self.v_nue_to_nue_mc_osc_factor_ubbaseline,
            self.v_numu_to_numu_mc_osc_factor_ubbaseline,
            self.v_ub_numu_to_nue_mb_mc_cv_weights,
            self.v_ub_nue_to_nue_mb_mc_cv_weights,
            self.v_ub_numu_to_numu_mb_mc_cv_weights,
            self.v_ub_numu_to_nue_mb_mc_slices,
            self.v_ub_nue_to_nue_mb_mc_slices,
            self.v_ub_numu_to_numu_mb_mc_slices,
            self.v_ub_numu_to_nue_mb_cv_expect,
            self.v_ub_nue_to_nue_mb_cv_expect,
            self.v_ub_numu_to_numu_mb_cv_expect,
            self.v_nue_constrained_bkg_dl_template,
            self.v_dl_NuE_nue_smearing_matrix,
            self.v_numu_fitted_bkg_dl_template,
            self.v_dl_NuE_nue_constrained_cv_ratio,
            self.v_wc_nue_FC_smearing_matrix,
            self.v_wc_nue_PC_smearing_matrix,
            self.v_wc_numu_FC_smearing_matrix,
            self.v_wc_numu_PC_smearing_matrix,
            self.v_nue_FC_bkg_wc_template,
            self.v_nue_PC_bkg_wc_template,
            self.v_numu_FC_bkg_wc_template,
            self.v_numu_PC_bkg_wc_template,
            self.v_wc_nue_FC_DR_SM_ratio,
            self.v_wc_nue_PC_DR_SM_ratio,
            self.v_wc_numu_FC_DR_SM_ratio,
            self.v_wc_numu_PC_DR_SM_ratio,
            self.v_dl_gauss_likelihood_prefactor,
            self.v_dl_expect_frac_mc_error,
            self.v_wc_gauss_likelihood_prefactor,
            self.v_wc_expect_frac_mc_error,
            self.dl_zero_eigen_vector,
            self.wc_zero_eigen_vector,
        )
        return res

    # def mb_2nu_nllh_grad_cached_dm2(self, data, dm2, r, theta, norm):
    def mb_2nu_nllh_grad_cached_dm2(self, data, dm2, r, theta):
        self.cached_dm2_grad_mb(dm2)
        res = mb_2nu_nllh_grad(
            data,
            r,
            theta,
            # norm,
            self.v_numu_to_nue_mc_osc_sin2_grad,
            self.v_numubar_to_nuebar_mc_osc_sin2_grad,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            # self.v_norm_var,
        )
        return res

    # def mb_nllh_grad_cached_dm2(self, data, dm2, r, theta, norm):
    def mb_nllh_grad_cached_dm2(self, data, dm2, r, theta):
        self.cached_dm2_grad_mb(dm2)
        res = mb_nllh_grad(
            data,
            r,
            theta,
            # norm,
            self.v_numu_to_nue_mc_osc_sin2_grad,
            self.v_nue_to_nue_mc_osc_sin2_grad,
            self.v_numu_to_numu_mc_osc_sin2_grad,
            self.v_numubar_to_nuebar_mc_osc_sin2_grad,
            self.v_nuebar_to_nuebar_mc_osc_sin2_grad,
            self.v_numubar_to_numubar_mc_osc_sin2_grad,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_osc_factor,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_osc_factor,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_osc_factor,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_osc_factor,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            # self.v_norm_var,
        )
        return res

    # def joint_nllh_cached_dm2(self, data, ub_data, dm2, r, theta, norm, ub_norm, ub_ratios):
    def joint_nllh_cached_dm2(self, data, ub_data, dm2, r, theta, ub_ratios):
        self.cached_dm2(dm2)
        res = joint_nllh(
            data,
            ub_data,
            self.ub_mode,
            self.ub_fit_mode,
            self.ub_cov,
            r,
            theta,
            # norm,
            # ub_norm,
            ub_ratios,
            self.dl_modes,
            self.dl_nonzero_eigen_values,
            self.wc_modes,
            self.wc_nonzero_eigen_values,
            self.v_numu_to_nue_mc_osc_sin2,
            self.v_nue_to_nue_mc_osc_sin2,
            self.v_numu_to_numu_mc_osc_sin2,
            self.v_numubar_to_nuebar_mc_osc_sin2,
            self.v_nuebar_to_nuebar_mc_osc_sin2,
            self.v_numubar_to_numubar_mc_osc_sin2,
            self.v_nue_to_nue_mc_osc_sin2_ubbaseline,
            self.v_numu_to_nue_mc_osc_sin2_ubbaseline,
            self.v_numu_to_numu_mc_osc_sin2_ubbaseline,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_osc_factor,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_osc_factor,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_osc_factor,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_osc_factor,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            self.v_numu_to_nue_mc_osc_factor_ubbaseline,
            self.v_nue_to_nue_mc_osc_factor_ubbaseline,
            self.v_numu_to_numu_mc_osc_factor_ubbaseline,
            self.v_ub_numu_to_nue_mb_mc_cv_weights,
            self.v_ub_nue_to_nue_mb_mc_cv_weights,
            self.v_ub_numu_to_numu_mb_mc_cv_weights,
            self.v_ub_numu_to_nue_mb_mc_slices,
            self.v_ub_nue_to_nue_mb_mc_slices,
            self.v_ub_numu_to_numu_mb_mc_slices,
            self.v_ub_numu_to_nue_mb_cv_expect,
            self.v_ub_nue_to_nue_mb_cv_expect,
            self.v_ub_numu_to_numu_mb_cv_expect,
            self.v_nue_constrained_bkg_dl_template,
            self.v_dl_NuE_nue_smearing_matrix,
            self.v_numu_fitted_bkg_dl_template,
            self.v_dl_NuE_nue_constrained_cv_ratio,
            self.v_wc_nue_FC_smearing_matrix,
            self.v_wc_nue_PC_smearing_matrix,
            self.v_wc_numu_FC_smearing_matrix,
            self.v_wc_numu_PC_smearing_matrix,
            self.v_nue_FC_bkg_wc_template,
            self.v_nue_PC_bkg_wc_template,
            self.v_numu_FC_bkg_wc_template,
            self.v_numu_PC_bkg_wc_template,
            self.v_wc_nue_FC_DR_SM_ratio,
            self.v_wc_nue_PC_DR_SM_ratio,
            self.v_wc_numu_FC_DR_SM_ratio,
            self.v_wc_numu_PC_DR_SM_ratio,
            self.v_dl_gauss_likelihood_prefactor,
            self.v_dl_expect_frac_mc_error,
            self.v_wc_gauss_likelihood_prefactor,
            self.v_wc_expect_frac_mc_error,
            self.dl_zero_eigen_vector,
            self.wc_zero_eigen_vector,
        )
        return res

    # def mb_2nu_nllh_cached_dm2(self, data, dm2, r, theta, norm):
    def mb_2nu_nllh_cached_dm2(self, data, dm2, r, theta):
        self.cached_dm2_mb(dm2)
        res = mb_2nu_nllh(
            data,
            r,
            theta,
            # norm,
            self.v_numu_to_nue_mc_osc_sin2,
            self.v_numubar_to_nuebar_mc_osc_sin2,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            # self.v_norm_var,
        )
        return res

    # def mb_nllh_cached_dm2(self, data, dm2, r, theta, norm):
    def mb_nllh_cached_dm2(self, data, dm2, r, theta):
        self.cached_dm2_mb(dm2)
        res = mb_nllh(
            data,
            r,
            theta,
            # norm,
            self.v_numu_to_nue_mc_osc_sin2,
            self.v_nue_to_nue_mc_osc_sin2,
            self.v_numu_to_numu_mc_osc_sin2,
            self.v_numubar_to_nuebar_mc_osc_sin2,
            self.v_nuebar_to_nuebar_mc_osc_sin2,
            self.v_numubar_to_numubar_mc_osc_sin2,
            self.v_rebinned_osc_mask,
            self.v_raw_to_rebin_slices,
            self.v_rebin_to_collapsed_slices,
            self.v_rebinned_sys_frac_cov,
            self.v_numu_to_nue_sorted_mc,
            self.v_numu_to_nue_mc_osc_factor,
            self.v_numu_to_nue_mc_cv_weights,
            self.v_numu_to_nue_mc_slices,
            self.v_nue_to_nue_sorted_mc,
            self.v_nue_to_nue_mc_osc_factor,
            self.v_nue_to_nue_mc_cv_weights,
            self.v_nue_to_nue_mc_slices,
            self.v_numu_to_numu_sorted_mc,
            self.v_numu_to_numu_mc_osc_factor,
            self.v_numu_to_numu_mc_cv_weights,
            self.v_numu_to_numu_mc_slices,
            self.v_numubar_to_nuebar_sorted_mc,
            self.v_numubar_to_nuebar_mc_osc_factor,
            self.v_numubar_to_nuebar_mc_cv_weights,
            self.v_numubar_to_nuebar_mc_slices,
            self.v_nuebar_to_nuebar_sorted_mc,
            self.v_nuebar_to_nuebar_mc_osc_factor,
            self.v_nuebar_to_nuebar_mc_cv_weights,
            self.v_nuebar_to_nuebar_mc_slices,
            self.v_numubar_to_numubar_sorted_mc,
            self.v_numubar_to_numubar_mc_osc_factor,
            self.v_numubar_to_numubar_mc_cv_weights,
            self.v_numubar_to_numubar_mc_slices,
            # self.v_norm_var,
        )
        return res

    # def ub_nllh_cached_dm2(self, ub_data, dm2, r, theta, ub_norm, ub_ratios):
    def ub_nllh_cached_dm2(self, ub_data, dm2, r, theta, ub_ratios):
        self.cached_dm2_ub(dm2)
        res = ub_nllh(
            ub_data,
            self.ub_mode,
            self.ub_fit_mode,
            self.ub_cov,
            r,
            theta,
            # ub_norm,
            ub_ratios,
            self.dl_modes,
            self.dl_nonzero_eigen_values,
            self.wc_modes,
            self.wc_nonzero_eigen_values,
            self.v_numu_to_nue_mc_osc_sin2_ubbaseline,
            self.v_nue_to_nue_mc_osc_sin2_ubbaseline,
            self.v_numu_to_numu_mc_osc_sin2_ubbaseline,
            self.v_numu_to_nue_mc_osc_factor_ubbaseline,
            self.v_nue_to_nue_mc_osc_factor_ubbaseline,
            self.v_numu_to_numu_mc_osc_factor_ubbaseline,
            self.v_ub_numu_to_nue_mb_mc_cv_weights,
            self.v_ub_nue_to_nue_mb_mc_cv_weights,
            self.v_ub_numu_to_numu_mb_mc_cv_weights,
            self.v_ub_numu_to_nue_mb_mc_slices,
            self.v_ub_nue_to_nue_mb_mc_slices,
            self.v_ub_numu_to_numu_mb_mc_slices,
            self.v_ub_numu_to_nue_mb_cv_expect,
            self.v_ub_nue_to_nue_mb_cv_expect,
            self.v_ub_numu_to_numu_mb_cv_expect,
            self.v_nue_constrained_bkg_dl_template,
            self.v_dl_NuE_nue_smearing_matrix,
            self.v_numu_fitted_bkg_dl_template,
            self.v_dl_NuE_nue_constrained_cv_ratio,
            self.v_wc_nue_FC_smearing_matrix,
            self.v_wc_nue_PC_smearing_matrix,
            self.v_wc_numu_FC_smearing_matrix,
            self.v_wc_numu_PC_smearing_matrix,
            self.v_nue_FC_bkg_wc_template,
            self.v_nue_PC_bkg_wc_template,
            self.v_numu_FC_bkg_wc_template,
            self.v_numu_PC_bkg_wc_template,
            self.v_wc_nue_FC_DR_SM_ratio,
            self.v_wc_nue_PC_DR_SM_ratio,
            self.v_wc_numu_FC_DR_SM_ratio,
            self.v_wc_numu_PC_DR_SM_ratio,
            self.v_dl_gauss_likelihood_prefactor,
            self.v_dl_expect_frac_mc_error,
            self.v_wc_gauss_likelihood_prefactor,
            self.v_wc_expect_frac_mc_error,
            self.dl_zero_eigen_vector,
            self.wc_zero_eigen_vector,
        )
        return res

    def ub_unscaled_expect_cached_dm2(self, dm2, r, theta):
        # print(dm2, r, theta, ub_norm, ub_ratios)
        # print(dm2, r, theta, ub_ratios)
        self.cached_dm2_ub(dm2)
        if self.ub_mode == "DL":
            return dl_unscaled_expect(
                r,
                theta,
                self.v_numu_to_nue_mc_osc_factor_ubbaseline,
                self.v_numu_to_nue_mc_osc_sin2_ubbaseline,
                self.v_nue_to_nue_mc_osc_factor_ubbaseline,
                self.v_nue_to_nue_mc_osc_sin2_ubbaseline,
                self.v_ub_numu_to_nue_mb_mc_cv_weights,
                self.v_ub_nue_to_nue_mb_mc_cv_weights,
                self.v_ub_numu_to_nue_mb_mc_slices,
                self.v_ub_nue_to_nue_mb_mc_slices,
                self.v_ub_numu_to_nue_mb_cv_expect,
                self.v_ub_nue_to_nue_mb_cv_expect,
                self.v_dl_NuE_nue_smearing_matrix,
                self.v_numu_fitted_bkg_dl_template,
                self.v_dl_NuE_nue_constrained_cv_ratio,
            )
        elif self.ub_mode == "WC":
            return wc_unscaled_expect(
                r,
                theta,
                self.v_numu_to_nue_mc_osc_factor_ubbaseline,
                self.v_numu_to_nue_mc_osc_sin2_ubbaseline,
                self.v_nue_to_nue_mc_osc_factor_ubbaseline,
                self.v_nue_to_nue_mc_osc_sin2_ubbaseline,
                self.v_numu_to_numu_mc_osc_factor_ubbaseline,
                self.v_numu_to_numu_mc_osc_sin2_ubbaseline,
                self.v_ub_numu_to_nue_mb_mc_cv_weights,
                self.v_ub_nue_to_nue_mb_mc_cv_weights,
                self.v_ub_numu_to_numu_mb_mc_cv_weights,
                self.v_ub_numu_to_nue_mb_mc_slices,
                self.v_ub_nue_to_nue_mb_mc_slices,
                self.v_ub_numu_to_numu_mb_mc_slices,
                self.v_ub_numu_to_nue_mb_cv_expect,
                self.v_ub_nue_to_nue_mb_cv_expect,
                self.v_ub_numu_to_numu_mb_cv_expect,
                self.v_wc_nue_FC_smearing_matrix,
                self.v_wc_nue_PC_smearing_matrix,
                self.v_wc_numu_FC_smearing_matrix,
                self.v_wc_numu_PC_smearing_matrix,
                self.v_nue_FC_bkg_wc_template,
                self.v_nue_PC_bkg_wc_template,
                self.v_numu_FC_bkg_wc_template,
                self.v_numu_PC_bkg_wc_template,
                self.v_wc_nue_FC_DR_SM_ratio,
                self.v_wc_nue_PC_DR_SM_ratio,
                self.v_wc_numu_FC_DR_SM_ratio,
                self.v_wc_numu_PC_DR_SM_ratio,
            )

    def fitness(self, x):
        # dm2, r, theta, norm, ub_norm = x[:5]
        dm2, r, theta = x[:3]
        # ub_ratios = x[5:]
        ub_ratios = x[3]
        return self.joint_nllh_cached_dm2(
            # self.v_mb_binned_data, self.v_ub_binned_data, dm2, r, theta, norm, ub_norm, ub_ratios
            self.v_mb_binned_data,
            self.v_ub_binned_data,
            dm2,
            r,
            theta,
            ub_ratios,
        )

    def get_bounds(self):
        dm2_bounds = [[1e-2, 1e2]]
        r_bounds = [[0, 1]]
        theta_bounds = [[0, np.pi / 2.0]]
        # sigma = np.sqrt(self.v_norm_var)
        # ub_sigma = np.sqrt(self.v_ub_norm_var)
        # norm_bounds = [[max(0.0, 1.0 - sigma*4), min(3.0, 1.0 + sigma*4)], [max(0.0, 1.0 - ub_sigma*4), min(3.0, 1.0 + ub_sigma*4)]]
        sigma = np.sqrt(np.diag(self.v_dl_NuE_nominal_constrained_frac_cov))
        lower_mins = np.zeros(len(self.v_dl_binned_data)) + 1e-8
        var_mins = 1.0 - 4 * sigma
        mins = np.amax([lower_mins, var_mins], axis=0)
        maxs = 1.0 + 4 * sigma
        ub_mean_bounds = (np.array([mins, maxs]).T).tolist()
        # bounds = dm2_bounds + r_bounds + theta_bounds + norm_bounds + ub_mean_bounds
        bounds = dm2_bounds + r_bounds + theta_bounds + ub_mean_bounds
        return bounds

    def has_gradient(self):
        return True

    def gradient(self, x):
        # dm2, r, theta, norm, ub_norm = x[:5]
        dm2, r, theta = x[:3]
        # ub_ratios = x[5:]
        ub_ratios = x[3:]
        L_grad = self.joint_nllh_grad_cached_dm2(
            # self.v_mb_binned_data, self.v_ub_binned_data, dm2, r, theta, norm, ub_norm, ub_ratios
            self.v_mb_binned_data,
            self.v_ub_binned_data,
            dm2,
            r,
            theta,
            ub_ratios,
        )
        return L_grad[1:]

    def set_experimental_data(self):
        self.v_dl_binned_data = self.the_store.get_prop("dl_NuE_binned_data")
        self.v_wc_binned_data = the_store.get_prop("nue_numu_data_wc_template")
        self.v_mb_binned_data = self.the_store.get_prop("binned_data")

    def set_asimov_data(
        self, dm2, r, theta, norm=None, ub_norm=None, ub_ratios=None, ub_params=None
    ):
        if ub_ratios is not None:
            ub_ratio_vec = ub_ratios
        elif ub_params is not None:
            ub_scale = ub_params * self.ub_nonzero_eigen_values
            full_ub_diff_ratio = np.sum(
                np.expand_dims(ub_scale, 0) * self.ub_modes, axis=1
            )
            ub_ratio_vec = full_ub_diff_ratio + 1.0
        else:
            ub_ratio_vec = np.ones(len(self.v_ub_binned_data))
        if norm is None:
            norm = 1.0
        if ub_norm is None:
            ub_norm = 1.0

        props = {
            "dm2": dm2,
            "effective_radius": r,
            "effective_theta": theta,
            "norm": norm,
            "ub_norm": ub_norm,
            "ub_ratio_vec": ub_ratio_vec,
        }

        if self.ub_mode == "DL":
            self.v_ub_binned_data = self.the_store.get_prop("dl_expect", props)
        elif self.ub_mode == "WC":
            self.v_ub_binned_data = self.the_store.get_prop("wc_expect", props)
        self.v_mb_binned_data = self.the_store.get_prop("expect", props)
        return props

    def set_trial_data(
        self, dm2, r, theta, norm=None, ub_norm=None, ub_ratios=None, ub_params=None
    ):
        props = self.set_asimov_data(
            self,
            dm2,
            r,
            theta,
            norm=norm,
            ub_norm=ub_norm,
            ub_ratios=ub_ratios,
            ub_params=ub_params,
        )
        ub_mean = self.v_ub_binned_data
        mb_mean = self.v_mb_binned_data

        mb_sys_cov = np.copy(self.the_store.get_prop("sys_cov", props))
        mb_sys_cov -= np.diag(mb_mean)

        mb_mc_error = self.the_store.get_prop("stat_error", props)
        mb_sys_cov -= mb_mc_error
