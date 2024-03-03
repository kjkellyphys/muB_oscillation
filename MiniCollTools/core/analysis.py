import numpy as np
import prop_store
import load_sample
import binning
import functions as f
import scipy
import scipy.linalg
from numba import njit

### How to build an analysis
## Instantiate the store
#
# the_store = prop_store.store()
#
## The store computes values for you and caches the result so you don't compute anything twice.
## If you need a computed value, you go to the store:
#
# my_expensive_numbers = the_store.get_prop("expensive_numbers", physics_parameters)
#
## But first you have to tell the store how to do things.
## So you define a function:
#
# def my_func(foo, bar, physics_param):
#     ret = ...
#     ... compute things ...
#     return ret
#
## Once defined, you must register the function with the store.
## This requires a name for the output of the function,
## the ordered names of parameters it depends on,
## and the function itself.
#
# the_store.add_prop("my_value", ["value_of_foo", "value_of_bar", "physics_value"], my_func)
#
## Now you can register other functions that use the output and so on...
#
# def my_other_func(my_value):
#     return (my_value + 1)/2.0
# the_store.add_prop("my_other_value", ["my_value"], my_other_func)
#
## This implicitly depends on "physics_value", but that is handled by the store.
## Just please don't define an infinite loop via interdependency...
#
## Finally you have to initialize the store so it can work out the implicit dependencies of the
## things you defined, figure out what the physics parameters are, and spin up the object caches
#
# the_store.initialize()
#
## If you are reinitializing the store after adding props and you want to keep the caches
#
# the_store.initialize(keep_cache=True)
#
## Now you can ask the store for values as long as you give it the appropriate physics parameters
## It will work out all the details and try not to recompute anything if it can help it
#
# physics_parameters = {"physics_value": np.pi/4.}
# value = the_store.get_prop("my_value", physics_parameters)


def setup_analysis():

    the_store = prop_store.store()

    the_store.add_prop("bin_transforms", [], binning.bin_transforms)
    the_store.add_prop(
        "data_bin_transforms", [], lambda: binning.bin_transforms(is_data=True)
    )
    for i, label in enumerate(
        ["transform_slices", "orig_idx_order", "component_bin_masks", "raw_ids"]
    ):
        the_store.add_prop(label, ["bin_transforms"], f.get_item(i))
        the_store.add_prop("data_" + label, ["data_bin_transforms"], f.get_item(i))
    for level in ["raw", "rebinned", "collapsed"]:
        for category in ["osc", "intrinsic", "nue", "numu", "nubar", "numubar"]:
            the_store.add_prop(
                level + "_" + category + "_mask",
                ["component_bin_masks"],
                f.get_item((level, category)),
            )
        the_store.add_prop(
            "data_" + level + "_mask",
            ["data_component_bin_masks"],
            f.get_item((level, category)),
        )
    the_store.add_prop(
        "raw_to_rebin_slices", ["transform_slices"], f.get_item("raw_to_rebin")
    )
    the_store.add_prop(
        "rebin_to_collapsed_slices",
        ["transform_slices"],
        f.get_item("rebin_to_collapsed"),
    )
    the_store.add_prop(
        "data_raw_to_rebin_slices",
        ["data_transform_slices"],
        f.get_item("raw_to_rebin"),
    )
    the_store.add_prop(
        "data_rebin_to_collapsed_slices",
        ["data_transform_slices"],
        f.get_item("rebin_to_collapsed"),
    )

    the_store.add_prop("data", [], load_sample.load_data)
    the_store.add_prop("mc", [], load_sample.load_mc)
    the_store.add_prop("split_mc", ["mc", "component_bin_masks"], f.split_mc)

    the_store.add_prop(
        "data_sorting_info", ["data", "data_bin_transforms"], f.data_sorting_info
    )
    the_store.add_prop("sorted_data", ["data_sorting_info"], f.get_item(0))
    the_store.add_prop("data_slices", ["data_sorting_info"], f.get_item(1))
    the_store.add_prop("raw_data", ["data_slices"], f.binned_data)
    the_store.add_prop(
        "rebinned_data", ["raw_data", "data_raw_to_rebin_slices"], f.expect
    )
    the_store.add_prop(
        "binned_data", ["rebinned_data", "data_rebin_to_collapsed_slices"], f.expect
    )

    mc_samples = [
        "numu_to_nue",
        "numubar_to_nuebar",
        "numu_to_numu",
        "numubar_to_numubar",
        "nue_to_nue",
        "nuebar_to_nuebar",
    ]
    for sample in mc_samples:
        the_store.add_prop(sample + "_mc", ["split_mc"], f.get_item(sample))

        the_store.add_prop(
            sample + "_mc_sorting_info",
            [sample + "_mc", "bin_transforms"],
            f.mc_sorting_info,
        )
        the_store.add_prop(
            sample + "_sorted_mc", [sample + "_mc_sorting_info"], f.get_item(0)
        )
        the_store.add_prop(
            sample + "_mc_slices", [sample + "_mc_sorting_info"], f.get_item(1)
        )
        the_store.add_prop(
            sample + "_mc_raw_center",
            [sample + "_mc_weights", sample + "_mc_slices"],
            f.raw_center,
        )
        the_store.add_prop(
            sample + "_mc_raw_center_sq",
            [sample + "_mc_weights", sample + "_mc_slices"],
            f.raw_center_sq,
        )
        the_store.add_prop(
            sample + "_mc_raw_center_grad",
            [sample + "_mc_weights_grad", sample + "_mc_slices"],
            f.raw_center_grad,
        )
        the_store.add_prop(
            sample + "_mc_raw_center_sq_grad",
            [sample + "_mc_weights_grad", sample + "_mc_slices"],
            f.raw_center_sq_grad,
        )
        the_store.add_prop(
            sample + "_mc_osc_weight_masks",
            [sample + "_sorted_mc"],
            f.mc_osc_weight_masks,
        )
        the_store.add_prop(
            sample + "_mc_osc_factor", [sample + "_sorted_mc"], f.mc_osc_factor
        )
        the_store.add_prop(
            sample + "_mc_osc_sin2_grad",
            [sample + "_mc_osc_factor", "dm2"],
            f.mc_osc_sin2_grad,
        )
        the_store.add_prop(
            sample + "_mc_osc_sin2",
            [sample + "_mc_osc_factor", "dm2"],
            f.mc_osc_sin2,
        )

    for sample in [
        "numu_to_nue",
        "numubar_to_nuebar",
        "numu_to_numu",
        "numubar_to_numubar",
        "nue_to_nue",
        "nuebar_to_nuebar",
    ]:
        the_store.add_prop(
            sample + "_mc_cv_weights", [sample + "_sorted_mc"], f.cv_weight
        )
        the_store.add_prop(
            sample + "_mc_weights",
            [sample + "_mc_cv_weights", sample + "_mc_osc_weights"],
            f.mc_weights,
        )
        the_store.add_prop(
            sample + "_mc_weights_grad",
            [sample + "_mc_cv_weights", sample + "_mc_osc_weights_grad"],
            f.mc_weights_grad,
        )

    for sample in []:
        the_store.add_prop(sample + "_mc_weights", [sample + "_sorted_mc"], f.cv_weight)
        the_store.add_prop(
            sample + "_mc_weights_grad",
            [sample + "_sorted_mc"],
            lambda x: np.expand_dims(f.cv_weight(x), 1),
        )

    sum_args = lambda *args: sum(args)

    def sum_args(*args):
        res = sum(args)
        return res

    the_store.add_prop(
        "raw_center_unscaled",
        [sample + "_mc_raw_center" for sample in mc_samples],
        sum_args,
    )
    # the_store.add_prop(
    #     "raw_center", ["raw_center_unscaled", "norm"], f.apply_norm
    # )
    the_store.add_prop("raw_center", ["raw_center_unscaled"], f.identity)
    the_store.add_prop(
        "raw_center_unscaled_sq",
        [sample + "_mc_raw_center_sq" for sample in mc_samples],
        sum_args,
    )
    the_store.add_prop(
        "raw_center_sq",
        ["raw_center_unscaled_sq"],
        f.apply_norm_sq,
    )
    the_store.add_prop(
        "raw_center_unscaled_grad",
        [sample + "_mc_raw_center_grad" for sample in mc_samples],
        sum_args,
    )
    # the_store.add_prop(
    #     "raw_center_grad",
    #     ["raw_center_unscaled_grad", "norm"],
    #     f.apply_norm_grad,
    # )
    the_store.add_prop(
        "raw_center_grad",
        ["raw_center_unscaled_grad"],
        f.identity,
    )
    the_store.add_prop(
        "raw_center_unscaled_sq_grad",
        [sample + "_mc_raw_center_sq_grad" for sample in mc_samples],
        sum_args,
    )
    # the_store.add_prop(
    #     "raw_center_sq_grad",
    #     ["raw_center_unscaled_sq_grad", "norm"],
    #     f.apply_norm_sq_grad,
    # )
    the_store.add_prop(
        "raw_center_sq_grad",
        ["raw_center_unscaled_sq_grad"],
        f.identity,
    )

    the_store.add_prop(
        "raw_null_center",
        [sample + "_mc_raw_center" for sample in mc_samples],
        sum_args,
        override_params={"dm2": -1, "effective_radius": -1, "effective_theta": -1},
    )
    the_store.add_prop(
        "raw_null_center_sq",
        [sample + "_mc_raw_center_sq" for sample in mc_samples],
        sum_args,
        override_params={"dm2": -1, "effective_radius": -1, "effective_theta": -1},
    )
    the_store.add_prop(
        "raw_null_center_grad",
        [sample + "_mc_raw_center_grad" for sample in mc_samples],
        sum_args,
        override_params={"dm2": -1, "effective_radius": -1, "effective_theta": -1},
    )
    the_store.add_prop(
        "raw_null_center_sq_grad",
        [sample + "_mc_raw_center_sq_grad" for sample in mc_samples],
        sum_args,
        override_params={"dm2": -1, "effective_radius": -1, "effective_theta": -1},
    )

    the_store.add_prop(
        "numu_to_nue_mc_osc_weights",
        [
            "numu_to_nue_mc_osc_factor",
            "numu_to_nue_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_nue_osc_weights,
    )
    the_store.add_prop(
        "numubar_to_nuebar_mc_osc_weights",
        [
            "numubar_to_nuebar_mc_osc_factor",
            "numubar_to_nuebar_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_nue_osc_weights,
    )

    the_store.add_prop(
        "nue_to_nue_mc_osc_weights",
        [
            "nue_to_nue_mc_osc_factor",
            "nue_to_nue_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.nue_to_nue_osc_weights,
    )
    the_store.add_prop(
        "nuebar_to_nuebar_mc_osc_weights",
        [
            "nuebar_to_nuebar_mc_osc_factor",
            "nuebar_to_nuebar_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.nue_to_nue_osc_weights,
    )

    the_store.add_prop(
        "numu_to_numu_mc_osc_weights",
        [
            "numu_to_numu_mc_osc_factor",
            "numu_to_numu_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_numu_osc_weights,
    )
    the_store.add_prop(
        "numubar_to_numubar_mc_osc_weights",
        [
            "numubar_to_numubar_mc_osc_factor",
            "numubar_to_numubar_mc_osc_sin2",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_numu_osc_weights,
    )

    the_store.add_prop(
        "numu_to_nue_mc_osc_weights_grad",
        [
            "numu_to_nue_mc_osc_factor",
            "numu_to_nue_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_nue_osc_weights_grad,
    )
    the_store.add_prop(
        "numubar_to_nuebar_mc_osc_weights_grad",
        [
            "numubar_to_nuebar_mc_osc_factor",
            "numubar_to_nuebar_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_nue_osc_weights_grad,
    )

    the_store.add_prop(
        "nue_to_nue_mc_osc_weights_grad",
        [
            "nue_to_nue_mc_osc_factor",
            "nue_to_nue_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.nue_to_nue_osc_weights_grad,
    )
    the_store.add_prop(
        "nuebar_to_nuebar_mc_osc_weights_grad",
        [
            "nuebar_to_nuebar_mc_osc_factor",
            "nuebar_to_nuebar_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.nue_to_nue_osc_weights_grad,
    )

    the_store.add_prop(
        "numu_to_numu_mc_osc_weights_grad",
        [
            "numu_to_numu_mc_osc_factor",
            "numu_to_numu_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_numu_osc_weights_grad,
    )
    the_store.add_prop(
        "numubar_to_numubar_mc_osc_weights_grad",
        [
            "numubar_to_numubar_mc_osc_factor",
            "numubar_to_numubar_mc_osc_sin2_grad",
            "effective_radius",
            "effective_theta",
        ],
        f.numu_to_numu_osc_weights_grad,
    )

    the_store.add_prop(
        "rebinned_center_grad",
        ["raw_center_grad", "raw_to_rebin_slices"],
        f.rebinned_center,
    )
    the_store.add_prop(
        "rebinned_center", ["raw_center", "raw_to_rebin_slices"], f.rebinned_center
    )
    the_store.add_prop(
        "rebinned_null_center",
        ["raw_null_center", "raw_to_rebin_slices"],
        f.rebinned_center,
    )
    the_store.add_prop(
        "rebinned_center_sq_grad",
        ["raw_center_sq_grad", "raw_to_rebin_slices"],
        f.rebinned_center,
    )
    the_store.add_prop(
        "rebinned_center_sq",
        ["raw_center_sq", "raw_to_rebin_slices"],
        f.rebinned_center,
    )
    the_store.add_prop(
        "rebinned_null_center_sq",
        ["raw_null_center_sq", "raw_to_rebin_slices"],
        f.rebinned_center,
    )
    the_store.add_prop(
        "cov_cv_map",
        [
            "raw_null_center",
            "raw_null_center_sq",
            "transform_slices",
            "orig_idx_order",
            "component_bin_masks",
        ],
        f.covariance.load_cov,
    )
    the_store.add_prop(
        "raw_stat_frac_error",
        [
            "raw_null_center_sq",
            "raw_null_center",
            "component_bin_masks",
        ],
        f.covariance.mc_stat_error,
    )
    the_store.add_prop(
        "rebinned_stat_frac_error",
        ["raw_stat_frac_error", "rebinned_null_center", "raw_to_rebin_slices"],
        f.rebinned_sys_frac_cov,
    )
    the_store.add_prop(
        "rebinned_stat_error",
        ["rebinned_stat_frac_error", "rebinned_expect"],
        f.mul_cov,
    )
    the_store.add_prop(
        "stat_error", ["rebinned_stat_error", "rebin_to_collapsed_slices"], f.sys_cov
    )
    the_store.add_prop(
        "raw_frac_cov_list",
        ["cov_cv_map"],
        f.raw_frac_cov_list,
    )
    the_store.add_prop("raw_sys_frac_cov", ["raw_frac_cov_list"], f.raw_sys_frac_cov)
    the_store.add_prop(
        "raw_sys_cov_grad",
        ["raw_sys_frac_cov", "raw_null_center_grad"],
        f.raw_sys_cov_grad,
    )
    the_store.add_prop(
        "raw_sys_cov",
        ["raw_sys_frac_cov", "raw_null_center"],
        f.mul_cov,
    )
    the_store.add_prop(
        "rebinned_sys_frac_cov",
        ["raw_sys_cov", "rebinned_null_center", "raw_to_rebin_slices"],
        f.rebinned_sys_frac_cov,
    )
    the_store.add_prop(
        "rebinned_sys_cov_grad",
        ["rebinned_sys_frac_cov", "rebinned_expect_grad"],
        f.mul_const_cov_grad,
    )
    the_store.add_prop(
        "rebinned_sys_cov",
        ["rebinned_sys_frac_cov", "rebinned_expect"],
        f.mul_cov,
    )
    the_store.add_prop("raw_expect_grad", ["raw_center_grad"], f.identity, cache_size=0)
    the_store.add_prop("raw_expect", ["raw_center"], f.identity, cache_size=0)
    the_store.add_prop(
        "raw_expect_sq_grad", ["raw_center_sq_grad"], f.identity, cache_size=0
    )
    the_store.add_prop("raw_expect_sq", ["raw_center_sq"], f.identity, cache_size=0)
    the_store.add_prop(
        "rebinned_expect_grad", ["rebinned_center_grad"], f.identity, cache_size=0
    )
    the_store.add_prop("rebinned_expect", ["rebinned_center"], f.identity, cache_size=0)
    the_store.add_prop(
        "rebinned_expect_sq_grad", ["rebinned_center_sq_grad"], f.identity, cache_size=0
    )
    the_store.add_prop(
        "rebinned_expect_sq", ["rebinned_center_sq"], f.identity, cache_size=0
    )

    ## the_store.add_prop(
    ##     "rebinned_osc_stat_error_grad",
    ##     ["rebinned_center_grad", "rebinned_osc_mask"],
    ##     f.rebinned_osc_stat_error_grad,
    ## )
    ## the_store.add_prop(
    ##     "rebinned_osc_stat_error",
    ##     ["rebinned_center", "rebinned_osc_mask"],
    ##     f.rebinned_osc_stat_error,
    ## )
    the_store.add_prop(
        "rebinned_intrinsic_stat_error_grad",
        ["rebinned_center_grad", "mc_rebinned_intrinsic_mask"],
        f.rebinned_intrinsic_stat_error_grad,
    )
    the_store.add_prop(
        "rebinned_intrinsic_stat_error",
        ["rebinned_center", "mc_rebinned_intrinsic_mask"],
        f.rebinned_intrinsic_stat_error,
    )
    the_store.add_prop(
        "expect_grad",
        ["rebinned_expect_grad", "rebin_to_collapsed_slices"],
        f.expect,
    )
    the_store.add_prop(
        "asimov_expect_grad",
        ["rebinned_expect_grad", "rebin_to_collapsed_slices"],
        f.expect,
    )
    the_store.add_prop(
        "expect", ["rebinned_expect", "rebin_to_collapsed_slices"], f.expect
    )
    the_store.add_prop(
        "asimov_expect", ["rebinned_expect", "rebin_to_collapsed_slices"], f.expect
    )
    the_store.add_prop(
        "expect_sq_grad",
        ["rebinned_expect_sq_grad", "rebin_to_collapsed_slices"],
        f.expect,
    )
    the_store.add_prop(
        "asimov_expect_sq_grad",
        ["rebinned_expect_sq_grad", "rebin_to_collapsed_slices"],
        f.expect,
    )
    the_store.add_prop(
        "expect_sq", ["rebinned_expect_sq", "rebin_to_collapsed_slices"], f.expect
    )
    the_store.add_prop(
        "asimov_expect_sq",
        ["rebinned_expect_sq", "rebin_to_collapsed_slices"],
        f.expect,
    )
    the_store.add_prop(
        "sys_cov", ["rebinned_sys_cov", "rebin_to_collapsed_slices"], f.sys_cov
    )
    the_store.add_prop(
        "sys_cov_grad",
        ["rebinned_sys_cov_grad", "rebin_to_collapsed_slices"],
        f.sys_cov,
    )
    the_store.add_prop(
        "osc_stat_error",
        ["rebinned_osc_stat_error", "rebin_to_collapsed_slices"],
        f.osc_stat_error,
    )
    the_store.add_prop(
        "osc_stat_error_grad",
        ["rebinned_osc_stat_error_grad", "rebin_to_collapsed_slices"],
        f.osc_stat_error_grad,
    )
    the_store.add_prop(
        "intrinsic_stat_error",
        ["rebinned_intrinsic_stat_error", "rebin_to_collapsed_slices"],
        f.osc_stat_error,
    )
    the_store.add_prop(
        "intrinsic_stat_error_grad",
        ["rebinned_intrinsic_stat_error_grad", "rebin_to_collapsed_slices"],
        f.osc_stat_error_grad,
    )
    the_store.add_prop(
        "stat_error",
        ["expect"],
        f.stat_error,
    )
    the_store.add_prop("cov", ["sys_cov", "stat_error"], f.cov_add_error)
    the_store.add_prop("cov_grad", ["sys_cov_grad", "stat_error_grad"], f.cov_add_error)
    the_store.add_prop("cov_inv", ["cov"], f.cov_inv)
    the_store.add_prop("cov_inv_grad", ["cov_grad"], f.cov_inv_grad)
    the_store.add_prop("gauss_likelihood_prefactor", ["cov"], f.gauss_prefactor)
    the_store.add_prop(
        "gauss_likelihood_prefactor_grad",
        ["cov_grad", "cov_inv_grad"],
        f.gauss_prefactor_grad,
    )
    the_store.add_prop(
        "gauss_likelihood_exponent", ["diff", "cov_inv"], f.gauss_exponent
    )
    the_store.add_prop(
        "gauss_likelihood_exponent_grad",
        ["diff_grad", "cov_grad", "cov_inv_grad"],
        f.gauss_exponent_grad,
    )
    the_store.add_prop(
        "gauss_likelihood_grad",
        ["gauss_likelihood_exponent_grad", "gauss_likelihood_prefactor_grad"],
        f.gauss_grad,
    )

    ## Normalization prior

    # the_store.add_prop(
    #     "norm_var",
    #     ["cov_cv_map"],
    #     f.norm_var_from_covs,
    # )

    # the_store.add_prop(
    #     "norm_prior",
    #     ["norm", "norm_var"],
    #     lambda norm, norm_var: f.normal_prior(norm, 1.0, norm_var),
    # )

    ## uB likelihood

    for ubsamp in ["dl", "wc"]:
        for sample in mc_samples:
            # microboone baseline weights
            the_store.add_prop(
                ubsamp + "_" + sample + "_mc_osc_factor_ubbaseline",
                [ubsamp + "_" + sample + "_mb_sorted_mc"],
                f.mc_osc_factor_ubbaseline,
            )  # sorted by true energy
            the_store.add_prop(
                ubsamp + "_" + sample + "_mc_osc_sin2_grad_ubbaseline",
                [ubsamp + "_" + sample + "_mc_osc_factor_ubbaseline", "dm2"],
                f.mc_osc_sin2_check_grad,
            )  # sorted by true energy
            the_store.add_prop(
                ubsamp + "_" + sample + "_mc_osc_sin2_ubbaseline",
                [ubsamp + "_" + sample + "_mc_osc_factor_ubbaseline", "dm2"],
                f.mc_osc_sin2_check,
            )  # sorted by true energy
            the_store.add_prop(
                ubsamp + "_" + sample + "_mb_mc_weights_ubbaseline",
                [
                    ubsamp + "_" + sample + "_mb_mc_cv_weights",
                    ubsamp + "_" + sample + "_mc_osc_weights_ubbaseline",
                ],
                f.mc_weights,
            )  # sorted by true energy
            the_store.add_prop(
                ubsamp + "_" + sample + "_mb_mc_weights_grad_ubbaseline",
                [
                    ubsamp + "_" + sample + "_mb_mc_cv_weights",
                    ubsamp + "_" + sample + "_mc_osc_weights_grad_ubbaseline",
                ],
                f.mc_weights_grad,
            )  # sorted by true energy

        the_store.add_prop(
            ubsamp + "_" + "numu_to_nue_mc_osc_weights_ubbaseline",
            [
                ubsamp + "_" + "numu_to_nue_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "numu_to_nue_mc_osc_sin2_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.numu_to_nue_osc_weights,
        )  # sorted by true energy
        the_store.add_prop(
            ubsamp + "_" + "nue_to_nue_mc_osc_weights_ubbaseline",
            [
                ubsamp + "_" + "nue_to_nue_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "nue_to_nue_mc_osc_sin2_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.nue_to_nue_osc_weights,
        )  # sorted by true energy
        the_store.add_prop(
            ubsamp + "_" + "numu_to_numu_mc_osc_weights_ubbaseline",
            [
                ubsamp + "_" + "numu_to_numu_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "numu_to_numu_mc_osc_sin2_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.numu_to_numu_osc_weights,
        )  # sorted by true energy
        the_store.add_prop(
            ubsamp + "_" + "numu_to_nue_mc_osc_weights_grad_ubbaseline",
            [
                ubsamp + "_" + "numu_to_nue_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "numu_to_nue_mc_osc_sin2_grad_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.numu_to_nue_osc_weights_grad,
        )  # sorted by true energy
        the_store.add_prop(
            ubsamp + "_" + "nue_to_nue_mc_osc_weights_grad_ubbaseline",
            [
                ubsamp + "_" + "nue_to_nue_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "nue_to_nue_mc_osc_sin2_grad_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.nue_to_nue_osc_weights_grad,
        )  # sorted by true energy
        the_store.add_prop(
            ubsamp + "_" + "numu_to_numu_mc_osc_weights_grad_ubbaseline",
            [
                ubsamp + "_" + "numu_to_numu_mc_osc_factor_ubbaseline",
                ubsamp + "_" + "numu_to_numu_mc_osc_sin2_grad_ubbaseline",
                "effective_radius",
                "effective_theta",
            ],
            f.numu_to_numu_osc_weights_grad,
        )  # sorted by true energy

    the_store.add_prop(
        "dl_NuE_bin_edges",
        [],
        load_sample.load_dl_bin_edges,
    )
    the_store.add_prop(
        "wc_bin_edges",
        [],
        load_sample.load_wc_bin_edges,
    )
    the_store.add_prop(
        "dl_NuE_binned_data",
        [],
        load_sample.load_dl_binned_data,
    )
    the_store.add_prop(
        "wc_binned_data",
        [],
        load_sample.load_wc_binned_data,
    )
    the_store.add_prop("dl_templates", [], load_sample.load_dl_templates)
    the_store.add_prop("wc_templates", [], load_sample.load_wc_templates)
    dl_mc_samples = [
        "nue_nominal_bkg",
        "nue_constrained_bkg",
        "numu_fitted_bkg",
        "tot_nominal_bkg",
        "tot_constrained_bkg",
        "tot_constrained_lee",
    ]
    wc_mc_datasets = [
        "nue_FC_Constr",
        "nue_FC",
        "nue_PC",
        "numu_FC",
        "numu_PC",
        "nue_numu",
    ]
    wc_mc_samples = [
        "bkg",
        "sig",
        "sig_bkg",
    ]
    for sample in dl_mc_samples:
        the_store.add_prop(
            sample + "_dl_template", ["dl_templates"], f.get_item(sample)
        )
    for ds in wc_mc_datasets:
        the_store.add_prop(
            ds + "_data_wc_template", ["wc_binned_data"], f.get_item(ds + "_data")
        )
        for sample in wc_mc_samples:
            key = ds + "_" + sample
            the_store.add_prop(key + "_wc_template", ["wc_templates"], f.get_item(key))

    the_store.add_prop(
        "dl_NuE_cov_cv_map",
        ["tot_nominal_bkg_dl_template", "tot_constrained_lee_dl_template"],
        f.covariance.load_dl_cov,
    )
    the_store.add_prop(
        "wc_nue_numu_cov_cv_map",
        ["nue_numu_sig_bkg_wc_template"],
        f.covariance.load_wc_cov,
    )
    the_store.add_prop(
        "dl_NuE_mc_frac_error", ["dl_NuE_cov_cv_map"], f.get_item("mc_frac_error")
    )
    the_store.add_prop(
        "wc_nue_numu_mc_frac_error",
        ["wc_nue_numu_cov_cv_map"],
        f.get_item("mc_frac_error"),
    )
    the_store.add_prop(
        "dl_NuE_nominal_constrained_frac_cov", ["dl_NuE_cov_cv_map"], f.get_item("cov")
    )
    the_store.add_prop(
        "wc_nue_numu_nominal_frac_cov",
        ["wc_nue_numu_cov_cv_map"],
        f.get_item("frac_cov"),
    )
    # the_store.add_prop("ub_norm_var",
    #     ["ub_NuE_cov_cv_map"],
    #     f.get_item("norm_var")
    # )
    the_store.add_prop(
        "dl_NuE_nominal_constrained_frac_cov_inv",
        ["dl_NuE_nominal_constrained_frac_cov"],
        f.cov_inv,
    )
    the_store.add_prop(
        "wc_nue_numu_nominal_frac_cov_inv", ["wc_nue_numu_nominal_frac_cov"], f.cov_inv
    )

    the_store.add_prop("dl_expect_frac_mc_error", ["dl_NuE_mc_frac_error"], f.identity)
    the_store.add_prop(
        "wc_expect_frac_mc_error", ["wc_nue_numu_mc_frac_error"], f.identity
    )

    the_store.add_prop(
        "dl_true_energy_binning",
        [],
        binning.dl_mb_mc_true_energy_binning,
    )
    the_store.add_prop(
        "wc_true_energy_binning",
        [],
        binning.wc_mb_mc_true_energy_binning,
    )

    the_store.add_prop(
        "dl_nue_to_nue_mb_mc_sorting_info",
        ["nue_to_nue_sorted_mc", "dl_true_energy_binning"],
        binning.ub_sort_mb_mc,
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_mc_sorting_info",
        ["numu_to_nue_sorted_mc", "dl_true_energy_binning"],
        binning.ub_sort_mb_mc,
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_mc_sorting_info",
        ["nue_to_nue_sorted_mc", "wc_true_energy_binning"],
        binning.ub_sort_mb_mc,
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_mc_sorting_info",
        ["numu_to_nue_sorted_mc", "wc_true_energy_binning"],
        binning.ub_sort_mb_mc,
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_mc_sorting_info",
        ["numu_to_numu_sorted_mc", "wc_true_energy_binning"],
        binning.ub_sort_mb_mc,
    )

    the_store.add_prop(
        "dl_nue_to_nue_mb_sorted_mc",
        ["dl_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "dl_nue_to_nue_mb_mc_slices",
        ["dl_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "dl_nue_to_nue_mb_mc_idx_sort",
        ["dl_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(2),
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_sorted_mc",
        ["dl_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_mc_slices",
        ["dl_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_mc_idx_sort",
        ["dl_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(2),
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_sorted_mc",
        ["wc_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_mc_slices",
        ["wc_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_mc_idx_sort",
        ["wc_nue_to_nue_mb_mc_sorting_info"],
        f.get_item(2),
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_sorted_mc",
        ["wc_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_mc_slices",
        ["wc_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_mc_idx_sort",
        ["wc_numu_to_nue_mb_mc_sorting_info"],
        f.get_item(2),
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_sorted_mc",
        ["wc_numu_to_numu_mb_mc_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_mc_slices",
        ["wc_numu_to_numu_mb_mc_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_mc_idx_sort",
        ["wc_numu_to_numu_mb_mc_sorting_info"],
        f.get_item(2),
    )

    # load dl osc weight maps
    the_store.add_prop(
        "dl_nue_to_nue_mb_mc_cv_weights",
        ["nue_to_nue_mc_cv_weights", "dl_nue_to_nue_mb_mc_idx_sort"],
        lambda weights, idx: weights[idx],
    )  # sorted by true energy
    the_store.add_prop(
        "dl_numu_to_nue_mb_mc_cv_weights",
        ["numu_to_nue_mc_cv_weights", "dl_numu_to_nue_mb_mc_idx_sort"],
        lambda weights, idx: weights[idx],
    )  # sorted by true energy
    the_store.add_prop(
        "dl_nue_to_nue_mb_cv_expect",
        ["dl_nue_to_nue_mb_mc_cv_weights", "dl_nue_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_cv_expect",
        ["dl_numu_to_nue_mb_mc_cv_weights", "dl_numu_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "dl_nue_to_nue_mb_osc_expect_grad",
        ["dl_nue_to_nue_mb_mc_weights_grad_ubbaseline", "dl_nue_to_nue_mb_mc_slices"],
        f.raw_center_grad,
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_osc_expect_grad",
        ["dl_numu_to_nue_mb_mc_weights_grad_ubbaseline", "dl_numu_to_nue_mb_mc_slices"],
        f.raw_center_grad,
    )
    the_store.add_prop(
        "dl_nue_to_nue_mb_osc_expect",
        ["dl_nue_to_nue_mb_mc_weights_ubbaseline", "dl_nue_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "dl_numu_to_nue_mb_osc_expect",
        ["dl_numu_to_nue_mb_mc_weights_ubbaseline", "dl_numu_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "dl_nue_to_nue_osc_weight_map_grad",
        [
            "dl_nue_to_nue_mb_cv_expect",
            "dl_nue_to_nue_mb_osc_expect_grad",
        ],
        f.ub_osc_weight_map_grad,
    )
    the_store.add_prop(
        "dl_numu_to_nue_osc_weight_map_grad",
        [
            "dl_nue_to_nue_mb_cv_expect",
            "dl_numu_to_nue_mb_osc_expect_grad",
        ],
        f.ub_osc_weight_map_grad,
    )
    the_store.add_prop(
        "dl_nue_to_nue_osc_weight_map",
        [
            "dl_nue_to_nue_mb_cv_expect",
            "dl_nue_to_nue_mb_osc_expect",
        ],
        f.ub_osc_weight_map,
    )
    the_store.add_prop(
        "dl_numu_to_nue_osc_weight_map",
        [
            "dl_nue_to_nue_mb_cv_expect",
            "dl_numu_to_nue_mb_osc_expect",
        ],
        f.ub_osc_weight_map,
    )

    # load wc osc weight maps
    the_store.add_prop(
        "wc_nue_to_nue_mb_mc_cv_weights",
        ["nue_to_nue_mc_cv_weights", "wc_nue_to_nue_mb_mc_idx_sort"],
        lambda weights, idx: weights[idx],
    )  # sorted by true energy
    the_store.add_prop(
        "wc_numu_to_nue_mb_mc_cv_weights",
        ["numu_to_nue_mc_cv_weights", "wc_numu_to_nue_mb_mc_idx_sort"],
        lambda weights, idx: weights[idx],
    )  # sorted by true energy
    the_store.add_prop(
        "wc_numu_to_numu_mb_mc_cv_weights",
        ["numu_to_numu_mc_cv_weights", "wc_numu_to_numu_mb_mc_idx_sort"],
        lambda weights, idx: weights[idx],
    )  # sorted by true energy
    the_store.add_prop(
        "wc_nue_to_nue_mb_cv_expect",
        ["wc_nue_to_nue_mb_mc_cv_weights", "wc_nue_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_cv_expect",
        ["wc_numu_to_nue_mb_mc_cv_weights", "wc_numu_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_cv_expect",
        ["wc_numu_to_numu_mb_mc_cv_weights", "wc_numu_to_numu_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_osc_expect_grad",
        ["wc_nue_to_nue_mb_mc_weights_grad_ubbaseline", "wc_nue_to_nue_mb_mc_slices"],
        f.raw_center_grad,
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_osc_expect_grad",
        ["wc_numu_to_nue_mb_mc_weights_grad_ubbaseline", "wc_numu_to_nue_mb_mc_slices"],
        f.raw_center_grad,
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_osc_expect_grad",
        [
            "wc_numu_to_numu_mb_mc_weights_grad_ubbaseline",
            "wc_numu_to_numu_mb_mc_slices",
        ],
        f.raw_center_grad,
    )
    the_store.add_prop(
        "wc_nue_to_nue_mb_osc_expect",
        ["wc_nue_to_nue_mb_mc_weights_ubbaseline", "wc_nue_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_numu_to_nue_mb_osc_expect",
        ["wc_numu_to_nue_mb_mc_weights_ubbaseline", "wc_numu_to_nue_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_numu_to_numu_mb_osc_expect",
        ["wc_numu_to_numu_mb_mc_weights_ubbaseline", "wc_numu_to_numu_mb_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "wc_nue_to_nue_osc_weight_map_grad",
        [
            "wc_nue_to_nue_mb_cv_expect",
            "wc_nue_to_nue_mb_osc_expect_grad",
        ],
        f.ub_osc_weight_map_grad,
    )
    the_store.add_prop(
        "wc_numu_to_nue_osc_weight_map_grad",
        [
            "wc_nue_to_nue_mb_cv_expect",
            "wc_numu_to_nue_mb_osc_expect_grad",
        ],
        f.ub_osc_weight_map_grad,
    )
    the_store.add_prop(
        "wc_numu_to_numu_osc_weight_map_grad",
        [
            "wc_numu_to_numu_mb_cv_expect",
            "wc_numu_to_numu_mb_osc_expect",
        ],
        f.ub_osc_weight_map_grad,
    )
    the_store.add_prop(
        "wc_nue_to_nue_osc_weight_map",
        [
            "wc_nue_to_nue_mb_cv_expect",
            "wc_nue_to_nue_mb_osc_expect",
        ],
        f.ub_osc_weight_map,
    )
    the_store.add_prop(
        "wc_numu_to_nue_osc_weight_map",
        [
            "wc_nue_to_nue_mb_cv_expect",
            "wc_numu_to_nue_mb_osc_expect",
        ],
        f.ub_osc_weight_map,
    )
    the_store.add_prop(
        "wc_numu_to_numu_osc_weight_map",
        [
            "wc_numu_to_numu_mb_cv_expect",
            "wc_numu_to_numu_mb_osc_expect",
        ],
        f.ub_osc_weight_map,
    )

    # DL mc smearing info
    the_store.add_prop(
        "dl_NuE_nue_mc",
        [],
        load_sample.load_dl_nue_mc,
    )
    the_store.add_prop(
        "dl_NuE_nue_mc_smearing_sorting_info",
        ["dl_NuE_nue_mc", "dl_true_energy_binning", "dl_NuE_bin_edges"],
        binning.dl_sort_smearing_mc,
    )
    the_store.add_prop(
        "dl_NuE_nue_smearing_sorted_mc",
        ["dl_NuE_nue_mc_smearing_sorting_info"],
        f.get_item(0),
    )
    the_store.add_prop(
        "dl_NuE_nue_sorted_mc", ["dl_NuE_nue_mc_smearing_sorting_info"], f.get_item(0)
    )
    the_store.add_prop(
        "dl_NuE_nue_smearing_mc_slices",
        ["dl_NuE_nue_mc_smearing_sorting_info"],
        f.get_item(1),
    )
    the_store.add_prop(
        "dl_NuE_nue_mc_slices", ["dl_NuE_nue_mc_smearing_sorting_info"], f.get_item(2)
    )
    the_store.add_prop(
        "dl_NuE_nue_cv_weights", ["dl_NuE_nue_sorted_mc"], f.get_item("cv_weight")
    )
    the_store.add_prop(
        "dl_NuE_nue_cv_expect",
        ["dl_NuE_nue_cv_weights", "dl_NuE_nue_mc_slices"],
        f.raw_center,
    )
    the_store.add_prop(
        "dl_NuE_nue_constrained_cv_ratio",
        ["nue_constrained_bkg_dl_template", "dl_NuE_nue_cv_expect"],
        f.compute_constrained_cv_ratio,
    )
    the_store.add_prop(
        "dl_NuE_nue_smearing_matrix",
        [
            "dl_NuE_nue_smearing_sorted_mc",
            "dl_NuE_nue_smearing_mc_slices",
            "dl_true_energy_binning",
            "dl_NuE_bin_edges",
        ],
        f.dl_build_smearing_matrix,
    )
    # WC mc smearing info
    the_store.add_prop(
        "wc_nue_numu_smearing_matrices",
        ["wc_true_energy_binning", "wc_bin_edges"],
        load_sample.load_wc_smearing_matrices,
    )
    for ds in wc_mc_datasets:
        if ds == "nue_FC_Constr" or ds == "nue_numu":
            continue
        the_store.add_prop(
            "wc_" + ds + "_smearing_matrix",
            ["wc_nue_numu_smearing_matrices"],
            f.get_item(ds),
        )
        the_store.add_prop(
            "wc_" + ds + "_DR_SM_ratio",
            [ds + "_sig_wc_template", "wc_" + ds + "_smearing_matrix"],
            f.compute_DR_SM_ratio,
        )

    the_store.add_prop(
        "dl_NuE_numu_to_nue_cv_expect_grad",
        ["dl_numu_to_nue_osc_weight_map_grad", "dl_NuE_nue_smearing_matrix"],
        lambda x, y: np.sum(np.expand_dims(x, 0) * np.expand_dims(y, 2), axis=1),
    )
    the_store.add_prop(
        "dl_NuE_nue_to_nue_cv_expect_grad",
        ["dl_nue_to_nue_osc_weight_map_grad", "dl_NuE_nue_smearing_matrix"],
        lambda x, y: np.sum(np.expand_dims(x, 0) * np.expand_dims(y, 2), axis=1),
    )
    the_store.add_prop(
        "dl_NuE_numu_to_nue_cv_expect",
        ["dl_numu_to_nue_osc_weight_map", "dl_NuE_nue_smearing_matrix"],
        lambda x, y: np.sum(np.expand_dims(x, 0) * y, axis=1),
    )
    the_store.add_prop(
        "dl_NuE_nue_to_nue_cv_expect",
        ["dl_nue_to_nue_osc_weight_map", "dl_NuE_nue_smearing_matrix"],
        lambda x, y: np.sum(np.expand_dims(x, 0) * y, axis=1),
    )
    the_store.add_prop(
        "dl_NuE_numu_to_nue_constrained_expect_grad",
        ["dl_NuE_numu_to_nue_cv_expect_grad", "dl_NuE_nue_constrained_cv_ratio"],
        lambda x, y: x * np.expand_dims(y, 1),
    )
    the_store.add_prop(
        "dl_NuE_nue_to_nue_constrained_expect_grad",
        ["dl_NuE_nue_to_nue_cv_expect_grad", "dl_NuE_nue_constrained_cv_ratio"],
        lambda x, y: x * np.expand_dims(y, 1),
    )
    the_store.add_prop(
        "dl_NuE_numu_to_nue_constrained_expect",
        ["dl_NuE_numu_to_nue_cv_expect", "dl_NuE_nue_constrained_cv_ratio"],
        lambda x, y: x * y,
    )
    the_store.add_prop(
        "dl_NuE_nue_to_nue_constrained_expect",
        ["dl_NuE_nue_to_nue_cv_expect", "dl_NuE_nue_constrained_cv_ratio"],
        lambda x, y: x * y,
    )

    for ds in wc_mc_datasets:
        if ds == "nue_FC_Constr" or ds == "nue_numu":
            continue
        if "nue" in ds:
            the_store.add_prop(
                "wc_" + ds + "_numu_to_nue_cv_expect_grad",
                ["wc_numu_to_nue_osc_weight_map_grad", "wc_" + ds + "_smearing_matrix"],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * np.expand_dims(y, 2), axis=1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_nue_to_nue_cv_expect_grad",
                ["wc_nue_to_nue_osc_weight_map_grad", "wc_" + ds + "_smearing_matrix"],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * np.expand_dims(y, 2), axis=1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_nue_cv_expect",
                ["wc_numu_to_nue_osc_weight_map", "wc_" + ds + "_smearing_matrix"],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * y, axis = 1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_nue_to_nue_cv_expect",
                ["wc_nue_to_nue_osc_weight_map", "wc_" + ds + "_smearing_matrix"],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * y, axis = 1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_nue_constrained_expect_grad",
                [
                    "wc_" + ds + "_numu_to_nue_cv_expect_grad",
                    "wc_" + ds + "_DR_SM_ratio",
                ],
                lambda x, y: x * np.expand_dims(y, 1),
            )
            the_store.add_prop(
                "wc_" + ds + "_nue_to_nue_constrained_expect_grad",
                [
                    "wc_" + ds + "_nue_to_nue_cv_expect_grad",
                    "wc_" + ds + "_DR_SM_ratio",
                ],
                lambda x, y: x * np.expand_dims(y, 1),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_nue_constrained_expect",
                ["wc_" + ds + "_numu_to_nue_cv_expect", "wc_" + ds + "_DR_SM_ratio"],
                lambda x, y: x * y,
            )
            the_store.add_prop(
                "wc_" + ds + "_nue_to_nue_constrained_expect",
                ["wc_" + ds + "_nue_to_nue_cv_expect", "wc_" + ds + "_DR_SM_ratio"],
                lambda x, y: x * y,
            )
        elif "numu" in ds:
            the_store.add_prop(
                "wc_" + ds + "_numu_to_numu_cv_expect_grad",
                [
                    "wc_numu_to_numu_osc_weight_map_grad",
                    "wc_" + ds + "_smearing_matrix",
                ],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * np.expand_dims(y, 2), axis=1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_numu_cv_expect",
                ["wc_numu_to_numu_osc_weight_map", "wc_" + ds + "_smearing_matrix"],
                lambda x, y: np.sum(
                    np.expand_dims(x, 0) * y, axis=1
                ),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_numu_constrained_expect_grad",
                [
                    "wc_" + ds + "_numu_to_numu_cv_expect_grad",
                    "wc_" + ds + "_DR_SM_ratio",
                ],
                lambda x, y: x * np.expand_dims(y, 1),
            )
            the_store.add_prop(
                "wc_" + ds + "_numu_to_numu_constrained_expect",
                ["wc_" + ds + "_numu_to_numu_cv_expect", "wc_" + ds + "_DR_SM_ratio"],
                lambda x, y: x * y,
            )

    the_store.add_prop(
        "dl_unscaled_expect_grad",
        [
            "dl_NuE_nue_to_nue_constrained_expect_grad",
            "numu_fitted_bkg_dl_template",
            "dl_NuE_numu_to_nue_constrained_expect_grad",
        ],
        f.ub_expect_nue_grad,
    )
    the_store.add_prop(
        "dl_unscaled_expect",
        [
            "dl_NuE_nue_to_nue_constrained_expect",
            "numu_fitted_bkg_dl_template",
            "dl_NuE_numu_to_nue_constrained_expect",
        ],
        f.ub_expect_nue,
    )
    for ds in ["nue_FC", "nue_PC"]:
        the_store.add_prop(
            "wc_" + ds + "_unscaled_expect_grad",
            [
                "wc_" + ds + "_nue_to_nue_constrained_expect_grad",
                ds + "_bkg_wc_template",
                "wc_" + ds + "_numu_to_nue_constrained_expect_grad",
            ],
            f.ub_expect_nue_grad,
        )
        the_store.add_prop(
            "wc_" + ds + "_unscaled_expect",
            [
                "wc_" + ds + "_nue_to_nue_constrained_expect",
                ds + "_bkg_wc_template",
                "wc_" + ds + "_numu_to_nue_constrained_expect",
            ],
            f.ub_expect_nue,
        )
    for ds in ["numu_FC", "numu_PC"]:
        the_store.add_prop(
            "wc_" + ds + "_unscaled_expect_grad",
            [
                "wc_" + ds + "_numu_to_numu_constrained_expect_grad",
                ds + "_bkg_wc_template",
            ],
            f.ub_expect_numu_grad,
        )
        the_store.add_prop(
            "wc_" + ds + "_unscaled_expect",
            [
                "wc_" + ds + "_numu_to_numu_constrained_expect",
                ds + "_bkg_wc_template",
            ],
            f.ub_expect_numu,
        )

    the_store.add_prop(
        "wc_unscaled_expect",
        [
            "wc_nue_FC_unscaled_expect",
            "wc_nue_PC_unscaled_expect",
            "wc_numu_FC_unscaled_expect",
            "wc_numu_PC_unscaled_expect",
        ],
        f.wc_expect,
    )
    the_store.add_prop(
        "wc_unscaled_expect_grad",
        [
            "wc_nue_FC_unscaled_expect_grad",
            "wc_nue_PC_unscaled_expect_grad",
            "wc_numu_FC_unscaled_expect_grad",
            "wc_numu_PC_unscaled_expect_grad",
        ],
        f.wc_expect_grad,
    )

    # the_store.add_prop(
    #     "ub_nominal_expect_grad",
    #     ["ub_unscaled_expect_grad", "ub_norm"],
    #     f.apply_norm_grad
    # )
    # the_store.add_prop(
    #     "ub_nominal_expect",
    #     ["ub_unscaled_expect", "ub_norm"],
    #     f.apply_norm
    # )
    the_store.add_prop(
        "dl_nominal_expect_grad", ["dl_unscaled_expect_grad"], f.identity
    )
    the_store.add_prop("dl_nominal_expect", ["dl_unscaled_expect"], f.identity)
    the_store.add_prop(
        "wc_nominal_expect_grad", ["wc_unscaled_expect_grad"], f.identity
    )
    the_store.add_prop("wc_nominal_expect", ["wc_unscaled_expect"], f.identity)

    the_store.add_prop("dl_ratio_vec", ["dl_ratios"], np.array)
    the_store.add_prop("wc_ratio_vec", ["wc_ratios"], np.array)

    the_store.add_prop(
        "dl_expect_grad",
        ["dl_nominal_expect_grad", "dl_ratio_vec"],
        f.ub_apply_sys_norms_grad,
    )
    the_store.add_prop(
        "dl_expect", ["dl_nominal_expect", "dl_ratio_vec"], f.ub_apply_sys_norms
    )
    the_store.add_prop(
        "wc_expect_grad",
        ["wc_nominal_expect_grad", "wc_ratio_vec"],
        f.ub_apply_sys_norms_grad,
    )
    the_store.add_prop(
        "wc_expect", ["wc_nominal_expect", "wc_ratio_vec"], f.ub_apply_sys_norms
    )

    the_store.add_prop(
        "dl_eigen_decomp", ["dl_NuE_nominal_constrained_frac_cov"], scipy.linalg.eigh
    )
    the_store.add_prop("dl_eigen_values", ["dl_eigen_decomp"], f.get_item(0))
    the_store.add_prop("dl_eigen_vectors", ["dl_eigen_decomp"], f.get_item(1))
    the_store.add_prop(
        "dl_eigen_nonzero_mask", ["dl_eigen_values"], f.eigen_nonzero_mask
    )
    the_store.add_prop("dl_n_modes", ["dl_eigen_nonzero_mask"], f.eigen_count_nonzero)
    the_store.add_prop(
        "dl_nonzero_eigen_values",
        ["dl_eigen_values", "dl_eigen_nonzero_mask"],
        f.eigen_nonzero_lambdas,
    )
    the_store.add_prop(
        "dl_nonzero_eigen_vectors",
        ["dl_eigen_vectors", "dl_eigen_nonzero_mask"],
        f.eigen_nonzero_vectors,
    )
    the_store.add_prop(
        "dl_zero_eigen_vector",
        ["dl_eigen_vectors", "dl_eigen_nonzero_mask"],
        f.eigen_zero_vector,
    )
    the_store.add_prop(
        "dl_modes", ["dl_nonzero_eigen_vectors"], f.identity, cache_size=0
    )
    the_store.add_prop(
        "dl_gauss_likelihood_prefactor",
        ["dl_nonzero_eigen_values"],
        f.eigen_gauss_prefactor,
    )
    the_store.add_prop(
        "wc_eigen_decomp", ["wc_nue_numu_nominal_frac_cov"], scipy.linalg.eigh
    )
    the_store.add_prop("wc_eigen_values", ["wc_eigen_decomp"], f.get_item(0))
    the_store.add_prop("wc_eigen_vectors", ["wc_eigen_decomp"], f.get_item(1))
    the_store.add_prop(
        "wc_eigen_nonzero_mask", ["wc_eigen_values"], f.eigen_nonzero_mask
    )
    the_store.add_prop("wc_n_modes", ["wc_eigen_nonzero_mask"], f.eigen_count_nonzero)
    the_store.add_prop(
        "wc_nonzero_eigen_values",
        ["wc_eigen_values", "wc_eigen_nonzero_mask"],
        f.eigen_nonzero_lambdas,
    )
    the_store.add_prop(
        "wc_nonzero_eigen_vectors",
        ["wc_eigen_vectors", "wc_eigen_nonzero_mask"],
        f.eigen_nonzero_vectors,
    )
    the_store.add_prop(
        "wc_zero_eigen_vector",
        ["wc_eigen_vectors", "wc_eigen_nonzero_mask"],
        f.eigen_zero_vector,
    )
    the_store.add_prop(
        "wc_modes", ["wc_nonzero_eigen_vectors"], f.identity, cache_size=0
    )
    the_store.add_prop(
        "wc_pred_mask",
        [
            "nue_numu_sig_bkg_wc_template",
        ],
        f.ub_expect_mask,
    )
    the_store.add_prop(
        "wc_modes_masked", ["wc_modes", "wc_pred_mask"], f.ub_mask_modes, cache_size=0
    )
    the_store.add_prop(
        "wc_gauss_likelihood_prefactor",
        ["wc_nonzero_eigen_values"],
        f.eigen_gauss_prefactor,
    )

    # Spin up the caches
    the_store.initialize(keep_cache=True)

    return the_store
