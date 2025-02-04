import numpy as np
import scipy

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


##################################################################
# MiniBooNE data
mb_data_year = "2020"
path_mb_data_release_fhc = f"OscTools.include.MB_data_release_{mb_data_year}.fhcmode"
path_mb_data_release_rhc = f"OscTools.include.MB_data_release_{mb_data_year}.rhcmode"

# reco neutrino energy, true neutrino energy, neutrino beampipe, and event weight
mb_mc_data_release = np.genfromtxt(
    files(path_mb_data_release_fhc)
    .joinpath("miniboone_numunuefullosc_ntuple.txt")
    .open(),
)

bin_edges = np.genfromtxt(
    files(path_mb_data_release_fhc)
    .joinpath("miniboone_binboundaries_nue_lowe.txt")
    .open(),
)

bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
bin_width = np.diff(bin_edges)
mb_nue_analysis_data = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_nuedata_lowe.txt").open(),
)
mb_numu_analyis_data = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_numudata.txt").open(),
)
mb_nue_analysis_predicted_background = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_nuebgr_lowe.txt").open(),
)
mb_numu_analyis_prediction = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_numu.txt").open(),
)

fractional_covariance_matrix = np.genfromtxt(
    files(path_mb_data_release_fhc)
    .joinpath("miniboone_full_fractcovmatrix_nu_lowe.txt")
    .open(),
)

bin_edges_reco = np.array(
    [
        0.200,
        0.250,
        0.300,
        0.350,
        0.400,
        0.450,
        0.500,
        0.600,
        0.800,
        1.000,
        1.500,
        2.000,
        2.500,
        3.000,
    ]
)
bin_centers_reco = bin_edges_reco[:-1] + np.diff(bin_edges_reco) / 2.0
bin_width_reco = np.diff(bin_edges_reco)

bin_edges_numu = np.array([0, 0.5, 0.7, 0.9, 1.1, 1.3, 1.50, 1.7, 1.9]) * 1e3  # MeV


##################
# MC data releases

# FHC mode
MC_nue_bkg_tot = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_nuebgr_lowe.txt").open()
)
MC_numu_bkg_tot = np.genfromtxt(
    files(path_mb_data_release_fhc).joinpath("miniboone_numu.txt").open()
)

# RHC mode
MC_nuebar_bkg_tot = np.genfromtxt(
    files(path_mb_data_release_rhc).joinpath("miniboone_nuebarbgr_lowe.txt").open()
)
MC_numubar_bkg_tot = np.genfromtxt(
    files(path_mb_data_release_rhc).joinpath("miniboone_numubar.txt").open()
)


def StackCovarianceMatrix(big_covariance, n_signal, n_numu):
    covariance = np.zeros([n_signal + n_numu, n_signal + n_numu])

    covariance[0:n_signal, 0:n_signal] = (
        big_covariance[0:n_signal, 0:n_signal]
        + big_covariance[n_signal : 2 * n_signal, 0:n_signal]
        + big_covariance[0:n_signal, n_signal : 2 * n_signal]
        + big_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    )
    covariance[n_signal : (n_signal + n_numu), 0:n_signal] = (
        big_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal]
        + big_covariance[
            2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal
        ]
    )
    covariance[0:n_signal, n_signal : (n_signal + n_numu)] = (
        big_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
        + big_covariance[
            n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)
        ]
    )
    covariance[n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)] = (
        big_covariance[
            2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
        ]
    )

    # assert np.abs(np.sum(covariance) - np.sum(big_covariance)) < 1.0e-3

    return covariance


def MassageCovarianceMatrix(big_covariance, n_signal, n_numu, remove_HE=False):

    if remove_HE:
        n_total = n_signal + n_numu - 2
        n_total_big = n_signal * 2 + n_numu - 4

        covariance = np.zeros([n_total * 2, n_total * 2])

        covariance[0:n_total, 0:n_total] = StackCovarianceMatrix(
            big_covariance[0:n_total_big, 0:n_total_big], n_signal, n_numu
        )
        covariance[n_total : (2 * n_total), 0:n_total] = StackCovarianceMatrix(
            big_covariance[n_total_big : (2 * n_total_big), 0:n_total_big],
            n_signal,
            n_numu,
        )
        covariance[0:n_total, n_total : (2 * n_total)] = StackCovarianceMatrix(
            big_covariance[0:n_total_big, n_total_big : (2 * n_total_big)],
            n_signal,
            n_numu,
        )
        covariance[n_total : (2 * n_total), n_total : (2 * n_total)] = (
            StackCovarianceMatrix(
                big_covariance[
                    n_total_big : (2 * n_total_big), n_total_big : (2 * n_total_big)
                ],
                n_signal,
                n_numu,
            )
        )
    else:
        n_total = n_signal + n_numu
        n_total_big = n_signal * 2 + n_numu

        covariance = np.zeros([n_total * 2, n_total * 2])

        covariance[0:n_total, 0:n_total] = StackCovarianceMatrix(
            big_covariance[0:n_total_big, 0:n_total_big], n_signal, n_numu
        )
        covariance[n_total : (2 * n_total), 0:n_total] = StackCovarianceMatrix(
            big_covariance[n_total_big : (2 * n_total_big), 0:n_total_big],
            n_signal,
            n_numu,
        )
        covariance[0:n_total, n_total : (2 * n_total)] = StackCovarianceMatrix(
            big_covariance[0:n_total_big, n_total_big : (2 * n_total_big)],
            n_signal,
            n_numu,
        )
        covariance[n_total : (2 * n_total), n_total : (2 * n_total)] = (
            StackCovarianceMatrix(
                big_covariance[
                    n_total_big : (2 * n_total_big), n_total_big : (2 * n_total_big)
                ],
                n_signal,
                n_numu,
            )
        )
    # assert np.abs(np.sum(covariance) - np.sum(big_covariance)) < 1.0e-3
    return covariance


def chi2_MiniBooNE(
    MC_nue_app, MC_nue_dis=None, MC_numu_dis=None, mode="fhc", year="2020"
):
    """chi2_MiniBooNE Get MiniBOoNE chi2 from a given data release and running mode (FHC, RHC)

    Parameters
    ----------
    MC_nue_app:np.array
        Monte Carlo prediction for the numu -> nu_e appearance rate
    MC_nue_dis:np.array, default None
        Monte Carlo prediction for the nu_e disappearance rate
    MC_numu_dis:np.array, default None
        Monte Carlo prediction for the nu_mu disappearance rate

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    mode = mode.lower()
    bar = "bar" if mode == "rhc" else ""

    nue_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.{mode}mode")
        .joinpath(f"miniboone_nue{bar}data_lowe.txt")
        .open()
    )
    numu_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.{mode}mode")
        .joinpath(f"miniboone_numu{bar}data.txt")
        .open()
    )

    fract_covariance = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.{mode}mode")
        .joinpath(f"miniboone_full_fractcovmatrix_nu{bar}_lowe.txt")
        .open()
    )

    # # energy bins -- same for nu and nubar
    # bin_e = np.genfromtxt(
    #     files(#         f"MiniTools.include.MB_data_release_{year}.{mode}mode").joinpath(#         "miniboone_binboundaries_nue_lowe.txt").open()     )
    # )

    # NOTE:new method from Tao.
    if MC_nue_dis is not None:
        nue_bkg = MC_nue_dis
    else:
        nue_bkg = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.{mode}mode")
            .joinpath(f"miniboone_nue{bar}bgr_lowe.txt")
            .open()
        )

    if MC_numu_dis is not None:
        numu_MC = MC_numu_dis
    else:
        numu_MC = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.{mode}mode")
            .joinpath(f"miniboone_numu{bar}.txt")
            .open()
        )

    NP_diag_matrix = np.diag(np.concatenate([MC_nue_app, nue_bkg * 0.0, numu_MC * 0.0]))
    tot_diag_matrix = np.diag(np.concatenate([MC_nue_app, nue_bkg, numu_MC]))

    rescaled_covariance = np.dot(
        tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix)
    )
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(MC_nue_app)
    n_numu = len(numu_MC)

    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal + n_numu, n_signal + n_numu])
    error_matrix[0:n_signal, 0:n_signal] = (
        rescaled_covariance[0:n_signal, 0:n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, 0:n_signal]
        + rescaled_covariance[0:n_signal, n_signal : 2 * n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    )
    error_matrix[n_signal : (n_signal + n_numu), 0:n_signal] = (
        rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal]
        + rescaled_covariance[
            2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal
        ]
    )
    error_matrix[0:n_signal, n_signal : (n_signal + n_numu)] = (
        rescaled_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
        + rescaled_covariance[
            n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)
        ]
    )
    error_matrix[n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)] = (
        rescaled_covariance[
            2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
        ]
    )

    # compute residuals
    residuals = np.concatenate(
        [nue_data - (MC_nue_app + nue_bkg), (numu_data - numu_MC)]
    )

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(
        residuals, np.dot(inv_cov, residuals)
    )  # + np.log(np.linalg.det(error_matrix))

    if chi2 >= 0:
        return chi2
    else:
        return 1e10


def chi2_MiniBooNE_combined(
    MC_nue_app,
    MC_nuebar_app,
    MC_nue_dis=None,
    MC_numu_dis=None,
    MC_nuebar_dis=None,
    MC_numubar_dis=None,
    year="2020",
):
    """chi2_MiniBooNE_combined Get MiniBooNE chi2 from a given data release from FHC + RHC

    Parameters
    ----------
    MC_nue_app:np.array
        Monte Carlo prediction for the numu -> nu_e appearance rate
    MC_nue_dis:np.array, default None
        Monte Carlo prediction for the nu_e disappearance rate
    MC_numu_dis:np.array, default None
        Monte Carlo prediction for the nu_mu disappearance rate

    MC_nuebar_app:np.array
        Monte Carlo prediction for the numubar -> nu_ebar appearance rate
    MC_nuebar_dis:np.array, default None
        Monte Carlo prediction for the nu_ebar disappearance rate
    MC_numubar_dis:np.array, default None
        Monte Carlo prediction for the nu_mubar disappearance rate

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    ##########################################
    # Load neutrino data
    nue_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.combined")
        .joinpath(f"miniboone_nuedata_lowe.txt")
        .open()
    )
    numu_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.combined")
        .joinpath(f"miniboone_numudata.txt")
        .open()
    )

    ##########################################
    # Load antineutrino data
    nuebar_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.combined")
        .joinpath(f"miniboone_nuebardata_lowe.txt")
        .open()
    )
    numubar_data = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.combined")
        .joinpath(f"miniboone_numubardata.txt")
        .open()
    )

    ##########################################
    # Load covariance matrix
    fract_covariance = np.genfromtxt(
        files(f"MiniTools.include.MB_data_release_{year}.combined")
        .joinpath(f"miniboone_full_fractcovmatrix_combined_lowe.txt")
        .open()
    )

    if MC_nue_dis is not None:
        nue_bkg = MC_nue_dis
    else:
        nue_bkg = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.combined")
            .joinpath(f"miniboone_nuebgr_lowe.txt")
            .open()
        )

    if MC_numu_dis is not None:
        numu_MC = MC_numu_dis
    else:
        numu_MC = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.combined")
            .joinpath(f"miniboone_numu.txt")
            .open()
        )

    if MC_nuebar_dis is not None:
        nuebar_bkg = MC_nuebar_dis
    else:
        nuebar_bkg = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.combined")
            .joinpath(f"miniboone_nuebarbgr_lowe.txt")
            .open()
        )

    if MC_numubar_dis is not None:
        numubar_MC = MC_numubar_dis
    else:
        numubar_MC = np.genfromtxt(
            files(f"MiniTools.include.MB_data_release_{year}.combined")
            .joinpath(f"miniboone_numubar.txt")
            .open()
        )

    NP_diag_matrix = np.diag(
        np.concatenate(
            [
                MC_nue_app,
                nue_bkg * 0.0,
                numu_MC * 0.0,
                MC_nuebar_app,
                nuebar_bkg * 0.0,
                numubar_MC * 0.0,
            ]
        )
    )
    tot_diag_matrix = np.diag(
        np.concatenate(
            [MC_nue_app, nue_bkg, numu_MC, MC_nuebar_app, nuebar_bkg, numubar_MC]
        )
    )

    rescaled_covariance = np.dot(
        tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix)
    )
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(MC_nue_app)
    n_numu = len(numu_MC)

    error_matrix = MassageCovarianceMatrix(rescaled_covariance, n_signal, n_numu)

    # compute residuals
    residuals = np.concatenate(
        [
            nue_data - (MC_nue_app + nue_bkg),
            (numu_data - numu_MC),
            nuebar_data - (MC_nuebar_app + nuebar_bkg),
            (numubar_data - numubar_MC),
        ]
    )

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(
        residuals, np.dot(inv_cov, residuals)
    )  # + np.log(np.linalg.det(error_matrix))

    if chi2 >= 0:
        return chi2
    else:
        return 1e10


def get_pval(rates_dic, ndof=8.7):
    MB_chi2 = chi2_MiniBooNE_combined(
        MC_nue_app=rates_dic["MC_nue_app"],
        MC_nuebar_app=rates_dic["MC_nuebar_app"],
        MC_nue_dis=rates_dic["MC_nue_bkg_total_w_dis"],
        MC_numu_dis=rates_dic["MC_numu_bkg_total_w_dis"],
        MC_nuebar_dis=rates_dic["MC_nuebar_bkg_total_w_dis"],
        MC_numubar_dis=rates_dic["MC_numubar_bkg_total_w_dis"],
        year="2020",
    )
    return scipy.stats.chi2.sf(MB_chi2, ndof)


def get_pval_nu(rates_dic, ndof=8.7):
    MB_chi2 = chi2_MiniBooNE_combined(
        MC_nue_app=rates_dic["MC_nue_app"],
        MC_nue_dis=rates_dic["MC_nue_bkg_total_w_dis"],
        MC_numu_dis=rates_dic["MC_numu_bkg_total_w_dis"],
        year="2020",
    )
    return scipy.stats.chi2.sf(MB_chi2, ndof)
