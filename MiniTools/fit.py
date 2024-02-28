import numpy as np
from importlib.resources import open_text


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


def MassageCovarianceMatrix(big_covariance, n_signal, n_numu):
    n_total = n_signal + n_numu
    n_total_big = n_signal * 2 + n_numu

    covariance = np.zeros([n_total * 2, n_total * 2])

    covariance[0:n_total, 0:n_total] = StackCovarianceMatrix(
        big_covariance[0:n_total_big, 0:n_total_big], n_signal, n_numu
    )
    covariance[n_total : (2 * n_total), 0:n_total] = StackCovarianceMatrix(
        big_covariance[n_total_big : (2 * n_total_big), 0:n_total_big], n_signal, n_numu
    )
    covariance[0:n_total, n_total : (2 * n_total)] = StackCovarianceMatrix(
        big_covariance[0:n_total_big, n_total_big : (2 * n_total_big)], n_signal, n_numu
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
    MC_nue_app : np.array
        Monte Carlo prediction for the numu -> nu_e appearance rate
    MC_nue_dis : np.array, default None
        Monte Carlo prediction for the nu_e disappearance rate
    MC_numu_dis : np.array, default None
        Monte Carlo prediction for the nu_mu disappearance rate

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    mode = mode.lower()
    bar = "bar" if mode == "rhc" else ""

    nue_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_numu{bar}data.txt",
        )
    )

    fract_covariance = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_full_fractcovmatrix_nu{bar}_lowe.txt",
        )
    )

    # # energy bins -- same for nu and nubar
    # bin_e = np.genfromtxt(
    #     open_text(
    #         f"MiniTools.include.MB_data_release_{year}.{mode}mode",
    #         "miniboone_binboundaries_nue_lowe.txt",
    #     )
    # )

    # Apply average disapperance to the muon or electron samples
    # numu_MC *= Pmumu  # 8 bins
    # nue_bkg *= Pee  # 11 bins

    # NOTE: new method from Tao.
    if MC_nue_dis is not None:
        nue_bkg = MC_nue_dis
    else:
        nue_bkg = np.genfromtxt(
            open_text(
                f"MiniTools.include.MB_data_release_{year}.{mode}mode",
                f"miniboone_nue{bar}bgr_lowe.txt",
            )
        )

    if MC_numu_dis is not None:
        numu_MC = MC_numu_dis
    else:
        numu_MC = np.genfromtxt(
            open_text(
                f"MiniTools.include.MB_data_release_{year}.{mode}mode",
                f"miniboone_numu{bar}.txt",
            )
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


def chi2_MiniBooNE_2020_combined(NP_MC, NP_MC_BAR, NPevents=None, NPevents_BAR=None, year="2020"):
    """chi2_MiniBooNE_2020 Get MiniBOoNE chi2 from data release in 2020 for a given mode (FHC, RHC)

    Parameters
    ----------
    NP_MC : np.array
        Monte Carlo prediction for the FHC signal rate
    NP_MC : np.array
        Monte Carlo prediction for the RHC signal rate
    NPevents : np.float, optional
        Total number of signal events to renormalize the NP_MC prediction.
    NPevents_BAR : np.float, optional
        Total number of signal events to renormalize the NP_MC_BAR prediction.

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC) != 0 and NPevents is not None:
        NP_MC = (NP_MC / np.sum(NP_MC)) * NPevents
    if np.sum(NP_MC_BAR) != 0 and NPevents_BAR is not None:
        NP_MC_BAR = (NP_MC_BAR / np.sum(NP_MC_BAR)) * NPevents_BAR

    mode = "combined"

    ##########################################
    # Load neutrino data
    bar = ""
    nue_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_numu{bar}data.txt",
        )
    )

    nue_bkg = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_nue{bar}bgr_lowe.txt",
        )
    )
    numu_MC = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_numu{bar}.txt",
        )
    )

    ##########################################
    # Load antineutrino data
    bar = "bar"
    nue_data_bar = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data_bar = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_numu{bar}data.txt",
        )
    )

    nue_bkg_bar = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_nue{bar}bgr_lowe.txt",
        )
    )
    numu_MC_bar = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_numu{bar}.txt",
        )
    )

    ##########################################
    # Load covariance matrix
    fract_covariance = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_{year}.{mode}",
            f"miniboone_full_fractcovmatrix_combined_lowe.txt",
        )
    )

    NP_diag_matrix = np.diag(
        np.concatenate(
            [
                NP_MC,
                nue_bkg * 0.0,
                numu_MC * 0.0,
                NP_MC_BAR,
                nue_bkg_bar * 0.0,
                numu_MC_bar * 0.0,
            ]
        )
    )
    tot_diag_matrix = np.diag(
        np.concatenate([NP_MC, nue_bkg, numu_MC, NP_MC_BAR, nue_bkg_bar, numu_MC_bar])
    )

    rescaled_covariance = np.dot(
        tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix)
    )
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_MC)
    error_matrix = MassageCovarianceMatrix(rescaled_covariance, n_signal, n_numu)

    # compute residuals
    residuals = np.concatenate(
        [
            nue_data - (NP_MC + nue_bkg),
            (numu_data - numu_MC),
            nue_data_bar - (NP_MC_BAR + nue_bkg_bar),
            (numu_data_bar - numu_MC_bar),
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
