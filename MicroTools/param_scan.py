import numpy as np
import numba
import pickle
import copy
from scipy.stats import chi2
from .sterile_tools import Sterile
from .InclusiveTools.inclusive_osc_tools import (
    Decay_muB_OscChi2,
    DecayMuBNuMuDis,
    DecayMuBNuEDis,
)
import MiniTools as mini
from . import unfolder
from . import bin_edges, bin_edges_reco, bin_edges_numu, L_micro, L_mini

RHE = False
UFMB = True
GBPC = unfolder.MBtomuB(
    analysis="1eX_PC",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)
GBFC = unfolder.MBtomuB(
    analysis="1eX",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)

# Load the MiniBooNE MC from data release
MB_Ereco_unfold_bins = bin_edges_reco
MB_Ereco_official_bins = bin_edges * 1e-3
MB_Ereco_official_bins_numu = bin_edges_numu * 1e-3
e_prod_e_int_bins = np.linspace(0, 3, 51)  # GeV

# NOTE: this in principle can be a different set of bins...
e_prod_e_int_bins_numu = np.linspace(0, 3, 51)  # GeV

# NOTE: 2022 release has 2022 MC but still uses 2020 covariance matrices
Ereco_nue, Etrue_nue, Length_nue, Weight_nue = mini.apps.get_MC_from_data_release_nue(
    mode="fhc", year="2022"
)
Ereco_numu, Etrue_numu, Length_numu, Weight_numu = (
    mini.apps.get_MC_from_data_release_numu(mode="fhc", year="2022")
)

Ereco_nuebar, Etrue_nuebar, Length_nuebar, Weight_nuebar = (
    mini.apps.get_MC_from_data_release_nue(mode="rhc", year="2022")
)
Ereco_numubar, Etrue_numubar, Length_numubar, Weight_numubar = (
    mini.apps.get_MC_from_data_release_numu(mode="rhc", year="2022")
)


"""
    Create a distribution of interaction energy for every production energy
    based on the energy distribution of the daughter neutrinos (eqn 2.3&2.4 in 1911.01447)

    e_prod: parent neutrino energy
    n_replications: number of interaction energy bins per production energy

"""


@numba.jit(nopython=True)
def replicate(x, n):
    return np.repeat(x, n)


@numba.jit(nopython=True)
def create_e_daughter(e_prod, energy_degradation=True, n_replications=10):
    # e_prod: parent neutrino energy
    de = e_prod / n_replications
    if energy_degradation:
        e_daughter = np.linspace(de / 2, e_prod - de / 2, n_replications).astype(
            np.float64
        )
    else:
        e_daughter = np.repeat(e_prod, n_replications).astype(np.float64)
    return e_daughter


@numba.jit(nopython=True)
def create_Etrue_and_Weight_int(etrue, energy_degradation=True, n_replications=10):
    # For every Etrue, create a list of possible daughter neutrino energy
    Etrue_daughter = np.empty((etrue.size, n_replications))
    for i in range(etrue.size):
        Etrue_daughter[i] = create_e_daughter(
            etrue[i],
            n_replications=n_replications,
            energy_degradation=energy_degradation,
        )
    Etrue_extended = np.repeat(etrue, n_replications)
    return Etrue_extended, Etrue_daughter.flatten()


@numba.jit(nopython=True)
def numba_histogram(a, bin_edges, weights):
    """
    Custom weighted histogram function from Numba's page
    https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
    """
    hist = np.zeros((len(bin_edges) - 1,), dtype=np.float64)

    for i, x in enumerate(a.flat):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += weights[i]

    return hist, bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    n = bin_edges.shape[0] - 1

    # Find the bin index using binary search
    left = 0
    right = n

    while left < right:
        mid = (left + right) // 2
        if x >= bin_edges[mid] and x < bin_edges[mid + 1]:
            return mid
        elif x < bin_edges[mid]:
            right = mid
        else:
            left = mid + 1

    return None


def fast_histogram(data, bins, weights):
    return numba_histogram(data, bins, weights)


def create_grid_of_params(g, m4, Ue4Sq, Um4Sq):
    paramlist_decay = np.array(np.meshgrid(g, m4, Ue4Sq, Um4Sq)).T.reshape(-1, 4)
    paramlist = []
    for g, m4, ue4s, umu4s in paramlist_decay:
        if (umu4s + ue4s < 1.0) and (g**2 / 4 / np.pi) < 1.0:
            paramlist.append({"g": g, "m4": m4, "Ue4Sq": ue4s, "Um4Sq": umu4s})
    return np.array(paramlist)
    # return [{"g": g, "m4": m4, "Ue4Sq": Ue4Sq, "Um4Sq": Um4Sq} for g, m4, Ue4Sq, Um4Sq in paramlist_decay]


def create_grid_of_params_sin2theta(g, m4, sin2thetaSq, Um4Sq):
    paramlist_decay = np.array(np.meshgrid(g, m4, sin2thetaSq, Um4Sq)).T.reshape(-1, 4)
    paramlist = []
    for g, m4, s2ts, umu4s in paramlist_decay:
        ue4s = s2ts / 4 / umu4s
        if (umu4s + ue4s < 1.0) and ((g**2 / 4 / np.pi) < 1.0):
            paramlist.append({"g": g, "m4": m4, "Ue4Sq": ue4s, "Um4Sq": umu4s})
    return np.array(paramlist)


def get_subgrid(dic, var, var_range):
    if len(var_range) > 1:
        mask = (dic[var] < var_range[1]) & (dic[var] > var_range[0])
    else:
        mask = dic[var] == dic[var][np.argmin(np.abs(dic[var] - var_range))]
        print(f"Point found: {dic[var][np.argmin(np.abs(dic[var] - var_range))]}")
    for key in dic.keys():
        dic[key] = dic[key][mask]
    return dic


def profile_in_plane(
    x, y, chi2, profile_over_diff_chi2=None, x_max=np.inf, y_max=np.inf
):
    # Create a list of tuples for the unique pairs of Ue4SQR and Umu4SQR
    unique_pairs = np.array(list(set(zip(x, y))))

    # Find the minimum chi2 for each unique pair of Ue4SQR and Umu4SQR
    if profile_over_diff_chi2 is not None:
        profiled_chi2 = np.array(
            [
                (
                    chi2[(x == pair[0]) & (y == pair[1])][
                        np.argmin(
                            profile_over_diff_chi2[(x == pair[0]) & (y == pair[1])]
                        )
                    ]
                    if pair[0] < x_max and pair[1] < y_max
                    else np.nan
                )
                for pair in unique_pairs
            ]
        )
    else:
        profiled_chi2 = np.array(
            [
                (
                    np.min(chi2[(x == pair[0]) & (y == pair[1])])
                    if pair[0] < x_max and pair[1] < y_max
                    else np.nan
                )
                for pair in unique_pairs
            ]
        )

    return unique_pairs[:, 0], unique_pairs[:, 1], profiled_chi2


def profile_for_sin2theta(data_dic):
    dic_prof = {}
    # Profile each chi2
    dic_prof["sin2theta"], dic_prof["dm4SQR"], dic_prof["MiniApp_chi2"] = (
        profile_in_plane(
            data_dic["sin2theta"], data_dic["dm4SQR"], data_dic["MiniApp_chi2"]
        )
    )
    dic_prof["sin2theta"], dic_prof["dm4SQR"], dic_prof["MicroApp_chi2"] = (
        profile_in_plane(
            data_dic["sin2theta"], data_dic["dm4SQR"], data_dic["MicroApp_chi2"]
        )
    )
    dic_prof["sin2theta"], dic_prof["dm4SQR"], dic_prof["MicroApp_Asimov_chi2"] = (
        profile_in_plane(
            data_dic["sin2theta"], data_dic["dm4SQR"], data_dic["MicroApp_Asimov_chi2"]
        )
    )
    return dic_prof


def profile_x_y(
    data_dic, xlabel, ylabel, profile_over_diff_chi2=None, x_max=np.inf, y_max=np.inf
):
    dic_prof = {}
    kwargs = {
        "x_max": x_max,
        "y_max": y_max,
    }
    if profile_over_diff_chi2 is not None:
        if profile_over_diff_chi2 == "combined":
            kwargs["profile_over_diff_chi2"] = (
                data_dic["MicroApp_chi2"] + data_dic["MiniApp_chi2"]
            )
        else:
            kwargs["profile_over_diff_chi2"] = data_dic[profile_over_diff_chi2]
    else:
        kwargs["profile_over_diff_chi2"] = None

    # Profile each chi2
    dic_prof[xlabel], dic_prof[ylabel], dic_prof["MiniApp_chi2"] = profile_in_plane(
        data_dic[xlabel], data_dic[ylabel], data_dic["MiniApp_chi2"], **kwargs
    )
    dic_prof[xlabel], dic_prof[ylabel], dic_prof["MicroApp_chi2"] = profile_in_plane(
        data_dic[xlabel], data_dic[ylabel], data_dic["MicroApp_chi2"], **kwargs
    )
    dic_prof[xlabel], dic_prof[ylabel], dic_prof["MicroApp_Asimov_chi2"] = (
        profile_in_plane(
            data_dic[xlabel],
            data_dic[ylabel],
            data_dic["MicroApp_Asimov_chi2"],
            **kwargs,
        )
    )

    return dic_prof


def write_pickle(filename, data):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def pickle_read(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out


def load_scan_data(filename, wi=None, w_fixed=None, w2i=None, w2_fixed=None):
    data = pickle_read(filename)

    if wi is not None:
        cut_in_w = data[np.argmin(np.abs(data[:, wi] - w_fixed)), wi]
        data = data[(data[:, wi] == cut_in_w)]
        if w2i is not None:
            cut_in_w2 = data[np.argmin(np.abs(data[:, w2i] - w2_fixed)), w2i]
            data = data[(data[:, w2i] == cut_in_w2)]
    data_dic = {}
    data_dic["g"] = data[:, 0]
    data_dic["m4"] = data[:, 1]
    data_dic["Ue4SQR"] = data[:, 2]
    data_dic["Umu4SQR"] = data[:, 3]
    s2t = 4 * data_dic["Ue4SQR"] * data_dic["Umu4SQR"]
    data_dic["sin2theta"] = np.round(s2t, decimals=6)
    data_dic["dm4SQR"] = data[:, 1] ** 2

    data_dic["MiniApp_chi2"] = data[:, -3]
    data_dic["MicroApp_chi2"] = data[:, -2]
    data_dic["MicroApp_Asimov_chi2"] = data[:, -1]

    data_dic["MiniApp_deltachi2"] = data_dic["MiniApp_chi2"] - np.min(
        data_dic["MiniApp_chi2"]
    )
    data_dic["MicroApp_deltachi2"] = data_dic["MicroApp_chi2"] - np.min(
        data_dic["MicroApp_chi2"]
    )
    data_dic["MicroApp_Asimov_deltachi2"] = data_dic["MicroApp_Asimov_chi2"] - np.min(
        data_dic["MicroApp_Asimov_chi2"]
    )

    return data_dic


def get_best_fit_point(dic, Umu4SQRmax=1, Ue4SQRmax=1, dm4SQRmin=1e-2):
    mask = (
        (dic["Umu4SQR"] < Umu4SQRmax)
        & (dic["Ue4SQR"] < Ue4SQRmax)
        & (dic["dm4SQR"] > dm4SQRmin)
    )
    argmin = np.argmin(dic["MiniApp_chi2"][mask])
    dic_best_fit = {}
    for key in dic.keys():
        if key == "Ue4SQR":
            new_key = "Ue4Sq"
        elif key == "Umu4SQR":
            new_key = "Um4Sq"
        else:
            new_key = key
        dic_best_fit[new_key] = dic[key][mask][argmin]
    return dic_best_fit


def get_best_fit_point_MBpval(dic, ndof=20, UmuSQR4max=1, UeSQR4max=1):
    mask = (dic["Umu4SQR"] < UmuSQR4max) & (dic["Ue4SQR"] < UeSQR4max)
    chi2min = np.min(dic["MiniApp_chi2"][mask])
    return chi2.sf(chi2min, ndof)


def get_best_fit_point_MBchi2(dic, UmuSQR4max=1, UeSQR4max=1):
    mask = (dic["Umu4SQR"] < UmuSQR4max) & (dic["Ue4SQR"] < UeSQR4max)
    i_null = np.argmin(dic["Umu4SQR"][mask] + dic["Ue4SQR"][mask])
    i_bf = np.argmin(dic["MiniApp_chi2"][mask])
    return (
        dic["MiniApp_chi2"][mask][i_bf] - 69.059266748669
    )  # dic["MiniApp_chi2"][i_null]


def get_bf_point_definition(dic, ndof=20, Ue4SQRmax=1, Umu4SQRmax=1):
    mask = (dic["Ue4SQR"] < Ue4SQRmax) & (dic["Umu4SQR"] < Umu4SQRmax)
    i_bf = np.argmin(dic["MiniApp_chi2"][mask])
    i_null = np.argmin(dic["Umu4SQR"][mask] + dic["Ue4SQR"][mask])
    s = (
        (f'g = {dic["g"][mask][i_bf]:.3g}\n')
        + (f'dm4SQR = {dic["dm4SQR"][mask][i_bf]:.2g} eV^2\n')
        + (f'Ue4SQR = {dic["Ue4SQR"][mask][i_bf]:.2g}\n')
        + (f'Umu4SQR = {dic["Umu4SQR"][mask][i_bf]:.2g}\n')
        + (f'MB chi2 = {dic["MiniApp_chi2"][mask][i_bf]- 69.059:.2g}\n')
        + (f'MB pval = {chi2.sf(dic["MiniApp_chi2"][mask][i_bf], ndof)*100:.2g}%\n')
        + (
            f'Micro deltachi2 = {dic["MicroApp_chi2"][mask][i_bf] - dic["MicroApp_chi2"][mask][i_null]:.2g}\n'
        )
    )
    return s


def get_micro_bf_point_definition(
    dic, ndof=20, Ue4SQRmin=0, Umu4SQRmin=0, Ue4SQRmax=1, Umu4SQRmax=1
):
    mask = (
        (dic["Ue4SQR"] < Ue4SQRmax)
        & (dic["Umu4SQR"] < Umu4SQRmax)
        & (dic["Ue4SQR"] > Ue4SQRmin)
        & (dic["Umu4SQR"] > Umu4SQRmin)
    )
    i_bf = np.argmin(dic["MicroApp_chi2"][mask])
    i_null = np.argmin(dic["Umu4SQR"][mask] + dic["Ue4SQR"][mask])
    s = (
        (f'g = {dic["g"][mask][i_bf]:.3g}\n')
        + (f'dm4SQR = {dic["dm4SQR"][mask][i_bf]:.2g} eV^2\n')
        + (f'Ue4SQR = {dic["Ue4SQR"][mask][i_bf]:.2g}\n')
        + (f'Umu4SQR = {dic["Umu4SQR"][mask][i_bf]:.2g}\n')
        + (f'MB chi2 = {dic["MiniApp_chi2"][mask][i_bf]- 69.059:.2g}\n')
        + (f'MB pval = {chi2.sf(dic["MiniApp_chi2"][mask][i_bf], ndof)*100:.2g}%\n')
        + (
            f'Micro deltachi2 (bf - null) = {dic["MicroApp_chi2"][mask][i_bf] - dic["MicroApp_chi2"][mask][i_null]:.2g}\n'
        )
        + (f'Micro deltachi2 (null) = {dic["MicroApp_chi2"][mask][i_null]:.2g}\n')
        + (f'Micro deltachi2 (bf) = {dic["MicroApp_chi2"][mask][i_bf]:.2g}\n')
    )
    return s


def get_null_point_definition(dic, ndof=20):
    i_bf = np.argmin(dic["dm4SQR"] * dic["Ue4SQR"])
    s = (
        (f'g = {dic["g"][i_bf]:.3g}\n')
        + (f'dm4SQR = {dic["dm4SQR"][i_bf]:.2g} eV^2\n')
        + (f'Ue4SQR = {dic["Ue4SQR"][i_bf]:.2g}\n')
        + (f'Umu4SQR = {dic["Umu4SQR"][i_bf]:.2g}\n')
        + (f'MB chi2 = {dic["MiniApp_chi2"][i_bf]:.2g}\n')
        + (f'MB pval = {chi2.sf(dic["MiniApp_chi2"][i_bf], ndof)*100:.2g}%\n')
        + (f'Micro deltachi2 = {dic["MicroApp_deltachi2"][i_bf]:.2g}')
    )
    return s


def get_best_fit_point_DeltaMicroChi2(dic):
    return dic["MicroApp_deltachi2"][np.argmin(dic["MiniApp_chi2"])]


def MiniBooNEChi2_deGouvea(
    theta,
    oscillations=False,
    decay=True,
    disappearance=False,
    decouple_decay=True,
    n_replications=10,
    energy_degradation=True,
):
    """
    Returns the MicroBooNE chi2 for deGouvea's model
    """

    g = theta["g"]
    m4 = theta["m4"]
    Ue4Sq = theta["Ue4Sq"]
    Um4Sq = theta["Um4Sq"]

    sterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        CP=+1,
    )
    antisterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        CP=-1,
    )

    # Replicating events for multiple daughter neutrino energies
    Etrue_nue_parent, Etrue_nue_daughter = create_Etrue_and_Weight_int(
        etrue=Etrue_nue,
        n_replications=n_replications,
        energy_degradation=energy_degradation,
    )
    Etrue_nuebar_parent, Etrue_nuebar_daughter = create_Etrue_and_Weight_int(
        etrue=Etrue_nuebar,
        n_replications=n_replications,
        energy_degradation=energy_degradation,
    )

    # replicating entries of the MC data release -- baseline L and weight
    Length_nue_ext = replicate(Length_nue, n=n_replications)
    Weight_nue_ext = replicate(Weight_nue / n_replications, n=n_replications)

    Length_nuebar_ext = replicate(Length_nuebar, n=n_replications)
    Weight_nuebar_ext = replicate(Weight_nuebar / n_replications, n=n_replications)

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = sterile.Pme_deGouvea(Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext)
    Pmebar = antisterile.Pme_deGouvea(
        Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
    )

    Weight_nue_decay = Weight_nue_ext * Pme
    Weight_nuebar_decay = Weight_nuebar_ext * Pmebar

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        fast_histogram(
            Etrue_nue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_decay
        )[0],
        mini.apps.migration_matrix_official_bins_nue_11bins,
    )
    MBSig_for_MBfit_bar = np.dot(
        fast_histogram(
            Etrue_nuebar_daughter, bins=e_prod_e_int_bins, weights=Weight_nuebar_decay
        )[0],
        mini.apps.migration_matrix_official_bins_nuebar_11bins,
    )

    # Average disappearance in each bin of MB MC data release
    # P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=L_micro)
    # P_mumu_avg = (1 - Um4Sq) ** 2 + Um4Sq**2 * P_avg

    # MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

    ################################################
    # NOTE: Are you sure about L_micro here? Shouldnt it be L_mini?
    ################################################
    if disappearance:
        P_mumu_avg_deGouvea = sterile.PmmAvg_vec_deGouvea(
            MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_micro
        )
        MC_numu_bkg_total_w_dis_deGouvea = mini.MC_numu_bkg_tot * P_mumu_avg_deGouvea

        P_mumu_avg_deGouvea_bar = antisterile.PmmAvg_vec_deGouvea(
            MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_micro
        )
        MC_numubar_bkg_total_w_dis_deGouvea = (
            mini.MC_numu_bkg_tot * P_mumu_avg_deGouvea_bar
        )

        # Calculate MiniBooNE chi2
        MB_chi2 = mini.fit.chi2_MiniBooNE_combined(
            MC_nue_app=MBSig_for_MBfit,
            MC_nue_dis=None,
            MC_numu_dis=MC_numu_bkg_total_w_dis_deGouvea,
            MC_nuebar_app=MBSig_for_MBfit_bar,
            MC_nuebar_dis=None,
            MC_numubar_dis=MC_numubar_bkg_total_w_dis_deGouvea,
            year="2018",
        )

    else:

        # Calculate MiniBooNE chi2
        MB_chi2 = mini.fit.chi2_MiniBooNE_combined(
            MC_nue_app=MBSig_for_MBfit,
            MC_nue_dis=None,
            MC_numu_dis=None,
            MC_nuebar_app=MBSig_for_MBfit_bar,
            MC_nuebar_dis=None,
            MC_numubar_dis=None,
            year="2018",
        )

    return [g, m4, Ue4Sq, Um4Sq, MB_chi2]


def get_nue_rates(
    theta,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=False,
    energy_degradation=False,
    use_numu_MC=False,
    undo_numu_normalization=False,
    n_replications=10,
    include_antineutrinos=False,
    helicity="conserving",
):
    """
    Returns the neutrino event rates for the MicroBooNE analysis.

    Parameters
    ----------
    theta: Dict[str, np.ndarray]
        The dictionary containing the model parameters
    oscillations: bool, optional
        Whether to include oscillation-only analysis in the calculation.
        Defaults to True.
    decay: bool, optional
        Whether to include neutrino decay in the calculation. Defaults to False.
    decouple_decay: bool, optional
        Whether to decouple neutrino decay channels in the calculation.
        Defaults to False.
    disappearance: bool, optional
        Whether to include neutrino disappearance in the calculation.
        Defaults to False.
    energy_degradation: bool, optional
        Whether to include energy degradation in the calculation.
        Defaults to False.
    use_numu_MC: bool, optional
        Whether to use the NUMU MC data in the calculation. Defaults to False.
    undo_numu_normalization: bool, optional
        Whether to undo the normalization applied to the NUMU flux in the
        MC data. Defaults to False.
    n_replications: int, optional
        The number of replications to use when replicating the MC data.
        Defaults to 10.
    include_antineutrinos: bool, optional
        Whether to include antineutrino events in the analysis.
        Defaults to False.
    helicity: str, optional
        conserving: Ed/Ep
        flipping: 1 - Ed/Ep
    Returns
      -------
      Dict[str, np.ndarray]
          A dictionary containing the binned neutrino event rates.

    """
    # Will contains all the binned rates
    dic = {}

    # Our new physics class -- for deGouvea's model, we fix m4 = 1 eV, and identify g = gm4.
    sterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        CP=+1,
        helicity=helicity,
    )

    # Replicating events for multiple daughter neutrino energies
    Etrue_nue_parent, Etrue_nue_daughter = create_Etrue_and_Weight_int(
        etrue=Etrue_nue,
        n_replications=n_replications,
        energy_degradation=energy_degradation,
    )

    # replicating entries of the MC data release -- baseline L and weight
    Ereco_nue_ext = replicate(Ereco_nue, n=n_replications)
    Length_nue_ext = replicate(Length_nue, n=n_replications)
    Weight_nue_ext = replicate(Weight_nue / n_replications, n=n_replications)

    # decay and oscillation event weights.
    Weight_nue_osc_app = Weight_nue_ext * sterile.Pmeosc(
        Etrue_nue_parent, Length_nue_ext
    )
    Weight_nue_decay_app = Weight_nue_ext * sterile.Pmedecay(
        Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext
    )

    if include_antineutrinos:
        antisterile = Sterile(
            theta,
            oscillations=oscillations,
            decay=decay,
            decouple_decay=decouple_decay,
            CP=-1,
            helicity=helicity,
        )
        Etrue_nuebar_parent, Etrue_nuebar_daughter = create_Etrue_and_Weight_int(
            etrue=Etrue_nuebar,
            n_replications=n_replications,
            energy_degradation=energy_degradation,
        )

        Ereco_nuebar_ext = replicate(Ereco_nuebar, n=n_replications)
        Length_nuebar_ext = replicate(Length_nuebar, n=n_replications)
        Weight_nuebar_ext = replicate(Weight_nuebar / n_replications, n=n_replications)

        Weight_nuebar_osc_app = Weight_nuebar_ext * antisterile.Pmeosc(
            Etrue_nuebar_parent, Length_nuebar_ext
        )
        Weight_nuebar_decay_app = Weight_nuebar_ext * antisterile.Pmedecay(
            Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
        )

    if undo_numu_normalization:
        # undo Pmumu from MC prediction
        # NOTE: evaluated at nu_e energies since that is what the flux is based on

        Pmm = sterile.Pmm(Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext)
        Weight_nue_decay_app /= Pmm
        Weight_nue_osc_app /= Pmm

        if include_antineutrinos:
            Pmmbar = antisterile.Pmm(
                Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
            )
            Weight_nuebar_decay_app /= Pmmbar
            Weight_nuebar_osc_app /= Pmmbar

    # Calculate the MiniBooNE chi2
    if not decay and oscillations:
        # NOTE: Using Ereco from MC for oscillation-only
        dic["MC_nue_app"] = fast_histogram(
            Ereco_nue_ext,
            weights=Weight_nue_osc_app,
            bins=MB_Ereco_official_bins,
        )[0]
        if include_antineutrinos:
            dic["MC_nuebar_app"] = fast_histogram(
                Ereco_nuebar_ext,
                weights=Weight_nuebar_osc_app,
                bins=MB_Ereco_official_bins,
            )[0]

    else:
        # Migrate nue signal from Etrue to Ereco with 11 bins
        dic["MC_nue_app"] = np.dot(
            (
                fast_histogram(
                    Etrue_nue_daughter,
                    bins=e_prod_e_int_bins,
                    weights=Weight_nue_decay_app,
                )[0]
                + fast_histogram(
                    Etrue_nue_parent, bins=e_prod_e_int_bins, weights=Weight_nue_osc_app
                )[0]
            ),
            mini.apps.migration_matrix_official_bins_nue_11bins,
        )
        if include_antineutrinos:
            dic["MC_nuebar_app"] = np.dot(
                (
                    fast_histogram(
                        Etrue_nuebar_daughter,
                        bins=e_prod_e_int_bins,
                        weights=Weight_nuebar_decay_app,
                    )[0]
                    + fast_histogram(
                        Etrue_nuebar_parent,
                        bins=e_prod_e_int_bins,
                        weights=Weight_nuebar_osc_app,
                    )[0]
                ),
                mini.apps.migration_matrix_official_bins_nuebar_11bins,
            )

    # For MicroBooNE unfolding -- different binning
    dic["MC_nue_app_for_unfolding"] = np.dot(
        (
            fast_histogram(
                Etrue_nue_daughter, weights=Weight_nue_decay_app, bins=e_prod_e_int_bins
            )[0]
            + fast_histogram(
                Etrue_nue_parent, weights=Weight_nue_osc_app, bins=e_prod_e_int_bins
            )[0]
        ),
        mini.apps.migration_matrix_official_bins_nue_11bins,
    )

    # Average disappearance in each bin of MB MC data release
    if disappearance:
        Weight_nue_flux = mini.apps.reweight_MC_to_nue_flux(
            Etrue_nue_parent, Weight_nue_ext, mode="fhc"
        )
        Weight_nue_dis_osc = Weight_nue_flux * sterile.Peeosc(
            Etrue_nue_parent, Length_nue_ext
        )
        Weight_nue_dis_dec = Weight_nue_flux * sterile.Peedecay(
            Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext
        )
        if include_antineutrinos:
            Weight_nuebar_flux = mini.apps.reweight_MC_to_nue_flux(
                Etrue_nuebar_parent, Weight_nuebar_ext, mode="rhc"
            )
            Weight_nuebar_dis_osc = Weight_nuebar_flux * antisterile.Peeosc(
                Etrue_nuebar_parent, Length_nuebar_ext
            )
            Weight_nuebar_dis_dec = Weight_nuebar_flux * antisterile.Peedecay(
                Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
            )
        if (not decay) and oscillations:
            """
            If only oscillations, then we can simply histogram the MC events
            """
            MC_nue_bkg_intrinsic = fast_histogram(
                Ereco_nue_ext,
                weights=Weight_nue_flux,
                bins=MB_Ereco_official_bins,
            )[0]
            MC_nue_bkg_intrinsic_osc = fast_histogram(
                Ereco_nue_ext,
                weights=Weight_nue_dis_osc,
                bins=MB_Ereco_official_bins,
            )[0]
            if include_antineutrinos:

                MC_nuebar_bkg_intrinsic = fast_histogram(
                    Ereco_nuebar_ext,
                    weights=Weight_nuebar_flux,
                    bins=MB_Ereco_official_bins,
                )[0]
                MC_nuebar_bkg_intrinsic_osc = fast_histogram(
                    Ereco_nuebar_ext,
                    weights=Weight_nuebar_dis_osc,
                    bins=MB_Ereco_official_bins,
                )[0]
        else:
            """
            If decay is involved, then we take energy degradation into account
            """
            # Apply energy degradation to nue intrinsic background, then migrate nue signal from Etrue to Ereco with 11 bins
            MC_nue_bkg_intrinsic = np.dot(
                fast_histogram(
                    Etrue_nue_parent, bins=e_prod_e_int_bins, weights=Weight_nue_flux
                )[0],
                mini.apps.migration_matrix_official_bins_nue_11bins,
            )

            if include_antineutrinos:

                MC_nue_bkg_intrinsic_osc = np.dot(
                    (
                        fast_histogram(
                            Etrue_nue_parent,
                            bins=e_prod_e_int_bins,
                            weights=Weight_nue_dis_osc,
                        )[0]
                        + fast_histogram(
                            Etrue_nue_daughter,
                            bins=e_prod_e_int_bins,
                            weights=Weight_nue_dis_dec,
                        )[0]
                    ),
                    mini.apps.migration_matrix_official_bins_nue_11bins,
                )

                MC_nuebar_bkg_intrinsic = np.dot(
                    fast_histogram(
                        Etrue_nuebar_parent,
                        bins=e_prod_e_int_bins,
                        weights=Weight_nuebar_flux,
                    )[0],
                    mini.apps.migration_matrix_official_bins_nuebar_11bins,
                )

                MC_nuebar_bkg_intrinsic_osc = np.dot(
                    (
                        fast_histogram(
                            Etrue_nuebar_parent,
                            bins=e_prod_e_int_bins,
                            weights=Weight_nuebar_dis_osc,
                        )[0]
                        + fast_histogram(
                            Etrue_nuebar_daughter,
                            bins=e_prod_e_int_bins,
                            weights=Weight_nuebar_dis_dec,
                        )[0]
                    ),
                    mini.apps.migration_matrix_official_bins_nuebar_11bins,
                )

        # Final MC prediction for nu_e sample (w/ oscillated intrinsics)
        dic["MC_nue_bkg_total_w_dis"] = (
            mini.MC_nue_bkg_tot - MC_nue_bkg_intrinsic + MC_nue_bkg_intrinsic_osc
        )
        if include_antineutrinos:
            dic["MC_nuebar_bkg_total_w_dis"] = (
                mini.MC_nuebar_bkg_tot
                - MC_nuebar_bkg_intrinsic
                + MC_nuebar_bkg_intrinsic_osc
            )

        # NUMU DISAPPEARANCE
        if use_numu_MC:
            Etrue_numu_parent, Etrue_numu_daughter = create_Etrue_and_Weight_int(
                etrue=Etrue_numu,
                n_replications=n_replications,
                energy_degradation=energy_degradation,
            )
            # Ereco_numu_ext = replicate(Ereco_numu, n=n_replications)
            Length_numu_ext = replicate(Length_numu, n=n_replications)
            Weight_numu_ext = replicate(Weight_numu / n_replications, n=n_replications)

            Etrue_numubar_parent, Etrue_numubar_daughter = create_Etrue_and_Weight_int(
                etrue=Etrue_numubar,
                n_replications=n_replications,
                energy_degradation=energy_degradation,
            )
            # Ereco_numubar_ext = replicate(Ereco_numubar, n=n_replications)
            Length_numubar_ext = replicate(Length_numubar, n=n_replications)
            Weight_numubar_ext = replicate(
                Weight_numubar / n_replications, n=n_replications
            )

            if undo_numu_normalization:
                # do not apply Pmumu in this case as the flux is already normalized
                # NOTE: technically, there is also an energy dependent correction that we are ignoring here
                Weight_numu_dis_osc = Weight_numu_ext
                Weight_numu_dis_dec = Weight_numu_ext
                Weight_numubar_dis_osc = Weight_numubar_ext
                Weight_numubar_dis_dec = Weight_numubar_ext
            else:
                Weight_numu_dis_osc = Weight_numu_ext * sterile.Pmmosc(
                    Etrue_numu_parent, Length_numu_ext
                )
                Weight_numu_dis_dec = Weight_numu_ext * sterile.Pmmdecay(
                    Etrue_numu_parent, Etrue_numu_daughter, Length_numu_ext
                )
                Weight_numubar_dis_osc = Weight_numubar_ext * antisterile.Pmmosc(
                    Etrue_numubar_parent, Length_numubar_ext
                )
                Weight_numubar_dis_dec = Weight_numubar_ext * antisterile.Pmmdecay(
                    Etrue_numubar_parent, Etrue_numubar_daughter, Length_numubar_ext
                )

            dic["MC_numu_bkg_total_w_dis"] = np.dot(
                (
                    fast_histogram(
                        Etrue_numu_parent,
                        bins=e_prod_e_int_bins_numu,
                        weights=Weight_numu_dis_osc,
                    )[0]
                    + fast_histogram(
                        Etrue_numu_daughter,
                        bins=e_prod_e_int_bins,
                        weights=Weight_numu_dis_dec,
                    )[0]
                ),
                mini.apps.migration_matrix_official_bins_numu,
            )
            if include_antineutrinos:

                dic["MC_numubar_bkg_total_w_dis"] = np.dot(
                    (
                        fast_histogram(
                            Etrue_numubar_parent,
                            bins=e_prod_e_int_bins_numu,
                            weights=Weight_numubar_dis_osc,
                        )[0]
                        + fast_histogram(
                            Etrue_numubar_daughter,
                            bins=e_prod_e_int_bins,
                            weights=Weight_numubar_dis_dec,
                        )[0]
                    ),
                    mini.apps.migration_matrix_official_bins_numubar,
                )

        else:
            print("DEPRECATED: option `use_numu_MC = False` is deprecated")
            # NOTE: Averaged
            # Final MC prediction for nu_mu sample (w/ oscillated numus)
            # if undo_numu_normalization:
            # do not apply Pmumu in this case as the flux is already normalized
            # MC_numu_bkg_total_w_dis = mini.MC_numu_bkg_tot
            # else:
            # P_mumu_avg = sterile.PmmAvg_vec(
            # MB_Ereco_official_bins_numu[:-1],
            # MB_Ereco_official_bins_numu[1:],
            # L_mini,
            # )
            # dic["MC_numu_bkg_total_w_dis"] = mini.MC_numu_bkg_tot * P_mumu_avg

            # if include_antineutrinos:

            #     P_mumu_avg_bar = antisterile.PmmAvg_vec(
            #         MB_Ereco_official_bins_numu[:-1],
            #         MB_Ereco_official_bins_numu[1:],
            #         L_mini,
            #     )
            #     dic["MC_numubar_bkg_total_w_dis"] = (
            #         mini.MC_numubar_bkg_tot * P_mumu_avg_bar
            #     )

    return dic


# --------------------------------------------------------------------------------
def DecayReturnMicroBooNEChi2(
    theta,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=False,
    energy_degradation=False,
    use_numu_MC=False,
    undo_numu_normalization=False,
    n_replications=10,
    include_antineutrinos=False,
    helicity="conserving",
):

    rates_dic = get_nue_rates(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
        use_numu_MC=use_numu_MC,
        undo_numu_normalization=undo_numu_normalization,
        n_replications=n_replications,
        include_antineutrinos=include_antineutrinos,
        helicity=helicity,
    )

    if disappearance:
        if include_antineutrinos:
            # Calculate MiniBooNE chi2 -- nu + nubar
            MB_chi2 = mini.fit.chi2_MiniBooNE_combined(
                MC_nue_app=rates_dic["MC_nue_app"],
                MC_nuebar_app=rates_dic["MC_nuebar_app"],
                MC_nue_dis=rates_dic["MC_nue_bkg_total_w_dis"],
                MC_numu_dis=rates_dic["MC_numu_bkg_total_w_dis"],
                MC_nuebar_dis=rates_dic["MC_nuebar_bkg_total_w_dis"],
                MC_numubar_dis=rates_dic["MC_numubar_bkg_total_w_dis"],
                year="2020",
            )
        else:
            MB_chi2 = mini.fit.chi2_MiniBooNE(
                MC_nue_app=rates_dic["MC_nue_app"],
                MC_nue_dis=rates_dic["MC_nue_bkg_total_w_dis"],
                MC_numu_dis=rates_dic["MC_numu_bkg_total_w_dis"],
                year="2020",
            )

    else:
        if include_antineutrinos:
            MB_chi2 = mini.fit.chi2_MiniBooNE_combined(
                rates_dic["MC_nue_app"], rates_dic["MC_nuebar_app"], year="2020"
            )
        else:
            MB_chi2 = mini.fit.chi2_MiniBooNE(rates_dic["MC_nue_app"], year="2020")

    # NOTE: SKIPPING ENERGY DEGRATION FOR NOW
    # if energy_degradation:
    #     # MiniBooNE energy degradation
    #     # Questionable, MC file is meant for Pme channel. Not sure if it can be used for numu and nue disappearance.
    #     Ree_true = sterile.EnergyDegradation(
    #         fast_histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
    #         e_prod_e_int_bins,
    #         "Pee",
    #     )
    #     Rmm_true = sterile.EnergyDegradation(
    #         fast_histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
    #         e_prod_e_int_bins,
    #         "Pmm",
    #     )
    #     migration_matrix_pee = create_reco_migration_matrix(nue_bin_edges)
    #     migration_matrix_pmm = create_reco_migration_matrix(numu_bin_edges)
    #     Ree_reco = np.dot(Ree_true, migration_matrix_pee)
    #     Rmm_reco = np.dot(Rmm_true, migration_matrix_pmm)
    #     MB_chi2 = mini.fit.chi2_MiniBooNE_2020(
    #         MBSig_for_MBfit, Rmumu=Rmm_reco, Ree=Ree_reco
    #     )

    # Calculate the MicroBooNE chi2 by unfolding
    # MBSig_for_unfolding = np.dot(
    #     (fast_histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
    #     migration_matrix_unfolding_bins,
    # )

    ############################################################################################################
    # Now onto MicroBooNE  -- here only neutrinos are used (CP = +1)
    ############################################################################################################

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(rates_dic["MC_nue_app_for_unfolding"])
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # NOTE: copying is probably not needed, but who knows...
    MC_nue_app_for_unfolding2 = copy.deepcopy(rates_dic["MC_nue_app_for_unfolding"])
    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MC_nue_app_for_unfolding2)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
        helicity=helicity,
    )
    # \nu_e disappearance signal replacement
    NuEReps = DecayMuBNuEDis(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
        helicity=helicity,
    )
    # MicroBooNE
    MuB_chi2 = Decay_muB_OscChi2(
        theta,
        uBtemp,
        constrained=False,
        sigReps=[NuEReps[0], NuEReps[1], NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
    )

    MuB_chi2_Asimov = Decay_muB_OscChi2(
        theta,
        uBtemp,
        constrained=False,
        sigReps=[NuEReps[0], NuEReps[1], NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        Asimov=True,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
    )

    return [
        theta["g"],
        theta["m4"],
        theta["Ue4Sq"],
        theta["Um4Sq"],
        MB_chi2,
        MuB_chi2,
        MuB_chi2_Asimov,
    ]
