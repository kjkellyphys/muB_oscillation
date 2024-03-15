import numpy as np
import pickle
import copy

import MicroTools as micro
from MicroTools.sterile_tools import Sterile
from MicroTools.InclusiveTools.inclusive_osc_tools import (
    Decay_muB_OscChi2,
    DecayMuBNuMuDis,
    DecayMuBNuEDis,
)
import MiniTools as mini
import numba

RHE = False
UFMB = True
GBPC = micro.unfolder.MBtomuB(
    analysis="1eX_PC",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)
GBFC = micro.unfolder.MBtomuB(
    analysis="1eX",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)

# Load the MiniBooNE MC from data release
MB_Ereco_unfold_bins = micro.bin_edges_reco
MB_Ereco_official_bins = micro.bin_edges * 1e-3
MB_Ereco_official_bins_numu = micro.bin_edges_numu * 1e-3
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
def create_e_daughter(e_prod, n_replications=10):
    # e_prod: parent neutrino energy
    de = e_prod / n_replications
    e_daughter = np.linspace(de / 2, e_prod - de / 2, n_replications)
    return e_daughter


@numba.jit(nopython=True)
def create_Etrue_and_Weight_int(etrue, n_replications=10):
    # For every Etrue, create a list of possible daughter neutrino energy
    Etrue_daughter = np.empty((etrue.size, n_replications))
    for i in range(etrue.size):
        Etrue_daughter[i] = create_e_daughter(etrue[i], n_replications=n_replications)

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
        if (umu4s + ue4s <= 1.0) and (g**2 / 4 / np.pi) < 1.0:
            paramlist.append({"g": g, "m4": m4, "Ue4Sq": ue4s, "Um4Sq": umu4s})
    return np.array(paramlist)
    # return [{"g": g, "m4": m4, "Ue4Sq": Ue4Sq, "Um4Sq": Um4Sq} for g, m4, Ue4Sq, Um4Sq in paramlist_decay]


def create_grid_of_params_sin2theta(g, m4, sin2thetaSq, Um4Sq):
    paramlist_decay = np.array(np.meshgrid(g, m4, sin2thetaSq, Um4Sq)).T.reshape(-1, 4)
    paramlist = []
    for g, m4, s2ts, umu4s in paramlist_decay:
        ue4s = s2ts / 4 / umu4s
        if (umu4s + ue4s <= 1.0) and ((g**2 / 4 / np.pi) < 1.0):
            paramlist.append({"g": g, "m4": m4, "Ue4Sq": ue4s, "Um4Sq": umu4s})
    return np.array(paramlist)


def profile_in_plane(x, y, chi2):
    # Create a list of tuples for the unique pairs of Ue4SQR and Umu4SQR
    unique_pairs = np.array(list(set(zip(x, y))))

    # Find the minimum chi2 for each unique pair of Ue4SQR and Umu4SQR
    profiled_chi2 = np.array(
        [np.min(chi2[(x == pair[0]) & (y == pair[1])]) for pair in unique_pairs]
    )

    return unique_pairs[:, 0], unique_pairs[:, 1], profiled_chi2


def write_pickle(filename, data):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def pickle_read(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out


def MiniBooNEChi2_deGouvea(
    theta, oscillations=False, decay=True, decouple_decay=True, n_replications=10
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
        etrue=Etrue_nue, n_replications=n_replications
    )
    Etrue_nuebar_parent, Etrue_nuebar_daughter = create_Etrue_and_Weight_int(
        etrue=Etrue_nuebar, n_replications=n_replications
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

    # P_mumu_avg_deGouvea = sterile.PmmAvg_vec_deGouvea(
    # MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_micro
    # )
    # MC_numu_bkg_total_w_dis_deGouvea = mini.MC_numu_bkg_tot * P_mumu_avg_deGouvea

    # P_mumu_avg_deGouvea_bar = antisterile.PmmAvg_vec_deGouvea(
    # MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_micro
    # )
    # MC_numu_bkg_total_w_dis_deGouvea = mini.MC_numu_bkg_tot * P_mumu_avg_deGouvea

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
        etrue=Etrue_nue, n_replications=n_replications
    )
    Etrue_nuebar_parent, Etrue_nuebar_daughter = create_Etrue_and_Weight_int(
        etrue=Etrue_nuebar, n_replications=n_replications
    )

    # replicating entries of the MC data release -- baseline L and weight
    Ereco_nue_ext = replicate(Ereco_nue, n=n_replications)
    Length_nue_ext = replicate(Length_nue, n=n_replications)
    Weight_nue_ext = replicate(Weight_nue / n_replications, n=n_replications)

    Ereco_nuebar_ext = replicate(Ereco_nuebar, n=n_replications)
    Length_nuebar_ext = replicate(Length_nuebar, n=n_replications)
    Weight_nuebar_ext = replicate(Weight_nuebar / n_replications, n=n_replications)

    # Flavor transition probabilities
    Pme = sterile.Pme(Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext)
    Pmebar = antisterile.Pme(
        Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
    )

    Weight_nue_app = Weight_nue_ext * Pme
    Weight_nuebar_app = Weight_nuebar_ext * Pmebar

    if undo_numu_normalization:
        # flux is already normalized to data, so undo Pmumu from MC prediction
        # NOTE: evaluated at nu_e energies since that is what the flux is based on
        Pmm = sterile.Pmm(Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext)
        Pmmbar = antisterile.Pmm(
            Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
        )

        Weight_nue_app /= Pmm
        Weight_nuebar_app /= Pmmbar

    # Calculate the MiniBooNE chi2
    if not decay and oscillations:
        # NOTE: Using Ereco from MC for oscillation-only
        dic["MC_nue_app"] = fast_histogram(
            Ereco_nue_ext,
            weights=Weight_nue_app,
            bins=MB_Ereco_official_bins,
        )[0]
        dic["MC_nuebar_app"] = fast_histogram(
            Ereco_nuebar_ext,
            weights=Weight_nuebar_app,
            bins=MB_Ereco_official_bins,
        )[0]

    else:
        # Migrate nue signal from Etrue to Ereco with 11 bins
        dic["MC_nue_app"] = np.dot(
            fast_histogram(
                Etrue_nue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_app
            )[0],
            mini.apps.migration_matrix_official_bins_nue_11bins,
        )

        # NOTE: Need to update to nubar migration matrix!!!!
        dic["MC_nuebar_app"] = np.dot(
            fast_histogram(
                Etrue_nuebar_daughter, bins=e_prod_e_int_bins, weights=Weight_nuebar_app
            )[0],
            mini.apps.migration_matrix_official_bins_nuebar_11bins,
        )

    # For MicroBooNE unfolding -- different binning
    dic["MC_nue_app_for_unfolding"] = fast_histogram(
        Ereco_nue_ext, weights=Weight_nue_app, bins=MB_Ereco_official_bins
    )[0]

    # Average disappearance in each bin of MB MC data release
    if disappearance:
        Weight_nue_flux = mini.apps.reweight_MC_to_nue_flux(
            Etrue_nue_parent, Weight_nue_ext, mode="fhc"
        )
        Weight_nue_dis = Weight_nue_flux * sterile.Pee(
            Etrue_nue_parent, Etrue_nue_daughter, Length_nue_ext
        )

        Weight_nuebar_flux = mini.apps.reweight_MC_to_nue_flux(
            Etrue_nuebar_parent, Weight_nuebar_ext, mode="rhc"
        )
        Weight_nuebar_dis = Weight_nuebar_flux * antisterile.Pee(
            Etrue_nuebar_parent, Etrue_nuebar_daughter, Length_nuebar_ext
        )
        if (not decay) and oscillations:
            MC_nue_bkg_intrinsic = fast_histogram(
                Ereco_nue_ext,
                weights=Weight_nue_flux,
                bins=MB_Ereco_official_bins,
            )[0]
            MC_nue_bkg_intrinsic_osc = fast_histogram(
                Ereco_nue_ext,
                weights=Weight_nue_dis,
                bins=MB_Ereco_official_bins,
            )[0]

            MC_nuebar_bkg_intrinsic = fast_histogram(
                Ereco_nuebar_ext,
                weights=Weight_nuebar_flux,
                bins=MB_Ereco_official_bins,
            )[0]
            MC_nuebar_bkg_intrinsic_osc = fast_histogram(
                Ereco_nuebar_ext,
                weights=Weight_nuebar_dis,
                bins=MB_Ereco_official_bins,
            )[0]
        elif energy_degradation:
            # Apply energy degradation to nue intrinsic background, then migrate nue signal from Etrue to Ereco with 11 bins
            MC_nue_bkg_intrinsic = np.dot(
                fast_histogram(
                    Etrue_nue_parent, bins=e_prod_e_int_bins, weights=Weight_nue_flux
                )[0],
                mini.apps.migration_matrix_official_bins_nue_11bins,
            )
            MC_nue_bkg_intrinsic_osc = np.dot(
                sterile.EnergyDegradation(
                fast_histogram(
                    Etrue_nue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_dis
                )[0],e_prod_e_int_bins,"Pee"),
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
                sterile.EnergyDegradation(
                fast_histogram(
                    Etrue_nuebar_daughter,
                    bins=e_prod_e_int_bins,
                    weights=Weight_nuebar_dis,
                )[0],e_prod_e_int_bins,"Pee"),
                mini.apps.migration_matrix_official_bins_nuebar_11bins,
            )
        else:
            # Migrate nue signal from Etrue to Ereco with 11 bins
            MC_nue_bkg_intrinsic = np.dot(
                fast_histogram(
                    Etrue_nue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_flux
                )[0],
                mini.apps.migration_matrix_official_bins_nue_11bins,
            )
            MC_nue_bkg_intrinsic_osc = np.dot(
                fast_histogram(
                    Etrue_nue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_dis
                )[0],
                mini.apps.migration_matrix_official_bins_nue_11bins,
            )

            MC_nuebar_bkg_intrinsic = np.dot(
                fast_histogram(
                    Etrue_nuebar_daughter,
                    bins=e_prod_e_int_bins,
                    weights=Weight_nuebar_flux,
                )[0],
                mini.apps.migration_matrix_official_bins_nuebar_11bins,
            )
            MC_nuebar_bkg_intrinsic_osc = np.dot(
                fast_histogram(
                    Etrue_nuebar_daughter,
                    bins=e_prod_e_int_bins,
                    weights=Weight_nuebar_dis,
                )[0],
                mini.apps.migration_matrix_official_bins_nuebar_11bins,
            )

        # Final MC prediction for nu_e sample (w/ oscillated intrinsics)
        dic["MC_nue_bkg_total_w_dis"] = (
            mini.MC_nue_bkg_tot - MC_nue_bkg_intrinsic + MC_nue_bkg_intrinsic_osc
        )
        dic["MC_nuebar_bkg_total_w_dis"] = (
            mini.MC_nuebar_bkg_tot
            - MC_nuebar_bkg_intrinsic
            + MC_nuebar_bkg_intrinsic_osc
        )

        # NUMU DISAPPEARANCE
        if use_numu_MC:
            Etrue_numu_parent, Etrue_numu_daughter = create_Etrue_and_Weight_int(
                etrue=Etrue_numu, n_replications=n_replications
            )
            Ereco_numu_ext = replicate(Ereco_numu, n=n_replications)
            Length_numu_ext = replicate(Length_numu, n=n_replications)
            Weight_numu_ext = replicate(Weight_numu / n_replications, n=n_replications)

            Etrue_numubar_parent, Etrue_numubar_daughter = create_Etrue_and_Weight_int(
                etrue=Etrue_numubar, n_replications=n_replications
            )
            Ereco_numubar_ext = replicate(Ereco_numubar, n=n_replications)
            Length_numubar_ext = replicate(Length_numubar, n=n_replications)
            Weight_numubar_ext = replicate(
                Weight_numubar / n_replications, n=n_replications
            )

            # if undo_numu_normalization:
            #     # do not apply Pmumu in this case as the flux is already normalized
            #     Weight_numu_dis = Weight_numu_ext
            #     Weight_numubar_dis = Weight_numubar_ext
            # else:
            Weight_numu_dis = Weight_numu_ext * sterile.Pmm(
                Etrue_numu_parent, Etrue_numu_daughter, Length_numu_ext
            )
            Weight_numubar_dis = Weight_numubar_ext * antisterile.Pmm(
                Etrue_numubar_parent, Etrue_numubar_daughter, Length_numubar_ext
            )

            if not energy_degradation:
                dic["MC_numu_bkg_total_w_dis"] = fast_histogram(
                    Ereco_numu_ext,
                    weights=Weight_numu_dis,
                    bins=MB_Ereco_official_bins_numu,
                )[0]
                dic["MC_numubar_bkg_total_w_dis"] = fast_histogram(
                    Ereco_numubar_ext,
                    weights=Weight_numubar_dis,
                    bins=MB_Ereco_official_bins_numu,
                )[0]

            else:
                # now apply energy degradation to Etrue, then migrate to Ereco
                dic["MC_numu_bkg_total_w_dis"] = np.dot(
                    sterile.EnergyDegradation(
                    fast_histogram(
                        Etrue_numu_daughter,
                        bins=e_prod_e_int_bins_numu,
                        weights=Weight_numu_dis,
                    )[0],e_prod_e_int_bins_numu,"Pmm"),
                    mini.apps.migration_matrix_official_bins_numu,
                )
                dic["MC_numubar_bkg_total_w_dis"] = np.dot(
                    antisterile.EnergyDegradation(
                    fast_histogram(
                        Etrue_numubar_daughter,
                        bins=e_prod_e_int_bins_numu,
                        weights=Weight_numubar_dis,
                    )[0],e_prod_e_int_bins_numu,"Pmm"),
                    mini.apps.migration_matrix_official_bins_numubar,
                )
            
            # # Migrate nue signal from Etrue to Ereco with 11 bins
            # MC_numu_bkg_total_w_dis = np.dot(
            # fast_histogram(
            #     Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_numu_dis
            # )[0],
            # mini.apps.migration_matrix_official_bins_nue_11bins,
            # )

        else:
            # NOTE: Averaged
            # Final MC prediction for nu_mu sample (w/ oscillated numus)
            # if undo_numu_normalization:
            # do not apply Pmumu in this case as the flux is already normalized
            # MC_numu_bkg_total_w_dis = mini.MC_numu_bkg_tot
            # else:
            P_mumu_avg = sterile.PmmAvg_vec(
                MB_Ereco_official_bins_numu[:-1],
                MB_Ereco_official_bins_numu[1:],
                micro.L_mini,
            )
            dic["MC_numu_bkg_total_w_dis"] = mini.MC_numu_bkg_tot * P_mumu_avg

            P_mumu_avg_bar = antisterile.PmmAvg_vec(
                MB_Ereco_official_bins_numu[:-1],
                MB_Ereco_official_bins_numu[1:],
                micro.L_mini,
            )
            dic["MC_numubar_bkg_total_w_dis"] = mini.MC_numubar_bkg_tot * P_mumu_avg_bar

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
    )
    # \nu_e disappearance signal replacement
    NuEReps = DecayMuBNuEDis(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
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
