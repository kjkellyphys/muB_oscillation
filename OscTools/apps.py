import numpy as np
import warnings
import pickle
from scipy import interpolate
import copy

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import OscTools as osc


# 2021 MicroBooNE --> SBN
def rescale_micro_to_SBN(detector):
    if detector == "SBND":
        return (
            6.6
            / 6.369
            * osc.Mass_SBND
            / osc.Mass_micro
            * (osc.L_micro / osc.L_SBND) ** 2
        )
    elif detector == "ICARUS":
        return (
            6.6
            / 6.369
            * osc.Mass_ICARUS
            / osc.Mass_micro
            * (osc.L_micro / osc.L_ICARUS) ** 2
        )
    else:
        raise ValueError(f"Detector {detector} not recognized")


def rescale_mini_to_SBN(detector, mini_year):
    if detector == "SBND":
        return (
            6.6
            / osc.mini_POTs[mini_year]
            * osc.Mass_SBND
            / osc.Mass_mini
            * (osc.L_mini / osc.L_SBND) ** 2
        )
    elif detector == "ICARUS":
        return (
            6.6
            / osc.mini_POTs[mini_year]
            * osc.Mass_ICARUS
            / osc.Mass_mini
            * (osc.L_mini / osc.L_ICARUS) ** 2
        )
    else:
        raise ValueError(f"Detector {detector} not recognized")


def reweight_MC_to_nue_flux(Enu, weights, mode="fhc"):
    flux = np.genfromtxt(
        files("OscTools.include.fluxes")
        .joinpath(f"MiniBooNE_{mode.upper()}.dat")
        .open(),
    )
    ibar = 0 if mode == "fhc" else 3  # shift flux columns
    enu = flux[:, 0]  # MeV
    F_nue = interpolate.interp1d(
        enu, flux[:, 1 + ibar], bounds_error=False, fill_value=0
    )
    F_numu = interpolate.interp1d(
        enu, flux[:, 2 + ibar], bounds_error=False, fill_value=0
    )

    return weights * F_nue(Enu) / F_numu(Enu)


def create_reco_migration_matrix(
    ereco_bins, etrue_bins, ereco_events, etrue_events, weights
):
    # Set up a migration matrix that maps Etrue to Ereco
    h0_unnorm = np.histogram2d(
        etrue_events, ereco_events, bins=[etrue_bins, ereco_bins], weights=weights
    )[0]
    migration_matrix = copy.deepcopy(h0_unnorm)

    # Normalizing matrix elements w.r.t. to the interacting energy
    for j in range(len(etrue_bins) - 1):
        row_sum = np.sum(h0_unnorm[j])
        if row_sum < 0.0:
            print("negative row?")
        if row_sum == 0.0:
            continue
        migration_matrix[j] = h0_unnorm[j] / row_sum
    return migration_matrix


def write_pickle(filename, data):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def pickle_read(module, filename):
    f = files(module).joinpath(filename).open("rb")
    return pickle.load(f)


# Pre-computed migration matrices
migration_matrix_official_bins_numu = pickle_read(
    "OscTools.include.migration_matrices", "migration_matrix_official_bins_numu.pkl"
)

migration_matrix_official_bins_nue_11bins = pickle_read(
    "OscTools.include.migration_matrices",
    "migration_matrix_official_bins_nue_11bins.pkl",
)
migration_matrix_official_bins_nue_13bins = pickle_read(
    "OscTools.include.migration_matrices",
    "migration_matrix_official_bins_nue_13bins.pkl",
)

migration_matrix_official_bins_numubar = pickle_read(
    "OscTools.include.migration_matrices", "migration_matrix_official_bins_numubar.pkl"
)

migration_matrix_official_bins_nuebar_11bins = pickle_read(
    "OscTools.include.migration_matrices",
    "migration_matrix_official_bins_nuebar_11bins.pkl",
)

migration_matrix_official_bins_nuebar_13bins = pickle_read(
    "OscTools.include.migration_matrices",
    "migration_matrix_official_bins_nuebar_13bins.pkl",
)

################################################################################################
# Neutrino-Argon cross sections

tot_xsec_nue = np.load(
    files("OscTools.include").joinpath("GENIE_v3_muBtune2_tot_xsec_nue.npy").open("rb"),
    allow_pickle=True,
).item()
tot_xsec_numu = np.load(
    files("OscTools.include")
    .joinpath("GENIE_v3_muBtune2_tot_xsec_numu.npy")
    .open("rb"),
    allow_pickle=True,
).item()


################################################################################################
# Migration matrices


def build_migration_matrix(Etrue_mesh, Ereco_mesh, M_grid, Etrue_bins, Ereco_bins):

    M_small = np.histogram2d(
        Etrue_mesh.flatten(),
        Ereco_mesh.flatten(),
        bins=(Etrue_bins, Ereco_bins),
        weights=M_grid.flatten(),
    )[0]

    for i in range(len(Etrue_bins) - 1):
        if M_small[:, i].sum() > 0:
            M_small[:, i] = M_small[:, i] / M_small[:, i].sum()

    return M_small


################################################################################################
# MiniBooNE data releases


def get_MC_from_data_release_nue(mode="fhc", year="2020"):
    if year not in ["2009", "2012", "2018", "2020", "2022"]:
        raise ValueError(
            f"Only the data releases of 2009, 2012, 2018, 2020 and 2022 have a MC sample. You requested {year}."
        )
    if mode == "rhc":
        bar = "bar"
    else:
        bar = ""
    if year == "2022":
        Ereco, Etrue, Length, Weight = pickle_read(
            f"OscTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_numu{bar}nue{bar}fullosc_ntuple.pkl",
        ).T
    else:
        MiniBooNE_Signal = np.loadtxt(
            f"OscTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_numu{bar}nue{bar}fullosc_ntuple.txt",
        )
        Ereco = MiniBooNE_Signal[:, 0] / 1000  # GeV
        Etrue = MiniBooNE_Signal[:, 1] / 1000  # GeV
        Length = MiniBooNE_Signal[:, 2] / 100000  # Kilometers
        Weight = MiniBooNE_Signal[:, 3] / len(MiniBooNE_Signal[:, 3])
    return Ereco, Etrue, Length, Weight


def get_MC_from_data_release_numu(mode="fhc", year="2022"):

    if mode == "rhc":
        bar = "bar"
    else:
        bar = ""
    if year == "2022":
        Ereco, Etrue, Length, Weight = pickle_read(
            f"OscTools.include.MB_data_release_{year}.{mode}mode",
            f"miniboone_numu{bar}fullosc_ntuple.pkl",
        ).T
    elif year == "2009":
        warnings.warn("Loading 2009 MC -- this relies on a fudge factor!")
        MiniBooNE_Signal = np.loadtxt(
            files(
                f"MB_data_release_numudis_{year}.{mode}mode.miniboone_numu{bar}_ntuple.txt"
            ).open()
        )
        Ereco = MiniBooNE_Signal[:, 1]  # GeV
        Etrue = MiniBooNE_Signal[:, 2]  # GeV
        Length = MiniBooNE_Signal[:, 3]  # Kilometers
        RELATIVE_POTS_09_to_20_dis = 5.58 / 18.75
        FUDGE_FACTOR = (
            1 / 1.85
        )  # NOTE: Best we can do now until resolve the mismatch of numu samples
        TOT_RATE = {"fhc": 190_454, "rhc": 27_053}
        Weight = (
            MiniBooNE_Signal[:, 4]
            / np.sum(MiniBooNE_Signal[:, 4])
            * TOT_RATE[mode]
            / RELATIVE_POTS_09_to_20_dis
            * FUDGE_FACTOR
        )
    else:
        raise ValueError(
            f"Only the data releases of 2009 and 2022 have a numu MC sample. You requested {year}."
        )

    return Ereco, Etrue, Length, Weight
