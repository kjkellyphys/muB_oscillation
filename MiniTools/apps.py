import numpy as np
import warnings
import pickle
from scipy import interpolate
import copy
from importlib.resources import open_text

import MicroTools as micro

def reweight_MC_to_nue_flux(Enu, weights):
    flux = np.genfromtxt(
        open_text(
            f"MiniTools.include.fluxes",
            f"MiniBooNE_FHC.dat",
        )
    )
    enu = flux[:, 0]  # MeV
    F_nue = interpolate.interp1d(enu, flux[:, 1], bounds_error=False, fill_value=0)
    F_numu = interpolate.interp1d(enu, flux[:, 2], bounds_error=False, fill_value=0)
    return weights * F_nue(Enu) / F_numu(Enu)


def create_reco_migration_matrix(ereco_bins, etrue_bins, ereco_events, etrue_events, weights):
    # Set up a migration matrix that maps Etrue to Ereco with shape of (50,13)
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
def pickle_read(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out

# Pre-computed migration matrices 
migration_matrix_official_bins_numu = pickle_read('MiniTools/include/migration_matrices/migration_matrix_official_bins_numu.pkl')
migration_matrix_official_bins_nue_11bins = pickle_read('MiniTools/include/migration_matrices/migration_matrix_official_bins_nue_11bins.pkl')
migration_matrix_official_bins_nue_13bins = pickle_read('MiniTools/include/migration_matrices/migration_matrix_official_bins_nue_13bins.pkl')

def get_MC_from_data_release(mode='fhc', year='2020'):
    if year not in ['2009','2012', '2018','2020','2022']:
        raise ValueError(f"Only the data releases of 2009, 2012, 2018, 2020 and 2022 have a MC sample. You requested {year}.")
    if mode == 'rhc':
        bar = 'bar'
    else:
        bar = ''
    if year == '2022':
        Ereco, Etrue, Length, Weight = pickle_read(f'MiniTools/include/MB_data_release_{year}/{mode}mode/miniboone_numu{bar}nue{bar}fullosc_ntuple.pkl').T
    else:
        MiniBooNE_Signal = np.loadtxt(f'MiniTools/include/MB_data_release_{year}/{mode}mode/miniboone_numu{bar}nue{bar}fullosc_ntuple.txt')
        Ereco = MiniBooNE_Signal[:, 0] / 1000  # GeV
        Etrue = MiniBooNE_Signal[:, 1] / 1000  # GeV
        Length = MiniBooNE_Signal[:, 2] / 100000  # Kilometers
        Weight = MiniBooNE_Signal[:, 3] / len(MiniBooNE_Signal[:, 3])
    return Ereco, Etrue, Length, Weight

def get_MC_from_data_release_numu(mode='fhc', year='2022'):

    if year == '2022':
        Ereco, Etrue, Length, Weight = pickle_read(f'MiniTools/include/MB_data_release_{year}/{mode}mode/miniboone_numufullosc_ntuple.pkl').T 
    elif year == '2009':
        warnings.warn('Loading 2009 MC -- this relies on a fudge factor!')
        MiniBooNE_Signal = np.loadtxt(f"MB_data_release_numudis_{year}/{mode}mode/miniboone_numu_ntuple.txt")
        Ereco = MiniBooNE_Signal[:, 1]  # GeV
        Etrue = MiniBooNE_Signal[:, 2]  # GeV
        Length = MiniBooNE_Signal[:, 3]  # Kilometers
        RELATIVE_POTS_09_to_20_dis = 5.58 / 18.75
        FUDGE_FACTOR = 1/1.85 # NOTE: Best we can do now until resolve the mismatch of numu samples
        TOT_RATE = {'fhc': 190_454, 'rhc':  27_053}
        Weight = MiniBooNE_Signal[:, 4] / np.sum(MiniBooNE_Signal[:, 4]) * TOT_RATE[mode] / RELATIVE_POTS_09_to_20_dis * FUDGE_FACTOR
    else:
        raise ValueError(f"Only the data releases of 2009 and 2022 have a numu MC sample. You requested {year}.")

    return Ereco, Etrue, Length, Weight
