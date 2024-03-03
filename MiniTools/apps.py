import numpy as np
from scipy import integrate, interpolate
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



# Pre-computed migration matrices 
migration_matrix_official_bins_numu = np.load('MiniTools/include/migration_matrices/migration_matrix_official_bins_numu.npy', allow_pickle=True)
migration_matrix_official_bins_nue_11bins = np.load('MiniTools/include/migration_matrices/migration_matrix_official_bins_nue_11bins.npy', allow_pickle=True)
migration_matrix_official_bins_nue_13bins = np.load('MiniTools/include/migration_matrices/migration_matrix_official_bins_nue_13bins.npy', allow_pickle=True)

def get_MC_from_data_release_2009_numudis():
    # NOTE: 2009 numu disappearance 
    MC_sample_numu_dis = micro.mb_mc_data_release_numudis  
    Enumu_reco = MC_sample_numu_dis[:, 1]  # GeV
    Enumu_true = MC_sample_numu_dis[:, 2]  # GeV
    Length_numu = MC_sample_numu_dis[:, 3]  # Kilometers
    RELATIVE_POTS_09_to_20_dis = 5.58 / 18.75
    FUDGE_FACTOR = 1/1.85 # NOTE: Best we can do now until resolve the mismatch of numu samples
    Weight_numu = MC_sample_numu_dis[:, 4] / np.sum(MC_sample_numu_dis[:, 4]) * 1.90454e5 / RELATIVE_POTS_09_to_20_dis * FUDGE_FACTOR
    return Enumu_reco, Enumu_true, Length_numu, Weight_numu