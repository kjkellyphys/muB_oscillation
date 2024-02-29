import numpy as np
from pathlib import Path

local_dir = Path(__file__).parent

MeVToGeV = 1.0e-3

##################################################################
# main plots folder
path_plots = f"{local_dir}/../plots/"

##################################################################
# MiniBooNE data
path_mb_data = f"{local_dir}/../MiniTools/include/"
#path_mb_data_release = f"{path_mb_data}MB_data_release_2018/fhcmode/"
path_mb_data_release = f"{path_mb_data}MB_data_release_2020/fhcmode/"
path_mb_data_release_numu = f"{path_mb_data}MB_data_release_numudis_2009/fhcmode/"
# mb_data_osctables = f"{path_mb_data}MB_osc_tables/" # NOTE: This file is not used?

# reco neutrino energy, true neutrino energy, neutrino beampipe, and event weight
mb_mc_data_release = np.genfromtxt(
    f"{path_mb_data_release}miniboone_numunuefullosc_ntuple.txt"
)
mb_mc_data_release_numudis = np.genfromtxt(
    f"{path_mb_data_release_numu}miniboone_numu_ntuple.txt"
)
bin_edges = np.genfromtxt(f"{path_mb_data_release}miniboone_binboundaries_nue_lowe.txt")
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
bin_width = np.diff(bin_edges)
mb_nue_analysis_data = np.genfromtxt(
    path_mb_data_release + "/miniboone_nuedata_lowe.txt"
)
mb_numu_analyis_data = np.genfromtxt(path_mb_data_release + "/miniboone_numudata.txt")
mb_nue_analysis_predicted_background = np.genfromtxt(
    path_mb_data_release + "/miniboone_nuebgr_lowe.txt"
)
mb_numu_analyis_prediction = np.genfromtxt(path_mb_data_release + "/miniboone_numu.txt")
fractional_covariance_matrix = np.genfromtxt(
    path_mb_data_release + "/miniboone_full_fractcovmatrix_nu_lowe.txt"
)

bin_edges_reco = [
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
bin_centers_reco = bin_edges_reco[:-1] + np.diff(bin_edges_reco) / 2.0
bin_width_reco = np.diff(bin_edges_reco)

bin_edges_numu = np.array([0, 0.5, 0.7, 0.9, 1.1, 1.3, 1.50, 1.7, 1.9]) * 1e3  # MeV

##################################################################
# unfolding
path_unfolding_data = f"{local_dir}/muB_data/unfolding_data/"
path_unfolding_antinu_data = f"{path_unfolding_data}/antinu_aux/"

##################################################################
# Inclusive analysis
muB_inclusive_data_path = f"{local_dir}/muB_data/inclusive_data/"
muB_inclusive_datarelease_path = f"{muB_inclusive_data_path}/DataRelease/"

##################################################################
# Our oscillation results and other oscillation limits
path_osc_data = f"{local_dir}/osc_data/"
path_osc_app = f"{path_osc_data}/numu_to_nue/"
path_osc_numudis = f"{path_osc_data}/numu_dis/"
path_osc_nuedis = f"{path_osc_data}/nue_dis/"
