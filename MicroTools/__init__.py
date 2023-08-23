import numpy as np
from pathlib import Path
local_dir = Path(__file__).parent


MeVToGeV = 1.0e-3


################################################################## 
# main plots folder
path_plots = f"{local_dir}/../plots/"

################################################################## 
# MiniBooNE data
path_mb_data = f"{local_dir}/MB_data/"
path_mb_data_release = f"{path_mb_data}/data_release_2018/"
mb_data_osctables = f"{path_mb_data}/MB_osc_tables/"

# reco neutrino energy, true neutrino energy, neutrino beampipe, and event weight
mb_mc_data_release = np.genfromtxt(path_mb_data_release + "/miniboone_numunuefullosc_ntuple.txt")
bin_edges = np.genfromtxt(path_mb_data_release + "/miniboone_binboundaries_nue_lowe.txt")
bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2.
bin_width = np.diff(bin_edges)
mb_nue_analysis_data = np.genfromtxt(path_mb_data_release + "/miniboone_nuedata_lowe.txt")
mb_numu_analyis_data = np.genfromtxt(path_mb_data_release + "/miniboone_numudata.txt")
mb_nue_analysis_predicted_background = np.genfromtxt(path_mb_data_release + "/miniboone_nuebgr_lowe.txt")
mb_numu_analyis_prediction = np.genfromtxt(path_mb_data_release + "/miniboone_numu.txt")
fractional_covariance_matrix = np.genfromtxt(path_mb_data_release + "/miniboone_full_fractcovmatrix_nu_lowe.txt")

################################################################## 
# unfolding
path_unfolding_data = f"{local_dir}/muB_data/unfolding_data/"

################################################################## 
# MCMC
PATH_MCMC_CHAINS = f'{local_dir}/TemplateTools/mcmc_results/'

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
