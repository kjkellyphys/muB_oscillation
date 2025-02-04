import copy
import numpy as np
from scipy.linalg import inv
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import OscTools as osc
from OscTools import apps
from OscTools import const

# from OscTools import micro_tools as micro
# from OscTools.sterile_tools import Sterile

################################################################################################
# Fluxes
e, *f = np.genfromtxt(
    files("OscTools.include.fluxes").joinpath("MiniBooNE_FHC.dat").open(),
    skip_header=1,
    unpack=True,
)
flux_numu = interp1d(e, f[1], kind="linear", fill_value=0.0, bounds_error=False)
flux_numubar = interp1d(e, f[4], kind="linear", fill_value=0.0, bounds_error=False)
flux_nue = interp1d(e, f[0], kind="linear", fill_value=0.0, bounds_error=False)
flux_nuebar = interp1d(e, f[3], kind="linear", fill_value=0.0, bounds_error=False)

# Apply Gaussian smoothing to the ydata points
e, *f = np.genfromtxt(
    files("OscTools.include.fluxes").joinpath("SBND_FHC.dat").open(),
    skip_header=1,
    unpack=True,
)
flux_numu_SBND = interp1d(
    e,
    gaussian_filter1d(f[1], sigma=3, mode="nearest"),
    kind="linear",
    fill_value=0.0,
    bounds_error=False,
)
flux_numubar_SBND = interp1d(
    e,
    gaussian_filter1d(f[4], sigma=3, mode="nearest"),
    kind="linear",
    fill_value=0.0,
    bounds_error=False,
)
flux_nue_SBND = interp1d(
    e,
    gaussian_filter1d(f[0], sigma=3, mode="nearest"),
    kind="linear",
    fill_value=0.0,
    bounds_error=False,
)
flux_nuebar_SBND = interp1d(
    e,
    gaussian_filter1d(f[3], sigma=4, mode="nearest"),
    kind="linear",
    fill_value=0.0,
    bounds_error=False,
)


def neutrino_flux(E, detector, nuflavor):
    if nuflavor == "numu":
        if detector == "SBND":
            return flux_numu_SBND(E)
        elif detector == "ICARUS":
            return flux_numu(E) * (osc.L_mini / osc.L_ICARUS) ** 2
        else:
            raise ValueError(f"Detector {detector} not recognized")
    elif nuflavor == "nue":
        if detector == "SBND":
            return flux_nue_SBND(E)
        elif detector == "ICARUS":
            return flux_nue(E) * (osc.L_mini / osc.L_ICARUS) ** 2
        else:
            raise ValueError(f"Detector {detector} not recognized")
    elif nuflavor == "nuebar":
        if detector == "SBND":
            return flux_nuebar_SBND(E)
        elif detector == "ICARUS":
            return flux_nuebar(E) * (osc.L_mini / osc.L_ICARUS) ** 2
        else:
            raise ValueError(f"Detector {detector} not recognized")
    elif nuflavor == "numubar":
        if detector == "SBND":
            return flux_numubar_SBND(E)
        elif detector == "ICARUS":
            return flux_numubar(E) * (osc.L_mini / osc.L_ICARUS) ** 2
        else:
            raise ValueError(f"Detector {detector} not recognized")


################################################################################################
# Detector efficiencies

# Numu CC FC+PC
e, eff = np.genfromtxt(
    files("OscTools.include.muB_data.inclusive_data")
    .joinpath("Efficiency_numuCC_PCplusFC_paper.dat")
    .open(),
    skip_header=1,
    unpack=True,
)
eff_numu = interp1d(e, eff, kind="linear", fill_value=0.0, bounds_error=False)

# Nue CC FC+PC
e, eff = np.genfromtxt(
    files("OscTools.include.muB_data.inclusive_data")
    .joinpath("Efficiency_nueCC_PCplusFC_paper.dat")
    .open(),
    skip_header=1,
    unpack=True,
)
eff_nue = interp1d(e, eff, kind="linear", fill_value=0.0, bounds_error=False)

################################################################################################
# Migration matrices

Etrue_numuCC_FC, Ereco_numuCC_FC, M_numuCC_FC = np.load(
    "OscTools/include/muB_data/inclusive_data/refined_numuCC_FC_energy_res.npy"
)
Etrue_numuCC_PC, Ereco_numuCC_PC, M_numuCC_PC = np.load(
    "OscTools/include/muB_data/inclusive_data/refined_numuCC_PC_energy_res.npy"
)
Etrue_nueCC_FC, Ereco_nueCC_FC, M_nueCC_FC = np.load(
    "OscTools/include/muB_data/inclusive_data/refined_nueCC_FC_energy_res.npy"
)
Etrue_nueCC_PC, Ereco_nueCC_PC, M_nueCC_PC = np.load(
    "OscTools/include/muB_data/inclusive_data/refined_nueCC_PC_energy_res.npy"
)


################################################################################################
# Event rate estimator
# POT


class SBN:
    def __init__(
        self,
        exposure=6.6e20,
        Emin=0.001,
        Emax=3,
        nbins=26,
        flux_uncertainty=0.10,
        xsec_uncertainty=0.10,
        eff_uncertainty=0.03,
    ):

        self.exposure = exposure
        self.nbins = nbins
        self.Emin = Emin
        self.Emax = Emax

        self.flux_uncertainty = flux_uncertainty
        self.xsec_uncertainty = xsec_uncertainty
        self.eff_uncertainty = eff_uncertainty

        # True and reco neutrino energy bins
        # NOTE: assuming numu and nue have the same binning for simplicity
        self.E = np.linspace(Emin, Emax, 200)
        self.Etrue_bins = np.linspace(self.Emin, self.Emax, self.nbins + 1)
        self.Etrue_bin_center = self.Etrue_bins[:-1] + np.diff(self.Etrue_bins)[0] / 2
        self.Ereco_bins = np.linspace(self.Emin, self.Emax, self.nbins + 1)
        self.Ereco_bin_center = self.Ereco_bins[:-1] + np.diff(self.Ereco_bins)[0] / 2

        self.n_targets_SBND = osc.Mass_SBND / const.m_proton_in_t  # Nucleons (n or p+)
        self.n_targets_ICARUS = (
            osc.Mass_ICARUS / const.m_proton_in_t
        )  # Nucleons (n or p+)

        # Calculate the total number of events
        self.Ntotal_numu_SBND = (
            self.unosc_numu_rate_SBND(self.E) * np.diff(self.E)[0]
        ).sum()
        self.Ntotal_numu_ICARUS = (
            self.unosc_numu_rate_ICARUS(self.E) * np.diff(self.E)[0]
        ).sum()
        self.Ntotal_nue_SBND = (
            self.unosc_nue_rate_SBND(self.E) * np.diff(self.E)[0]
        ).sum()
        self.Ntotal_nue_ICARUS = (
            self.unosc_nue_rate_ICARUS(self.E) * np.diff(self.E)[0]
        ).sum()

        _ = self.build_migration_matrix()

        self.unosc_numu_SBND = self.reco_unosc_numu_rate_SBND()
        self.unosc_numu_ICARUS = self.reco_unosc_numu_rate_ICARUS()
        self.unosc_nue_SBND = self.reco_unosc_nue_rate_SBND()
        self.unosc_nue_ICARUS = self.reco_unosc_nue_rate_ICARUS()
        self.unosc_rate_vector = np.concatenate(
            [
                self.unosc_nue_SBND,
                self.unosc_nue_ICARUS,
                self.unosc_numu_SBND,
                self.unosc_numu_ICARUS,
            ]
        )

    def flux(self, E, nuflavor):
        return neutrino_flux(E, self.detector, nuflavor)

    def efficiency(self, E, nuflavor):
        if nuflavor == "numu":
            return eff_numu(E)
        elif nuflavor == "nue":
            return eff_nue(E)
        else:
            raise ValueError(f"Flavor {nuflavor} not recognized")

    # True unoscillated rates
    def unosc_numu_rate_SBND(self, E):
        return (
            neutrino_flux(E, "SBND", "numu")
            * self.n_targets_SBND
            * self.exposure
            * apps.tot_xsec_nue(E)
            * eff_numu(self.E)
        )

    def unosc_numu_rate_ICARUS(self, E):
        return (
            neutrino_flux(E, "ICARUS", "numu")
            * self.n_targets_ICARUS
            * self.exposure
            * apps.tot_xsec_nue(E)
            * eff_numu(self.E)
        )

    def unosc_nue_rate_SBND(self, E):
        return (
            neutrino_flux(E, "SBND", "nue")
            * self.n_targets_SBND
            * self.exposure
            * apps.tot_xsec_numu(E)
            * eff_nue(self.E)
        )

    def unosc_nue_rate_ICARUS(self, E):
        return (
            neutrino_flux(E, "ICARUS", "nue")
            * self.n_targets_ICARUS
            * self.exposure
            * apps.tot_xsec_numu(E)
            * eff_nue(self.E)
        )

    # True oscillated rates
    def osc_numu_rate_SBND(self, sterile, binned=True):
        diff_rate = sterile.Pmmosc(self.E, osc.L_SBND) * self.unosc_numu_rate_SBND(
            self.E
        ) + sterile.Pmeosc(self.E, osc.L_SBND) * self.unosc_nue_rate_SBND(self.E)

        if binned:
            h, be = np.histogram(
                self.E, weights=diff_rate * np.diff(self.E)[0], bins=self.Etrue_bins
            )
            return h / np.diff(be)
        else:
            return diff_rate

    def osc_numu_rate_ICARUS(self, sterile, binned=True):
        diff_rate = sterile.Pmmosc(self.E, osc.L_ICARUS) * self.unosc_numu_rate_ICARUS(
            self.E
        ) + sterile.Pmeosc(self.E, osc.L_ICARUS) * self.unosc_nue_rate_ICARUS(self.E)

        if binned:
            h, be = np.histogram(
                self.E, weights=diff_rate * np.diff(self.E)[0], bins=self.Etrue_bins
            )
            return h / np.diff(be)
        else:
            return diff_rate

    def osc_nue_rate_SBND(self, sterile, binned=True):
        diff_rate = sterile.Pmeosc(self.E, osc.L_SBND) * self.unosc_numu_rate_SBND(
            self.E
        ) + sterile.Peeosc(self.E, osc.L_SBND) * self.unosc_nue_rate_SBND(self.E)

        if binned:
            h, be = np.histogram(
                self.E, weights=diff_rate * np.diff(self.E)[0], bins=self.Etrue_bins
            )
            return h / np.diff(be)
        else:
            return diff_rate

    def osc_nue_rate_ICARUS(self, sterile, binned=True):
        diff_rate = sterile.Pmeosc(self.E, osc.L_ICARUS) * self.unosc_numu_rate_ICARUS(
            self.E
        ) + sterile.Peeosc(self.E, osc.L_ICARUS) * self.unosc_nue_rate_ICARUS(self.E)

        if binned:
            h, be = np.histogram(
                self.E, weights=diff_rate * np.diff(self.E)[0], bins=self.Etrue_bins
            )
            return h / np.diff(be)
        else:
            return diff_rate

    # Reco UNoscillated rates
    def reco_unosc_numu_rate_SBND(self):
        truth_rate = self.unosc_numu_rate_SBND(self.E)
        h, be = np.histogram(
            self.E, weights=truth_rate * np.diff(self.E)[0], bins=self.Etrue_bins
        )
        return self.M_numuCC.dot(h / np.diff(be))

    def reco_unosc_numu_rate_ICARUS(self):
        truth_rate = self.unosc_numu_rate_ICARUS(self.E)
        h, be = np.histogram(
            self.E, weights=truth_rate * np.diff(self.E)[0], bins=self.Etrue_bins
        )
        return self.M_numuCC.dot(h / np.diff(be))

    def reco_unosc_nue_rate_SBND(self):
        truth_rate = self.unosc_nue_rate_SBND(self.E)
        h, be = np.histogram(
            self.E, weights=truth_rate * np.diff(self.E)[0], bins=self.Etrue_bins
        )
        return self.M_nueCC.dot(h / np.diff(be))

    def reco_unosc_nue_rate_ICARUS(self):
        truth_rate = self.unosc_nue_rate_ICARUS(self.E)
        h, be = np.histogram(
            self.E, weights=truth_rate * np.diff(self.E)[0], bins=self.Etrue_bins
        )
        return self.M_nueCC.dot(h / np.diff(be))

    # Reco oscillated rates
    def reco_osc_numu_rate_SBND(self, sterile):
        truth_rate = self.osc_numu_rate_SBND(sterile, binned=True)
        return self.M_numuCC.dot(truth_rate)

    def reco_osc_numu_rate_ICARUS(self, sterile):
        truth_rate = self.osc_numu_rate_ICARUS(sterile, binned=True)
        return self.M_numuCC.dot(truth_rate)

    def reco_osc_nue_rate_SBND(self, sterile):
        truth_rate = self.osc_nue_rate_SBND(sterile, binned=True)
        return self.M_nueCC.dot(truth_rate)

    def reco_osc_nue_rate_ICARUS(self, sterile):
        truth_rate = self.osc_nue_rate_ICARUS(sterile, binned=True)
        return self.M_nueCC.dot(truth_rate)

    def build_migration_matrix(self):

        # NOTE: For now, I am ignoring the PC samples -- later we can add them fractionally wrt event rates
        self.M_numuCC = apps.build_migration_matrix(
            Etrue_numuCC_FC,
            Ereco_numuCC_FC,
            M_numuCC_FC,
            self.Etrue_bins,
            self.Ereco_bins,
        )
        self.M_nueCC = apps.build_migration_matrix(
            Etrue_nueCC_FC, Ereco_nueCC_FC, M_nueCC_FC, self.Etrue_bins, self.Ereco_bins
        )
        # self.M_numuCC_PC_small = apps.build_migration_matrix(
        #     Etrue_numuCC_PC, Ereco_numuCC_PC, M_numuCC_PC, Etrue_bins, Ereco_bins
        # )
        # self.M_nueCC_PC_small = apps.build_migration_matrix(
        #     Etrue_nueCC_PC, Ereco_nueCC_PC, M_nueCC_PC, Etrue_bins, Ereco_bins
        # )

    def build_covariance_matrix(self):
        # Define the systematic uncertainties as energy-dependent (one value per energy bin)
        delta_eta_flux = (
            np.ones(self.nbins) * self.flux_uncertainty
        )  # 10% flux uncertainty per bin
        delta_eta_xsec = (
            np.ones(self.nbins) * self.xsec_uncertainty
        )  # 10% cross-section uncertainty per bin
        delta_eta_eff = (
            np.ones(self.nbins) * self.eff_uncertainty
        )  # 3% detector efficiency uncertainty per bin

        # Initialize the covariance matrix
        num_rates = (
            4 * self.nbins
        )  # 4 rates: nu_e SBND, nu_e ICARUS, nu_mu SBND, nu_mu ICARUS
        self.frac_cov_matrix = np.zeros((num_rates, num_rates))

        # Function to add systematic uncertainty contributions to the covariance matrix
        def add_systematic_uncertainty(frac_cov_matrix, delta_eta, correlated_indices):
            for i in correlated_indices:
                for j in correlated_indices:
                    # Use the same delta_eta for the same energy bin (uncorrelated between bins)
                    bin_i = i % self.nbins  # Energy bin index for event i
                    bin_j = j % self.nbins  # Energy bin index for event j
                    if bin_i == bin_j:  # Only correlate within the same energy bin
                        frac_cov_matrix[i, j] += (delta_eta[bin_i]) * (delta_eta[bin_j])

        # Add flux uncertainty (correlated between SBND and ICARUS for the same energy bin, but uncorrelated between nu_e and nu_mu)
        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_flux, range(2 * self.nbins)
        )  # nu_e SBND + nu_e ICARUS

        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_flux, range(2 * self.nbins, 4 * self.nbins)
        )  # nu_mu SBND + nu_mu ICARUS

        # Add cross-section uncertainty (fully correlated between SBND, ICARUS, nu_e, and nu_mu for the same energy bin)
        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_xsec, range(num_rates)
        )

        # Add detector efficiency uncertainty (uncorrelated between detectors for the same energy bin)
        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_eff, range(self.nbins)
        )  # nu_e SBND

        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_eff, range(self.nbins, 2 * self.nbins)
        )  # nu_e ICARUS

        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_eff, range(2 * self.nbins, 3 * self.nbins)
        )  # nu_mu SBND

        add_systematic_uncertainty(
            self.frac_cov_matrix, delta_eta_eff, range(3 * self.nbins, 4 * self.nbins)
        )  # nu_mu ICARUS

        return self.frac_cov_matrix

    def calculate_cov(self, rate_vector):
        self.frac_cov_matrix = self.build_covariance_matrix()
        self.cov_matrix = np.outer(rate_vector, rate_vector) * self.frac_cov_matrix
        # Stat uncertainty
        self.cov_matrix += np.sqrt(rate_vector)
        return self.cov_matrix

    def calculate_inv_cov(self):
        cov = self.calculate_cov(self.unosc_rate_vector)
        self.inv_cov = inv(cov)
        return self.inv_cov

    def calculate_Asimov_chi2_from_rate_vector(self, test_rate_vector, inv_cov):
        # Calculate the Asimov chi-squared

        chi2 = np.dot(
            np.dot((self.unosc_rate_vector - test_rate_vector), inv_cov),
            (self.unosc_rate_vector - test_rate_vector),
        )
        return chi2

    def Asimov_chi2(self, sterile):
        test_rate_vector = np.concatenate(
            [
                self.reco_osc_nue_rate_SBND(sterile),
                self.reco_osc_nue_rate_ICARUS(sterile),
                self.reco_osc_numu_rate_SBND(sterile),
                self.reco_osc_numu_rate_ICARUS(sterile),
            ]
        )

        return self.calculate_Asimov_chi2_from_rate_vector(
            test_rate_vector, self.calculate_inv_cov()
        )


############################################################################################
# OLD FUNCTIONS THAT TRIED TO USE MICRO'S ANALYSIS CHAIN:
###########################################################################################

# ###########
# # Numu data (SBND and ICARUS)
# # Same rate as MicroBooNE but rescaled by POT and target mass
# SBND_NuMuCC_TrueEDist = micro.NuMuCC_TrueEDist_FC * apps.rescale_micro_to_SBN("SBND")
# ICARUS_NuMuCC_TrueEDist = micro.NuMuCC_TrueEDist_FC * apps.rescale_micro_to_SBN(
#     "ICARUS"
# )

# NuMuCC_MigMat_SBND = micro.NuMuCC_MigMat_FC
# NuMuCC_MigMat_ICARUS = micro.NuMuCC_MigMat_FC

# NuMuCC_Eff_SBND = micro.NuMuCC_Eff_FC
# NuMuCC_Eff_ICARUS = micro.NuMuCC_Eff_FC

# SBND_BinEdges_NuMu = [0.0 + 0.05 * j for j in range(61)]
# ICARUS_BinEdges_NuMu = [0.0 + 0.05 * j for j in range(61)]


# # MCT is MiniBooNE truth level distribution from 2018 provided by MicroBooNE
# MCT = np.load(
#     files(f"{micro.muB_inclusive_data_path}").joinpath("MuB_NuE_True.npy").open("rb")
# )
# Mini_True_BinEdges_used_by_MuB = [
#     0.200,
#     0.250,
#     0.300,
#     0.350,
#     0.400,
#     0.450,
#     0.500,
#     0.600,
#     0.800,
#     1.000,
#     1.500,
#     2.000,
#     2.500,
#     3.000,
# ]

# SBND_NuE = unfolder.MBtoLAr(
#     analysis="SBND",
#     remove_high_energy=False,
#     unfold=False,
#     effNoUnfold=False,
#     which_template="2018",
# )
# ICARUS_NuE = unfolder.MBtoLAr(
#     analysis="ICARUS",
#     remove_high_energy=False,
#     unfold=False,
#     effNoUnfold=False,
#     which_template="2018",
# )


# ###############
# def DecaySBNNuEDis(
#     theta,
#     oscillations=True,
#     decay=False,
#     decouple_decay=False,
#     disappearance=True,
#     energy_degradation=True,
#     helicity="conserving",
# ):
#     """Function for reweighting SBND/ICARUS nu_e spectra in terms of true energy instead of reconstructed energy"""

#     if decay:
#         raise ValueError("Decay is not implemented for SBND/ICARUS yet!")

#     # Load the Sterile class from param_scan
#     sterile = Sterile(
#         theta,
#         oscillations=oscillations,
#         decay=False,
#         decouple_decay=False,
#         helicity=helicity,
#     )

#     Pee_SBND = []
#     Pee_ICARUS = []
#     # MCT is MiniBooNE truth level distribution from 2018. That's why it needs to be rescaled when unfolding
#     MB_true_nue_rate = MCT
#     for k in range(len(MB_true_nue_rate)):
#         Pee_SBND.append(MB_true_nue_rate[k])
#         Pee_ICARUS.append(MB_true_nue_rate[k])
#     if disappearance:
#         # reset PeeRW
#         Pee_SBND = []
#         Pee_ICARUS = []
#         for k in range(len(MB_true_nue_rate)):
#             Pee_SBND.append(
#                 MB_true_nue_rate[k]
#                 * sterile.PeeoscAvg(
#                     Mini_True_BinEdges_used_by_MuB[k],
#                     Mini_True_BinEdges_used_by_MuB[k + 1],
#                     osc.L_SBND,
#                 )
#             )
#             Pee_ICARUS.append(
#                 MB_true_nue_rate[k]
#                 * sterile.PeeoscAvg(
#                     Mini_True_BinEdges_used_by_MuB[k],
#                     Mini_True_BinEdges_used_by_MuB[k + 1],
#                     osc.L_ICARUS,
#                 )
#             )

#     PeeSBND_2 = copy.deepcopy(Pee_SBND)
#     PeeICARUS_2 = copy.deepcopy(Pee_ICARUS)
#     SBND_NuE_rates = SBND_NuE.unfold(PeeSBND_2)
#     SBND_NuE_rates = np.insert(SBND_NuE_rates, 0, [0.0])
#     SBND_NuE_rates = np.append(SBND_NuE_rates, 0.0)

#     ICARUS_NuE_rates = ICARUS_NuE.unfold(PeeICARUS_2)
#     ICARUS_NuE_rates = np.insert(ICARUS_NuE_rates, 0, [0.0])
#     ICARUS_NuE_rates = np.append(ICARUS_NuE_rates, 0.0)

#     return [SBND_NuE_rates, ICARUS_NuE_rates]


# def DecaySBNNuMuDis(
#     theta,
#     oscillations=True,
#     decay=False,
#     decouple_decay=False,
#     disappearance=True,
#     energy_degradation=True,
#     helicity="conserving",
# ):
#     """Function for reweighting MicroBooNE nu_mu spectra in terms of true energy instead of reconstructed energy"""

#     # Load the Sterile class from param_scan
#     sterile = Sterile(
#         theta,
#         oscillations=oscillations,
#         decay=decay,
#         decouple_decay=decouple_decay,
#         helicity=helicity,
#     )
#     PmmRW_SBND = []
#     PmmRW_ICARUS = []
#     for k in range(len(SBND_NuMuCC_TrueEDist)):
#         PmmRW_SBND.append(SBND_NuMuCC_TrueEDist[k])
#         PmmRW_ICARUS.append(ICARUS_NuMuCC_TrueEDist[k])
#     if disappearance:
#         # reset PmmRW_SBND and PmmRW_ICARUS
#         PmmRW_SBND = []
#         PmmRW_ICARUS = []
#         for k in range(len(SBND_NuMuCC_TrueEDist)):
#             PmmRW_SBND.append(
#                 SBND_NuMuCC_TrueEDist[k]
#                 * sterile.PmmAvg(
#                     SBND_BinEdges_NuMu[k], SBND_BinEdges_NuMu[k + 1], osc.L_SBND
#                 )
#             )
#             PmmRW_ICARUS.append(
#                 ICARUS_NuMuCC_TrueEDist[k]
#                 * sterile.PmmAvg(
#                     SBND_BinEdges_NuMu[k], SBND_BinEdges_NuMu[k + 1], osc.L_ICARUS
#                 )
#             )
#         if energy_degradation:
#             PmmRW_SBND = sterile.EnergyDegradation(
#                 SBND_NuMuCC_TrueEDist,
#                 SBND_BinEdges_NuMu,
#                 which_channel="Pmm",
#                 which_experiment="microboone",
#             )
#             PmmRW_ICARUS = sterile.EnergyDegradation(
#                 ICARUS_NuMuCC_TrueEDist,
#                 SBND_BinEdges_NuMu,
#                 which_channel="Pmm",
#                 which_experiment="microboone",
#             )
#         if not decay and oscillations:
#             for k in range(len(SBND_NuMuCC_TrueEDist)):
#                 PmmRW_SBND[k] = SBND_NuMuCC_TrueEDist[k] * sterile.PmmoscAvg(
#                     SBND_BinEdges_NuMu[k], SBND_BinEdges_NuMu[k + 1], osc.L_SBND
#                 )
#                 PmmRW_ICARUS[k] = ICARUS_NuMuCC_TrueEDist[k] * sterile.PmmoscAvg(
#                     SBND_BinEdges_NuMu[k], SBND_BinEdges_NuMu[k + 1], osc.L_ICARUS
#                 )
#     RecoDist_SBND_0 = np.dot(NuMuCC_MigMat_SBND, PmmRW_SBND)
#     RecoDist_ICARUS_0 = np.dot(NuMuCC_MigMat_ICARUS, PmmRW_ICARUS)

#     RecoDist_SBND = []
#     RecoDist_ICARUS = []
#     for j in range(25):
#         RecoDist_SBND.append(
#             0.5 * (RecoDist_SBND_0[2 * j] + RecoDist_SBND_0[2 * j + 1])
#         )
#         RecoDist_ICARUS.append(
#             0.5 * (RecoDist_ICARUS_0[2 * j] + RecoDist_ICARUS_0[2 * j + 1])
#         )
#     RecoDist_SBND.append(np.sum(RecoDist_SBND_0[50:]))
#     RecoDist_ICARUS.append(np.sum(RecoDist_ICARUS_0[50:]))

#     SBND_Evts = [
#         RecoDist_SBND[kk] * NuMuCC_Eff_SBND[kk] for kk in range(len(NuMuCC_Eff_SBND))
#     ]
#     ICARUS_Evts = [
#         RecoDist_ICARUS[kk] * NuMuCC_Eff_ICARUS[kk]
#         for kk in range(len(NuMuCC_Eff_ICARUS))
#     ]

#     return [SBND_Evts, ICARUS_Evts]


# def SBN_OscChi2(
#     theta,
#     temp,
#     constrained=False,
#     RemoveOverflow=False,
#     sigReps=None,
#     Asimov=False,
#     oscillations=True,
#     decay=False,
#     decouple_decay=False,
#     disappearance=True,
#     energy_degradation=True,
#     helicity="conserving",
# ):
#     """Calculates the chi-squared from the full covariance matrix,
#     allowing for oscillated backgrounds (oscillating as a function of *reconstructed* neutrino energy)

#     "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
#     Default for our analyses will be "False"

#     "RemoveOverflow" allows for discarding the last (overflow) bin of each sample when calculating the test statistic

#     "Asimov" allows for determining the Asimov sensitivity expectation instead of the data-derived constraint

#     "sigReps" allows for replacement of the different signal samples (nu_e CC FC/PC, nu_mu CC FC/PC) instead of re-weighting the reconstructed-energy distributions
#     This allows for including oscillations as a function of *true* neutrino energy.

#     oscillations: bool, optional
#          whether to include oscillations in the flavor transition probability, by default True.
#          If False, then Losc goes to infinity.

#      decay: bool, optional
#          whether to include decay in the flavor transition probability, by default True.
#          If False, then Ldec goes to infinity.

#      decouple_decay : bool, optional
#          whether to decouple the decay rate like in deGouvea's model, by default False.
#          If True, then the decay rate is independent of the mixing angles and always into nu_e states.

#     disappearance: bool, optional
#         whether to include nu_e and nu_mu disappearance, by default True.
#         If False, Pmm = 1, Pee = 1

#     energy_degradation: bool, optional
#         whether to include energy degradation in disappearance channel, by default True.
#         If False, return to usual disappearance probability

#     helicity: str, optional
#         whether to include conserving or flipping helicity, by default "conserving".
#     """
#     CVStat = np.zeros(np.shape(FCov))
#     CVSyst = np.zeros(np.shape(FCov))
#     # Load the Sterile class from param_scan
#     sterile = Sterile(
#         theta,
#         oscillations=oscillations,
#         decay=decay,
#         decouple_decay=decouple_decay,
#         helicity=helicity,
#     )
#     if sigReps is not None:
#         if len(sigReps) != 7:
#             print("Signal Replacement Vector Needs to have 7 Elements!")
#             return 0
#     else:
#         sigReps = [None for k in range(7)]

#     SSRW = []
#     RWVec = []
#     for SI in range(len(micro.Sets)):
#         if sigReps[SI] is None:
#             ST = micro.SigTypes[SI]
#             BE = micro.BEdges[SI]

#             if ST == "nue":
#                 RWVec = [1.0 for kk in range(len(BE) - 1)]
#                 if disappearance:
#                     RWVec = [
#                         sterile.PeeAvg(BE[kk], BE[kk + 1], osc.L_micro)
#                         for kk in range(len(BE) - 1)
#                     ]
#                     if energy_degradation:
#                         RWVec = (
#                             sterile.EnergyDegradation(
#                                 micro.SigSets[SI],
#                                 BE,
#                                 which_channel="Pee",
#                                 which_experiment="microboone",
#                             )
#                             / micro.SigSets[SI]
#                         )
#                     if not decay and oscillations:
#                         RWVec = [
#                             sterile.PeeoscAvg(BE[kk], BE[kk + 1], osc.L_micro)
#                             for kk in range(len(BE) - 1)
#                         ]
#             elif ST == "numu":
#                 RWVec = [1.0 for kk in range(len(BE) - 1)]
#                 if disappearance:
#                     RWVec = [
#                         sterile.PmmAvg(BE[kk], BE[kk + 1], osc.L_micro)
#                         for kk in range(len(BE) - 1)
#                     ]
#                     if energy_degradation:
#                         RWVec = (
#                             sterile.EnergyDegradation(
#                                 micro.SigSets[SI],
#                                 BE,
#                                 which_channel="Pmm",
#                                 which_experiment="microboone",
#                             )
#                             / micro.SigSets[SI]
#                         )
#                     if not decay and oscillations:
#                         RWVec = [
#                             sterile.PmmoscAvg(BE[kk], BE[kk + 1], osc.L_micro)
#                             for kk in range(len(BE) - 1)
#                         ]
#             elif ST == "NCPi0" or ST == "numuPi0":
#                 RWVec = [1.0 for kk in range(len(BE) - 1)]
#             SSRW.append(RWVec * micro.SigSets[SI])
#         else:
#             SSRW.append(sigReps[SI])

#     SSRWF = np.concatenate(SSRW)
#     for ii in range(len(micro.SigSetsF)):
#         CVStat[ii][ii] = CNPStat(
#             micro.ObsSetsF[ii], SSRWF[ii] + micro.BkgSetsF[ii] + temp[ii]
#         )
#         for jj in range(len(micro.SigSetsF)):
#             CVSyst[ii][jj] = (
#                 micro.FCov[ii][jj]
#                 * (SSRWF[ii] + micro.BkgSetsF[ii] + temp[ii] + 1.0e-2)
#                 * (SSRWF[jj] + micro.BkgSetsF[jj] + temp[jj] + 1.0e-2)
#             )
#     CV = CVSyst + CVStat
#     if constrained:
#         CVYY = CV[26:, 26:]
#         CVXY = CV[:26, 26:]
#         CVYX = CV[26:, :26]
#         CVXX = CV[:26, :26]

#         nY = micro.ObsSetsF[26:]
#         muY = micro.BkgSetsF[26:] + SSRWF[26:] + temp[26:]
#         muX = micro.BkgSetsF[:26] + SSRWF[:26] + temp[:26]

#         muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY - muY)
#         CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

#         if Asimov:
#             nX = micro.BkgSetsF[:26] + micro.SigSetsF[:26]
#         else:
#             nX = micro.ObsSetsF[:26]
#         TS = np.dot(
#             np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25, :25])), nX[:25] - muXC[:25]
#         )
#     else:
#         if Asimov:
#             nXY = micro.BkgSetsF + micro.SigSetsF
#         else:
#             nXY = micro.ObsSetsF
#         muXY = micro.BkgSetsF + SSRWF + temp
#         XV = nXY - muXY
#         if RemoveOverflow:
#             XV[25], XV[51], XV[77], XV[103], XV[114], XV[125], XV[136] = (
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#             )

#         TS = np.dot(np.dot(XV, inv(CV)), XV)

#     return TS
