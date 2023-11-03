import numpy as np
import copy
from scipy.special import sici, expi
import MicroTools as micro
from MicroTools import unfolder
from MicroTools.InclusiveTools.inclusive_osc_tools import DecayPmmAvg
from MicroTools.InclusiveTools.inclusive_osc_tools import (
    Decay_muB_OscChi2,
    Decay_muB_OscChi2_3D,
    Decay_muB_OscChi2_4D,
)
from MicroTools.InclusiveTools.inclusive_osc_tools import (
    DecayMuBNuMuDis,
    DecayMuBNuMuDis3D,
    DecayMuBNuMuDis4D,
)

import MiniTools as mini


RHE = False
UFMB = False
GBPC = unfolder.MBtomuB(
    analysis="1eX_PC", remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True
)
GBFC = unfolder.MBtomuB(
    analysis="1eX", remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True
)

# NOTE: NOT SURE WHAT THIS IS? WHY REWEIGHTED?
# MiniBooNE_Signal = np.loadtxt(
# f"{mb_data_osctables}/miniboone_numunuefullosc_ntuple_reweighted.dat"
# )

# Load the MiniBooNE MC from data release
MiniBooNE_Signal = micro.mb_mc_data_release  # NOTE: updated to 2020
# MiniBooNE_Signal = np.loadtxt(
#     "/Users/taozhou/Documents/GitHub/muB_oscillation/MiniTools/include/miniboone_2020/miniboone_numunuefullosc_ntuple.txt"
# )
MB_Ereco_unfold_bins = micro.bin_edges_reco
MB_Ereco_official_bins = micro.bin_edges * 1e-3
MB_Ereco_official_bins_numu = micro.bin_edges_numu * 1e-3

LMBT = 0.4685  # Baseline length in kilometers
Ereco = MiniBooNE_Signal[:, 0] / 1000  # GeV
Etrue = MiniBooNE_Signal[:, 1] / 1000  # GeV
e_prod_e_int_bins = np.linspace(0, 3, 51)  # GeV
Length = MiniBooNE_Signal[:, 2] / 100000  # Kilometers
# Reweighted by a factor of 1/24860 to match Pedro's signal rate
Weight = MiniBooNE_Signal[:, 3] / len(MiniBooNE_Signal[:, 3])
# Weight = MiniBooNE_Signal[:, 3] / 24860
NREPLICATION = 10


def create_reco_migration_matrix(MB_bins):
    # Set up a migration matrix that maps Etrue to Ereco with shape of (50,13)
    h0_unnorm = np.histogram2d(
        Etrue, Ereco, bins=[e_prod_e_int_bins, MB_bins], weights=Weight
    )[0]
    migration_matrix = copy.deepcopy(h0_unnorm)

    # Normalizing matrix elements w.r.t. to the interacting energy
    for j in range(len(e_prod_e_int_bins) - 1):
        row_sum = np.sum(h0_unnorm[j])
        if row_sum < 0.0:
            print("negative row?")
        if row_sum == 0.0:
            continue
        migration_matrix[j] = h0_unnorm[j] / row_sum
    return migration_matrix


migration_matrix_unfolding_bins = create_reco_migration_matrix(MB_Ereco_unfold_bins)
migration_matrix_official_bins = create_reco_migration_matrix(MB_Ereco_official_bins)

"""
    Create a distribution of interaction energy for every production energy
    based on the energy distribution of the daughter neutrinos (eqn 2.3&2.4 in 1911.01447)
    
    e_prod: parent neutrino energy
    n_replications: number of interaction energy bins per production energy

"""


def create_e_daughter(e_prod, n_replications=NREPLICATION):
    # e_prod: parent neutrino energy
    de = e_prod / n_replications
    return np.linspace(de / 2, e_prod - de / 2, n_replications)


def create_Etrue_and_Weight_int(n_replications=NREPLICATION):
    # For every Etrue, create a list of possible daughter neutrino energy
    Etrue_daughter = np.array(
        [create_e_daughter(e, n_replications=n_replications) for e in Etrue]
    )
    Etrue_extended = np.stack([Etrue for _ in range(n_replications)], axis=0).T

    return Etrue_extended.flatten(), Etrue_daughter.flatten()


# --------------------------------------------------------------------------------
class Sterile:
    def __init__(self, g, m4, Ue4Sq, Um4Sq, decouple_decay=False):
        """__init__ Sterile neutrino class

        This is the model class.
        It should contain everything we need to compute for a fixed set of couplings.

        Parameters
        ----------
        g : float
            sterile-scalar coupling constant
        m4: float
            sterile mass
        Ue4Sq : float
            electron mixing squared
        Um4Sq : float
            muon mixing squared
        """
        self.g = g
        self.m4 = m4
        self.Um4Sq = Um4Sq
        self.Ue4Sq = Ue4Sq
        self.Us4Sq = 1 - self.Ue4Sq - self.Um4Sq  # Sterile mixing squared
        self.decouple_decay = decouple_decay

    def GammaLab(self, E4):
        """Etrue -- GeV"""
        if self.decouple_decay:
            return (self.g * self.m4) ** 2 / (16 * np.pi * E4)
        else:
            return (
                self.Us4Sq
                * (1 - self.Us4Sq)
                * (self.g * self.m4) ** 2
                / (16 * np.pi * E4)
            )

    def Pdecay(self, E4, Length):
        """E4 -- GeV, Length -- Kilometers"""
        return 1 - np.exp(-1.267 * (4 * self.GammaLab(E4) * Length))

    def Pme(self, E4, Length):
        return (
            self.Us4Sq
            * self.Um4Sq
            * self.Ue4Sq
            / (1 - self.Us4Sq)
            * self.Pdecay(E4, Length)
        )

    def Pdecay_binned_avg(self, E4_bin_edges, fixed_Length=LMBT):
        """E4_bin_edges -- array in GeV, Length -- Kilometers"""
        de = np.diff(E4_bin_edges)
        el = E4_bin_edges[:-1]

        # NOTE: We should check our fits are independent of this choice!!
        el[el == 0] = 1e-3  # 1 MeV regulator
        er = E4_bin_edges[1:]

        # exponential argument
        x = -1.267 * (4 * self.GammaLab(1) * fixed_Length)

        return (
            1
            / de
            * (
                (er * np.exp(x / er) - x * expi(x / er))
                - (el * np.exp(x / el) - x * expi(x / el))
            )
        )

    def dPdecaydX(self, Eparent, Edaughter):
        """The probability of daughter neutrino energy"""

        decay_w_base = Edaughter / Eparent

        return decay_w_base


# --------------------------------------------------------------------------------
def DecayReturnMicroBooNEChi2(theta, decouple_decay=True):
    """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

    Parameters
    ----------
    theta : list
        Model Input parameters as a list (gm4, Um4Sq)

    Returns
    -------
    list
        [Um4Sq, gm4, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
    """

    Um4Sq, gm4 = theta

    # Our new physics class
    # For deGouvea's model, we fix m4 = 1 eV, and identify g = gm4.
    sterile = Sterile(gm4, 1, 0, Um4Sq, decouple_decay=decouple_decay)

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

    # replicating entries of the MC data release -- baseline L and weight
    Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
    Weight_ext = np.stack(
        [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
    ).T.flatten()

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = Um4Sq * sterile.Pdecay(Etrue_parent, Length_ext)

    dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
    Weight_decay = Weight_ext * Pme * dPdX

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        migration_matrix_official_bins,
    )

    # Average disappearance in each bin of MB MC data release
    P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
    P_mumu_avg = (1 - Um4Sq) ** 2 + Um4Sq**2 * P_avg

    MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

    # Calculate the MicroBooNE chi2 by unfolding
    MBSig_for_unfolding = np.dot(
        (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
        migration_matrix_unfolding_bins,
    )

    # NOTE: not needed since we are fitting miniboone with nu_e and nu_mu samples simultaneously
    # MBSig_for_unfolding_RW = []
    # for k in range(len(MBSig_for_unfolding)):
    # RWFact = 1 / DecayPmmAvg(
    #         MB_Ereco_unfold_bins[k], MB_Ereco_unfold_bins[k + 1], LMBT, gm4, Um4Sq
    #     )
    #     MBSig_for_unfolding_RW.append(MBSig_for_unfolding[k] * RWFact)

    MBSig_for_unfolding_RW = MBSig_for_unfolding  # No undoing of numu disappearance

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig_for_unfolding_RW)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig_for_unfolding_RW)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis(gm4, Um4Sq)

    # MicroBooNE
    MuB_chi2 = Decay_muB_OscChi2(
        Um4Sq,
        gm4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
    )
    MuB_chi2_Asimov = Decay_muB_OscChi2(
        Um4Sq,
        gm4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        Asimov=True,
    )

    return [Um4Sq, gm4, MB_chi2, MuB_chi2, MuB_chi2_Asimov]


def DecayReturnMicroBooNEChi2_3D(theta, decouple_decay=False):
    """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

    Parameters
    ----------
    theta : list
        Model Input parameters as a list (gm4, Ue4Sq, Um4Sq)

    Returns
    -------
    list
        [gm4, Um4Sq, Ue4Sq, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
    """

    gm4, Ue4Sq, Um4Sq = theta

    # Our new physics class
    sterile = Sterile(gm4, Ue4Sq, Um4Sq, decouple_decay=decouple_decay)

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

    # replicating entries of the MC data release -- baseline L and weight
    Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
    Weight_ext = np.stack(
        [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
    ).T.flatten()

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = sterile.Pme_3D(Etrue_parent, Length_ext)

    dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
    Weight_decay = Weight_ext * Pme * dPdX

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        migration_matrix_official_bins,
    )

    # Average disappearance in each bin of MB MC data release
    P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
    P_mumu_avg = 1 + (1 - Ue4Sq - Um4Sq) * Um4Sq**2 / (Ue4Sq + Um4Sq) * (1 - P_avg)

    MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

    # Calculate the MicroBooNE chi2 by unfolding
    MBSig_for_unfolding = np.dot(
        (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
        migration_matrix_unfolding_bins,
    )

    # NOTE: not needed since we are fitting miniboone with nu_e and nu_mu samples simultaneously
    # MBSig_for_unfolding_RW = []
    # for k in range(len(MBSig_for_unfolding)):
    # RWFact = 1 / DecayPmmAvg(
    #         MB_Ereco_unfold_bins[k], MB_Ereco_unfold_bins[k + 1], LMBT, gm4, Um4Sq
    #     )
    #     MBSig_for_unfolding_RW.append(MBSig_for_unfolding[k] * RWFact)

    MBSig_for_unfolding_RW = MBSig_for_unfolding  # No undoing of numu disappearance

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig_for_unfolding_RW)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig_for_unfolding_RW)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis3D(gm4, Ue4Sq, Um4Sq)

    # MicroBooNE
    MuB_chi2 = Decay_muB_OscChi2_3D(
        Ue4Sq,
        Um4Sq,
        gm4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
    )
    MuB_chi2_Asimov = Decay_muB_OscChi2_3D(
        Ue4Sq,
        Um4Sq,
        gm4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        Asimov=True,
    )

    return [gm4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]


def DecayReturnMicroBooNEChi2_4D(theta, decouple_decay=False):
    """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

    Parameters
    ----------
    theta : list
        Model Input parameters as a list (gm4, Ue4Sq, Um4Sq)

    Returns
    -------
    list
        [gm4, Um4Sq, Ue4Sq, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
    """

    g, m4, Ue4Sq, Um4Sq = theta

    # Our new physics class
    sterile = Sterile(g, m4, Ue4Sq, Um4Sq, decouple_decay=decouple_decay)

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

    # replicating entries of the MC data release -- baseline L and weight
    Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
    Weight_ext = np.stack(
        [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
    ).T.flatten()

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = sterile.Pme_3D(Etrue_parent, Length_ext)

    dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
    Weight_decay = Weight_ext * Pme * dPdX

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        migration_matrix_official_bins,
    )

    # Average disappearance in each bin of MB MC data release
    P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
    P_mumu_avg = 1 + (1 - Ue4Sq - Um4Sq) * Um4Sq**2 / (Ue4Sq + Um4Sq) * (1 - P_avg)
    P_ee_avg = 1 + (1 - Ue4Sq - Um4Sq) * Ue4Sq**2 / (Ue4Sq + Um4Sq) * (
        1 - P_avg
    )  # NOTE: what to do with this?

    MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

    # Calculate the MicroBooNE chi2 by unfolding
    MBSig_for_unfolding = np.dot(
        (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
        migration_matrix_unfolding_bins,
    )

    # NOTE: not needed since we are fitting miniboone with nu_e and nu_mu samples simultaneously
    # MBSig_for_unfolding_RW = []
    # for k in range(len(MBSig_for_unfolding)):
    # RWFact = 1 / DecayPmmAvg(
    #         MB_Ereco_unfold_bins[k], MB_Ereco_unfold_bins[k + 1], LMBT, gm4, Um4Sq
    #     )
    #     MBSig_for_unfolding_RW.append(MBSig_for_unfolding[k] * RWFact)

    MBSig_for_unfolding_RW = MBSig_for_unfolding  # No undoing of numu disappearance

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig_for_unfolding_RW)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig_for_unfolding_RW)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis4D(g, m4, Ue4Sq, Um4Sq)

    # MicroBooNE
    MuB_chi2 = Decay_muB_OscChi2_4D(
        Ue4Sq,
        Um4Sq,
        g,
        m4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
    )
    MuB_chi2_Asimov = Decay_muB_OscChi2_4D(
        Ue4Sq,
        Um4Sq,
        g,
        m4,
        uBtemp,
        constrained=False,
        sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        Asimov=True,
    )

    return [g, m4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]
