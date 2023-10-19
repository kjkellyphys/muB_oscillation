import numpy as np
import copy

import MicroTools as micro
from MicroTools import unfolder
from MicroTools.InclusiveTools.inclusive_osc_tools import DecayPmmAvg
from MicroTools.InclusiveTools.inclusive_osc_tools import Decay_muB_OscChi2
from MicroTools.InclusiveTools.inclusive_osc_tools import DecayMuBNuMuDis

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
MiniBooNE_Signal = micro.mb_mc_data_release
MB_Ereco_unfold_bins = micro.bin_edges_reco
MB_Ereco_official_bins = micro.bin_edges * 1e-3
LMBT = 0.4685  # Baseline length in kilometers
Ereco = MiniBooNE_Signal[:, 0] / 1000  # GeV
Etrue = MiniBooNE_Signal[:, 1] / 1000  # GeV
e_prod_e_int_bins = np.linspace(0, 3, 51)  # GeV
Length = MiniBooNE_Signal[:, 2] / 100000  # Kilometers
# Reweighted by a factor of 1/24860 to match Pedro's signal rate
Weight = MiniBooNE_Signal[:, 3] / len(MiniBooNE_Signal[:, 3])
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
    def __init__(self, gm4, Ue4Sq, Um4Sq):
        """__init__ Sterile neutrino class

        This is the model class.
        It should contain everything we need to compute for a fixed set of couplings.

        Parameters
        ----------
        gm4 : float
            sterile-scalar coupling constant
        Ue4Sq : float
            electron mixing squared
        Um4Sq : float
            muon mixing squared
        """
        self.gm4 = gm4
        self.Um4Sq = Um4Sq
        self.Ue4Sq = Ue4Sq

    def GammaLab(self, E4):
        """Etrue -- GeV"""
        return self.gm4**2 / (32 * np.pi * E4)

    def Pdecay(self, E4, Length):
        """E4 -- GeV, Length -- Kilometers"""
        # NOTE: Please check! There should be a factor of 4 with this unit conversion.
        return 1 - np.exp(-1.267 * (4 * self.GammaLab(E4) * Length))

    def dPdecaydX(self, Eparent, Edaughters):
        """The probability of daughter neutrino energy"""

        # NOTE -- this depends on the model -- not sure what is going on here?
        # decay_w_base = (
        #     np.linspace(1, 1 + 2 * (n_replications - 1), n_replications)
        #     / n_replications**2
        # )

        # I'm using this instead -- it might be (1 - Edaughters / Eparent) instead, should check.
        decay_w_base = Edaughters / Eparent

        return decay_w_base


# --------------------------------------------------------------------------------


def get_MBSig_Pmm_reweighted(gm4, Um4Sq, MBSig, MBbins):
    MBSig_rw = []
    # Reweight event rate in each bin
    for k in range(len(MBSig)):
        RWFact = 1 / DecayPmmAvg(MBbins[k], MBbins[k + 1], LMBT, gm4, Um4Sq)
        MBSig_rw.append(MBSig[k] * RWFact)
    return MBSig_rw


def DecayReturnMicroBooNEChi2(theta):
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
    # Weighted decay appearance probability from eqn 2.8 in 1911.01447
    sterile = Sterile(gm4, 0, Um4Sq)

    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

    # replicating the length entry of MC
    Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
    Weight_ext = np.stack(
        [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
    ).T.flatten()

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = Um4Sq * sterile.Pdecay(Etrue_parent, Length_ext)
    Pmm = 1 - Um4Sq * sterile.Pdecay(Etrue_parent, Length_ext)

    dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
    Weight_decay = Weight_ext * Pme * dPdX

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        migration_matrix_official_bins,
    )

    MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit)

    # Calculate the MicroBooNE chi2 by unfolding
    MBSig_for_unfolding = np.dot(
        (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
        migration_matrix_unfolding_bins,
    )

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig_for_unfolding)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig_for_unfolding)
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
