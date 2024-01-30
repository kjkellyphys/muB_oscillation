import numpy as np
import copy
from scipy.special import sici, expi
from scipy import integrate
import MicroTools as micro
from MicroTools import unfolder
from MicroTools.InclusiveTools.inclusive_osc_tools import DecayPmmAvg
from MicroTools.InclusiveTools.inclusive_osc_tools import (
    Decay_muB_OscChi2,
    DecayMuBNuMuDis,
    DecayMuBNuEDis,
)
import MiniTools as mini
import const

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
nue_bkg = [
    527.164624,
    315.423689,
    349.644825,
    186.211970,
    261.441799,
    195.534193,
    203.008745,
    165.664396,
    118.581365,
    143.989367,
    201.450357,
]
numu_MC = [
    56314.570151,
    86652.169005,
    77496.377767,
    54280.574682,
    33589.592857,
    18141.591625,
    8780.439580,
    4090.676535,
]


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


migration_matrix_unfolding_bins = create_reco_migration_matrix(
    MB_Ereco_unfold_bins
)  # 13 bins
migration_matrix_official_bins = create_reco_migration_matrix(
    MB_Ereco_official_bins
)  # 11 bins

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
    def __init__(self, theta, oscillations=True, decay=True, decouple_decay=False):
        """__init__ Sterile neutrino class

        This is the model class.
        It should contain everything we need to compute for a fixed set of couplings.

        Parameters
        ----------
        theta: dic
            The model parameters inside a dictionary. Entries should be:
                g : float
                    sterile-scalar coupling constant
                m4: float
                    sterile mass **in eV**
                Ue4Sq : float
                    electron mixing squared
                Um4Sq : float
                    muon mixing squared

        oscillations: bool, optional
            whether to include oscillations in the flavor transition probability, by default True.
            If False, then Losc goes to infinity.

        decay: bool, optional
            whether to include decay in the flavor transition probability, by default True.
            If False, then Ldec goes to infinity.

        decouple_decay : bool, optional
            whether to decouple the decay rate like in deGouvea's model, by default False.
            If True, then the decay rate is independent of the mixing angles and always into nu_e states.

        """
        self.g = 1
        self.m4 = 1
        self.Ue4Sq = 0
        self.Um4Sq = 0
        for k, v in theta.items():
            setattr(self, k, v)

        self.Us4Sq = 1 - self.Ue4Sq - self.Um4Sq  # Sterile mixing squared
        self.decouple_decay = decouple_decay
        self.oscillations = oscillations
        self.decay = decay

    def GammaLab(self, E4):
        """Decay rate in GeV, Etrue -- GeV"""
        if not self.decay:
            return 0
        else:
            if self.decouple_decay:
                return (self.g * self.m4 * 1e-9) ** 2 / (16 * np.pi * E4)
            else:
                return (
                    self.Us4Sq
                    * (1 - self.Us4Sq)
                    * (self.g * self.m4 * 1e-9) ** 2
                    / (16 * np.pi * E4)
                )

    def Ldec(self, E4):
        """Lab  frame decay length in km, E4 -- GeV"""
        if self.decay:
            return const.get_decay_rate_in_cm(self.GammaLab(E4)) * 1e-5
        else:
            return np.inf  # stable n4

    def Losc(self, E4):
        """Oscillation length in km, E4 -- GeV"""
        if self.oscillations:
            g = 1 / (2 * np.pi * E4 / (self.m4 * 1e-9) ** 2)  # mock rate in GeV
            return const.get_decay_rate_in_cm(g) * 1e-5
        else:
            return np.inf  # no oscillations

    def Fdecay(self, E4, Edaughter, Length):
        """Decay probability function, E4 -- GeV, Length -- Kilometers"""
        return (1 - np.exp(-Length / self.Ldec(E4))) * self.dPdecaydX(E4, Edaughter)

    def Fosc(self, E4, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        return (
            4
            * np.sin(np.pi / 2 * Length / self.Losc(E4)) ** 2
            * np.exp(-Length / self.Ldec(E4) / 2)
            + (1 - np.exp(-Length / self.Ldec(E4) / 2)) ** 2
        )

    def FdecayAvg(self, Emin, Emax, Length):
        """dPdecaydX --> 1"""
        integrand = lambda E4: (1 - np.exp(-Length / self.Ldec(E4)))
        return integrate.quad(integrand, Emin, Emax)[0] / (Emax - Emin)

    def FoscAvg(self, Emin, Emax, Length):
        integrand = lambda E4: self.Fosc(E4, Length)
        return integrate.quad(integrand, Emin, Emax)[0] / (Emax - Emin)

    def Pme(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.Um4Sq * self.Ue4Sq * self.Fosc(E4, Length)
        return pdecay + posc

    def Pmm(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.Um4Sq * (1 - self.Um4Sq) * self.Fosc(E4, Length)
        return 1 + pdecay - posc

    def Pee(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        pdecay = self.Ue4Sq * self.Fdecay(E4, Edaughter, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.Ue4Sq * (1 - self.Ue4Sq) * self.Fosc(E4, Length)
        return 1 + pdecay - posc

    def Pmmdecay(self, Emin, Emax, Eint, Length):
        # decay term in Pmm, Emin and Emax are E4 bin edges
        if Emax < 1:
            n = 2
        else:
            n = 1
        pdecay = self.Um4Sq * self.FdecayAvg(Emin, Emax, Length) * (Eint / Emax) ** n
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmmosc(self, Emin, Emax, Length):
        # osc term in Pmm, does not involve energy degradation
        return 1 - self.Um4Sq * (1 - self.Um4Sq) * self.FoscAvg(Emin, Emax, Length)

    def Peedecay(self, Emin, Emax, Eint, Length, noffset=0):
        # decay term in Pee, Emin and Emax are E4 bin edges
        if Emax < 1:
            n = 2 + noffset
        else:
            n = 1 + noffset
        pdecay = self.Ue4Sq * self.FdecayAvg(Emin, Emax, Length) * (Eint / Emax) ** n
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Peeosc(self, Emin, Emax, Length):
        # osc term in Pee, does not involve energy degradation
        return 1 - self.Ue4Sq * (1 - self.Ue4Sq) * self.FoscAvg(Emin, Emax, Length)

    # Next we migrate it to Ereco, and do the average

    def PmmAvg(self, Emin, Emax, Length):
        """
        Averaged Disappearance probability, E4 -- GeV, Length -- km
        E4 and Edaughter are approximated to be equal, since the discrepancy is suppressed by mixing squared
        """
        # Decay term
        pdecay = self.Um4Sq * self.FdecayAvg(Emin, Emax, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.Um4Sq * (1 - self.Um4Sq) * self.FoscAvg(Emin, Emax, Length)
        return 1 + pdecay - posc

    def PeeAvg(self, Emin, Emax, Length):
        """
        Averaged Disappearance probability, E4 -- GeV, Length -- km
        E4 and Edaughter are approximated to be equal, since the discrepancy is suppressed by mixing squared
        """
        # Decay term
        pdecay = self.Ue4Sq * self.FdecayAvg(Emin, Emax, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.Ue4Sq * (1 - self.Ue4Sq) * self.FoscAvg(Emin, Emax, Length)
        return 1 + pdecay - posc

    def dPdecaydX(self, Eparent, Edaughter):
        """The probability of daughter neutrino energy"""

        decay_w_base = Edaughter / Eparent

        return decay_w_base

    def Pdecay_binned_avg(self, E4_bin_edges, fixed_Length=LMBT):
        """E4_bin_edges -- array in GeV, Length -- Kilometers"""

        # NOTE: I guess we also have to update this to include oscillations etc.
        # My impression is that there's probably an easier way to use the same functions above to calculate the average,
        # instead of rewrite the formulae already integrated. Maybe enough to do a quad by hand?

        de = np.diff(E4_bin_edges)
        el = E4_bin_edges[:-1]

        # # NOTE: We should check our fits are independent of this choice!!
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

    def EnergyDegradation(self, Etrue_dist, Etrue_bins, which_channel):
        R_deg = np.zeros((len(Etrue_dist), len(Etrue_dist)))
        R_osc = []
        # degradation piece
        for k in range(len(Etrue_dist)):
            for i in range(k + 1):
                Pdecay = 1
                if which_channel == "Pee":
                    Pdecay = self.Peedecay(
                        Etrue_bins[k], Etrue_bins[k + 1], Etrue_bins[i + 1], LMBT
                    )
                elif which_channel == "Pmm":
                    Pdecay = self.Pmmdecay(
                        Etrue_bins[k], Etrue_bins[k + 1], Etrue_bins[i + 1], LMBT
                    )
                R_deg[k][i] = Pdecay * Etrue_dist[i]
        R_sum = np.sum(R_deg, axis=0)

        # oscillation piece
        for i in range(len(Etrue_dist)):
            Peeosc = self.Peeosc(Etrue_bins[i], Etrue_bins[i + 1], LMBT)
            R_osc.append(Peeosc * Etrue_dist[i])

        R_tot = R_sum + R_osc

        return R_tot


# --------------------------------------------------------------------------------
def DecayReturnMicroBooNEChi2(
    theta, oscillations=True, decay=False, decouple_decay=False
):
    """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

    Parameters
    ----------
    theta : dic
        Model Input parameters as a dictionary

    Returns
    -------
    list
        [gm4, Um4Sq, Ue4Sq, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
    """

    g = theta["g"]
    m4 = theta["m4"]
    Ue4Sq = theta["Ue4Sq"]
    Um4Sq = theta["Um4Sq"]

    # Our new physics class
    # For deGouvea's model, we fix m4 = 1 eV, and identify g = gm4.
    sterile = Sterile(
        theta, oscillations=oscillations, decay=decay, decouple_decay=decouple_decay
    )

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

    # replicating entries of the MC data release -- baseline L and weight
    Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
    Weight_ext = np.stack(
        [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
    ).T.flatten()

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = sterile.Pme(Etrue_parent, Etrue_daughter, Length_ext)

    Weight_decay = Weight_ext * Pme

    # Calculate the MiniBooNE chi2

    # Migrate nue signal from Etrue to Ereco with 11 bins
    MBSig_for_MBfit = np.dot(
        np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        migration_matrix_official_bins,
    )

    # Average disappearance in each bin of MB MC data release
    # P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
    # P_mumu_avg = (1 - Um4Sq) ** 2 + Um4Sq**2 * P_avg
    nue_bin_edges = MB_Ereco_official_bins
    numu_bin_edges = MB_Ereco_official_bins_numu
    # bin_c = (MB_Ereco_official_bins_numu[:-1] + MB_Ereco_official_bins_numu[1:]) / 2
    P_ee_avg = [
        sterile.PeeAvg(nue_bin_edges[i], nue_bin_edges[i + 1], LMBT)
        for i in range(len(nue_bin_edges) - 1)
    ]
    P_mumu_avg = [
        sterile.PmmAvg(numu_bin_edges[i], numu_bin_edges[i + 1], LMBT)
        for i in range(len(numu_bin_edges) - 1)
    ]
    # P_mumu_avg = sterile.Pmm(bin_c, bin_c, LMBT)
    Ree_true = sterile.EnergyDegradation(
        np.histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
        e_prod_e_int_bins,
        "Pee",
    )
    Rmm_true = sterile.EnergyDegradation(
        np.histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
        e_prod_e_int_bins,
        "Pmm",
    )
    migration_matrix_pee = create_reco_migration_matrix(nue_bin_edges)
    migration_matrix_pmm = create_reco_migration_matrix(numu_bin_edges)
    # Ree_reco = np.dot(Ree_true, migration_matrix_pee)
    # Rmm_reco = np.dot(Rmm_true, migration_matrix_pmm)
    Ree_reco = sterile.EnergyDegradation(nue_bkg, nue_bin_edges, "Pee")
    Rmm_reco = sterile.EnergyDegradation(numu_MC, numu_bin_edges, "Pmm")
    MB_chi2 = mini.fit.chi2_MiniBooNE_2020(
        MBSig_for_MBfit, Rmumu=Rmm_reco, Ree=Ree_reco
    )
    # MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=P_ee_avg)

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
    NuMuReps = DecayMuBNuMuDis(
        theta, oscillations=oscillations, decay=decay, decouple_decay=decouple_decay
    )
    # \nu_e disappearance signal replacement
    NuEReps = DecayMuBNuEDis(
        theta, oscillations=oscillations, decay=decay, decouple_decay=decouple_decay
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
    )

    return [g, m4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]


# def DecayReturnMicroBooNEChi2_3D(theta, decouple_decay=False):
#     """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

#     Parameters
#     ----------
#     theta : list
#         Model Input parameters as a list (gm4, Ue4Sq, Um4Sq)

#     Returns
#     -------
#     list
#         [gm4, Um4Sq, Ue4Sq, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
#     """

#     gm4, Ue4Sq, Um4Sq = theta

#     # Our new physics class
#     sterile = Sterile(gm4, Ue4Sq, Um4Sq, decouple_decay=decouple_decay)

#     # Replicating events for multiple daughter neutrino energies
#     Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

#     # replicating entries of the MC data release -- baseline L and weight
#     Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
#     Weight_ext = np.stack(
#         [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
#     ).T.flatten()

#     # Flavor transition probabilities -- Assuming nu4 decays only into nue
#     Pme = sterile.Pme(Etrue_parent, Length_ext)

#     dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
#     Weight_decay = Weight_ext * Pme * dPdX

#     # Calculate the MiniBooNE chi2
#     MBSig_for_MBfit = np.dot(
#         np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
#         migration_matrix_official_bins,
#     )

#     # Average disappearance in each bin of MB MC data release
#     P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
#     P_mumu_avg = 1 + (1 - Ue4Sq - Um4Sq) * Um4Sq**2 / (Ue4Sq + Um4Sq) * (1 - P_avg)

#     MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

#     # Calculate the MicroBooNE chi2 by unfolding
#     MBSig_for_unfolding = np.dot(
#         (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
#         migration_matrix_unfolding_bins,
#     )

#     # NOTE: not needed since we are fitting miniboone with nu_e and nu_mu samples simultaneously
#     # MBSig_for_unfolding_RW = []
#     # for k in range(len(MBSig_for_unfolding)):
#     # RWFact = 1 / DecayPmmAvg(
#     #         MB_Ereco_unfold_bins[k], MB_Ereco_unfold_bins[k + 1], LMBT, gm4, Um4Sq
#     #     )
#     #     MBSig_for_unfolding_RW.append(MBSig_for_unfolding[k] * RWFact)

#     MBSig_for_unfolding_RW = MBSig_for_unfolding  # No undoing of numu disappearance

#     # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
#     uBFC = GBFC.miniToMicro(MBSig_for_unfolding_RW)
#     uBFC = np.insert(uBFC, 0, [0.0])
#     uBFC = np.append(uBFC, 0.0)

#     # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
#     uBPC = GBPC.miniToMicro(MBSig_for_unfolding_RW)
#     uBPC = np.insert(uBPC, 0, [0.0])
#     uBPC = np.append(uBPC, 0.0)

#     uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

#     # \nu_mu disappearance signal replacement
#     NuMuReps = DecayMuBNuMuDis3D(gm4, Ue4Sq, Um4Sq)

#     # MicroBooNE
#     MuB_chi2 = Decay_muB_OscChi2_3D(
#         Ue4Sq,
#         Um4Sq,
#         gm4,
#         uBtemp,
#         constrained=False,
#         sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
#         RemoveOverflow=True,
#     )
#     MuB_chi2_Asimov = Decay_muB_OscChi2_3D(
#         Ue4Sq,
#         Um4Sq,
#         gm4,
#         uBtemp,
#         constrained=False,
#         sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
#         RemoveOverflow=True,
#         Asimov=True,
#     )

#     return [gm4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]


# def DecayReturnMicroBooNEChi2_4D(theta, decouple_decay=False):
#     """DecayReturnMicroBooNEChi2 Returns the MicroBooNE chi2

#     Parameters
#     ----------
#     theta : list
#         Model Input parameters as a list (gm4, Ue4Sq, Um4Sq)

#     Returns
#     -------
#     list
#         [gm4, Um4Sq, Ue4Sq, MiniBoone chi2, MicroBooNE chi2, MicroBooNE chi2 Asimov]
#     """

#     g, m4, Ue4Sq, Um4Sq = theta

#     # Our new physics class
#     sterile = Sterile(g, m4, Ue4Sq, Um4Sq, decouple_decay=decouple_decay)

#     # Replicating events for multiple daughter neutrino energies
#     Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int()

#     # replicating entries of the MC data release -- baseline L and weight
#     Length_ext = np.stack([Length for _ in range(NREPLICATION)], axis=0).T.flatten()
#     Weight_ext = np.stack(
#         [Weight / NREPLICATION for _ in range(NREPLICATION)], axis=0
#     ).T.flatten()

#     # Flavor transition probabilities -- Assuming nu4 decays only into nue
#     Pme = sterile.Pme_3D(Etrue_parent, Length_ext)

#     dPdX = sterile.dPdecaydX(Etrue_parent, Etrue_daughter)
#     Weight_decay = Weight_ext * Pme * dPdX

#     # Calculate the MiniBooNE chi2
#     MBSig_for_MBfit = np.dot(
#         np.histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
#         migration_matrix_official_bins,
#     )

#     # Average disappearance in each bin of MB MC data release
#     P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=LMBT)
#     P_mumu_avg = 1 + (1 - Ue4Sq - Um4Sq) * Um4Sq**2 / (Ue4Sq + Um4Sq) * (1 - P_avg)
#     P_ee_avg = 1 + (1 - Ue4Sq - Um4Sq) * Ue4Sq**2 / (Ue4Sq + Um4Sq) * (
#         1 - P_avg
#     )  # NOTE: what to do with this?

#     MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)

#     # Calculate the MicroBooNE chi2 by unfolding
#     MBSig_for_unfolding = np.dot(
#         (np.histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
#         migration_matrix_unfolding_bins,
#     )

#     # NOTE: not needed since we are fitting miniboone with nu_e and nu_mu samples simultaneously
#     # MBSig_for_unfolding_RW = []
#     # for k in range(len(MBSig_for_unfolding)):
#     # RWFact = 1 / DecayPmmAvg(
#     #         MB_Ereco_unfold_bins[k], MB_Ereco_unfold_bins[k + 1], LMBT, gm4, Um4Sq
#     #     )
#     #     MBSig_for_unfolding_RW.append(MBSig_for_unfolding[k] * RWFact)

#     MBSig_for_unfolding_RW = MBSig_for_unfolding  # No undoing of numu disappearance

#     # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
#     uBFC = GBFC.miniToMicro(MBSig_for_unfolding_RW)
#     uBFC = np.insert(uBFC, 0, [0.0])
#     uBFC = np.append(uBFC, 0.0)

#     # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
#     uBPC = GBPC.miniToMicro(MBSig_for_unfolding_RW)
#     uBPC = np.insert(uBPC, 0, [0.0])
#     uBPC = np.append(uBPC, 0.0)

#     uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

#     # \nu_mu disappearance signal replacement
#     NuMuReps = DecayMuBNuMuDis4D(g, m4, Ue4Sq, Um4Sq)

#     # MicroBooNE
#     MuB_chi2 = Decay_muB_OscChi2_4D(
#         Ue4Sq,
#         Um4Sq,
#         g,
#         m4,
#         uBtemp,
#         constrained=False,
#         sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
#         RemoveOverflow=True,
#     )
#     MuB_chi2_Asimov = Decay_muB_OscChi2_4D(
#         Ue4Sq,
#         Um4Sq,
#         g,
#         m4,
#         uBtemp,
#         constrained=False,
#         sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None],
#         RemoveOverflow=True,
#         Asimov=True,
#     )

#     return [g, m4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]
