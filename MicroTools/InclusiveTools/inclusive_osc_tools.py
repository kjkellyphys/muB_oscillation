import numpy as np
from scipy.linalg import inv
from scipy.special import sici, expi

from MicroTools import unfolder
from MicroTools import muB_inclusive_datarelease_path, muB_inclusive_data_path
from MicroTools.sterile_tools import Sterile

import copy

GBPC_NuE = unfolder.MBtomuB(
    analysis="1eX_PC",
    remove_high_energy=False,
    unfold=False,
    effNoUnfold=False,
    which_template="2018",
)
GBFC_NuE = unfolder.MBtomuB(
    analysis="1eX",
    remove_high_energy=False,
    unfold=False,
    effNoUnfold=False,
    which_template="2018",
)

Sets = [
    "nueCC_FC_",
    "nueCC_PC_",
    "numuCC_FC_",
    "numuCC_PC_",
    "numuCCpi0_FC_",
    "numuCCpi0_PC_",
    "NCpi0_",
]
LEEStr, SigStr, BkgStr, ObsStr = "LEE.npy", "Sig.npy", "Bkg.npy", "Obs.npy"
LEESets = [np.load(muB_inclusive_datarelease_path + si + LEEStr) for si in Sets]
SigSets = [np.load(muB_inclusive_datarelease_path + si + SigStr) for si in Sets]
BkgSets = [np.load(muB_inclusive_datarelease_path + si + BkgStr) for si in Sets]
ObsSets = [np.load(muB_inclusive_datarelease_path + si + ObsStr) for si in Sets]

LEESetsF = np.concatenate(LEESets)
SigSetsF = np.concatenate(SigSets)
BkgSetsF = np.concatenate(BkgSets)
ObsSetsF = np.concatenate(ObsSets)

FCov = np.load(muB_inclusive_datarelease_path + "MuBInclusive_FracCov_Square.npy")

SigTypes = ["nue", "nue", "numu", "numu", "numuPi0", "numuPi0", "NCPi0"]
BEdges0 = [0.0 + 0.1 * j for j in range(26)]
BEdges0.append(10.0)
Pi0BEdges0 = [0.0 + 0.1 * j for j in range(11)]
Pi0BEdges0.append(10.0)
BEdges = [BEdges0, BEdges0, BEdges0, BEdges0, Pi0BEdges0, Pi0BEdges0, Pi0BEdges0]
LMBT = 0.4685  # Baseline length in kilometers


###########
# Numu data
NuMuCC_TrueEDist_FC = np.loadtxt(f"{muB_inclusive_data_path}/TrueEDist_numuCC_FC.dat")
NuMuCC_MigMat_FC = np.loadtxt(f"{muB_inclusive_data_path}/MigMat_numuCC_FC.dat")
NuMuCC_Eff_FC = np.loadtxt(f"{muB_inclusive_data_path}/Efficiency_numuCC_FC.dat")

NuMuCC_TrueEDist_PC = np.loadtxt(f"{muB_inclusive_data_path}/TrueEDist_numuCC_PC.dat")
NuMuCC_MigMat_PC = np.loadtxt(f"{muB_inclusive_data_path}/MigMat_numuCC_PC.dat")
NuMuCC_Eff_PC = np.loadtxt(f"{muB_inclusive_data_path}/Efficiency_numuCC_PC.dat")

MuB_BinEdges_NuMu = [0.0 + 0.05 * j for j in range(61)]


# appearance probability averaged over energy bins for decay model from Eq (2.8) in 1911.01447
def expAvg(Emin, Emax, L, gm4):
    if Emin == 0.0:
        Emin = 0.000001
    x = -4 * 1.267 * gm4**2 * L / (32 * np.pi)
    return (
        1
        / (Emax - Emin)
        * (
            (Emax * np.exp(x / Emax) - x * expi(x / Emax))
            - (Emin * np.exp(x / Emin) - x * expi(x / Emin))
        )
    )


def DecayPmmAvg(Emin, Emax, L, gm4, Um4sq):
    return (1.0 - Um4sq) ** 2 + Um4sq**2 * expAvg(Emin, Emax, L, gm4)


def ssqAvg(Emin, Emax, L, dmsq):
    if Emin == 0.0:
        Emin = 0.000001
    xmin, xmax = Emin / (1.267 * dmsq * L), Emax / (1.267 * dmsq * L)
    return (
        1.267
        * dmsq
        * L
        / (Emax - Emin)
        * (
            (xmax * np.sin(1.0 / xmax) ** 2 - sici(2.0 / xmax)[0])
            - (xmin * np.sin(1.0 / xmin) ** 2 - sici(2.0 / xmin)[0])
        )
    )


def PeeAvg(Emin, Emax, L, dmsq, Ue4sq):
    return 1.0 - 4.0 * Ue4sq * (1.0 - Ue4sq) * ssqAvg(Emin, Emax, L, dmsq)


def PmmAvg(Emin, Emax, L, dmsq, Um4sq):
    return 1.0 - 4.0 * Um4sq * (1.0 - Um4sq) * ssqAvg(Emin, Emax, L, dmsq)


def PmeAvg(Emin, Emax, L, dmsq, Ue4sq, Um4sq):
    return 4.0 * Ue4sq * Um4sq * ssqAvg(Emin, Emax, L, dmsq)


def PmsAvg(Emin, Emax, L, dmsq, Ue4sq, Um4sq):
    return 4.0 * Um4sq * (1.0 - Ue4sq - Um4sq) * ssqAvg(Emin, Emax, L, dmsq)


# Disappearance probability from Matheus' notes
def expAvg3D(Ue4sq, Um4sq, Emin, Emax, L, gm4):
    if Emin == 0.0:
        Emin = 0.000001
    x = -(1 - Um4sq - Ue4sq) * (Um4sq + Ue4sq) * 1.267 * 4 * gm4**2 * L / (16 * np.pi)
    return (
        1
        / (Emax - Emin)
        * (
            (Emax * np.exp(x / Emax) - x * expi(x / Emax))
            - (Emin * np.exp(x / Emin) - x * expi(x / Emin))
        )
    )


def DecayPmmAvg3D(Emin, Emax, L, gm4, Ue4sq, Um4sq):
    return 1 + (1 - Ue4sq - Um4sq) * Um4sq**2 / (Ue4sq + Um4sq) * (
        1 - expAvg3D(Ue4sq, Um4sq, Emin, Emax, L, gm4)
    )


def expAvg4D(Ue4sq, Um4sq, Emin, Emax, L, g, m4):
    if Emin == 0.0:
        Emin = 0.000001
    x = (
        -(1 - Um4sq - Ue4sq)
        * (Um4sq + Ue4sq)
        * 1.267
        * 4
        * (g * m4) ** 2
        * L
        / (16 * np.pi)
    )
    return (
        1
        / (Emax - Emin)
        * (
            (Emax * np.exp(x / Emax) - x * expi(x / Emax))
            - (Emin * np.exp(x / Emin) - x * expi(x / Emin))
        )
    )


def DecayPmmAvg4D(Emin, Emax, L, g, m4, Ue4sq, Um4sq):
    return 1 + (1 - Ue4sq - Um4sq) * Um4sq**2 / (Ue4sq + Um4sq) * (
        1 - expAvg4D(Ue4sq, Um4sq, Emin, Emax, L, g, m4)
    )


def CNPStat(ni, mi):
    """Combined Neyman-Pearson Statistical Uncertainty
    Arguments:
       ni {int} -- Observation in bin i
       mi {float} -- Model Expectation in bin i
    Returns:
       [float] -- Statistical uncertainty of bin i
       See arXiv:1903.07185 for more details
    """
    if ni == 0.0:
        return mi / 2.0
    else:
        return 3.0 / (1.0 / ni + 2.0 / mi)


def muB_NoBkgOsc_Chi2(temp, constrained=False, Asimov=False, RemoveOverflow=False):
    """Calculates the chi-squared from the full covariance matrix,
    Focuses only on the addition of a signal template above background, with no additional background oscillation

    "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
    Default for our analyses will be "False"

    "Asimov" allows for calculation of Asimov-expected chi-squared, i.e. setting data to be equal to background expectation

    "RemoveOverflow" allows for discarding the last bin of each sample when calculating the test stastitic
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SigSetsF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = (
                FCov[ii][jj]
                * (SigSetsF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)
                * (SigSetsF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
            )
    CV = CVSyst + CVStat

    if constrained:
        CVYY = CV[26:, 26:]
        CVXY = CV[:26, 26:]
        CVYX = CV[26:, :26]
        CVXX = CV[:26, :26]

        if Asimov:
            nY = BkgSetsF[26:] + SigSetsF[26:]
        else:
            nY = ObsSetsF[26:]
        muY = BkgSetsF[26:] + SigSetsF[26:] + temp[26:]
        muX = BkgSetsF[:26] + SigSetsF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY - muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        if Asimov:
            nX = BkgSetsF[:26] + SigSetsF[:26]
        else:
            nX = ObsSetsF[:26]
        TS = np.dot(
            np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25, :25])), nX[:25] - muXC[:25]
        )
    else:
        if Asimov:
            nXY = BkgSetsF + SigSetsF
        else:
            nXY = ObsSetsF
        muXY = BkgSetsF + SigSetsF + temp
        XV = nXY - muXY
        if RemoveOverflow:
            XV[25], XV[51], XV[77], XV[103], XV[114], XV[125], XV[136] = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        TS = np.dot(np.dot(XV, inv(CV)), XV)

    return TS


def muB_OscChi2(
    Ue4sq,
    Um4sq,
    dm41,
    temp,
    constrained=False,
    RemoveOverflow=False,
    sigReps=None,
    Asimov=False,
):
    """Calculates the chi-squared from the full covariance matrix,
    allowing for oscillated backgrounds (oscillating as a function of *reconstructed* neutrino energy)

    "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
    Default for our analyses will be "False"

    "RemoveOverflow" allows for discarding the last (overflow) bin of each sample when calculating the test statistic

    "Asimov" allows for determining the Asimov sensitivity expectation instead of the data-derived constraint

    "sigReps" allows for replacement of the different signal samples (nu_e CC FC/PC, nu_mu CC FC/PC) instead of re-weighting the reconstructed-energy distributions
    This allows for including oscillations as a function of *true* neutrino energy.
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    if sigReps is not None:
        if len(sigReps) != 7:
            print("Signal Replacement Vector Needs to have 7 Elements!")
            return 0
    else:
        sigReps = [None for k in range(7)]

    SSRW = []
    for SI in range(len(Sets)):
        if sigReps[SI] is None:
            ST = SigTypes[SI]
            BE = BEdges[SI]

            if ST == "nue":
                RWVec = [
                    PeeAvg(BE[kk], BE[kk + 1], LMBT, dm41, Ue4sq)
                    for kk in range(len(BE) - 1)
                ]
            elif ST == "numu":
                RWVec = [
                    PmmAvg(BE[kk], BE[kk + 1], LMBT, dm41, Um4sq)
                    for kk in range(len(BE) - 1)
                ]
            elif ST == "NCPi0" or ST == "numuPi0":
                RWVec = [1.0 for kk in range(len(BE) - 1)]

            SSRW.append(SigSets[SI] * RWVec)
        else:
            SSRW.append(sigReps[SI])

    SSRWF = np.concatenate(SSRW)
    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SSRWF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = (
                FCov[ii][jj]
                * (SSRWF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)
                * (SSRWF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
            )
    CV = CVSyst + CVStat

    if constrained:
        CVYY = CV[26:, 26:]
        CVXY = CV[:26, 26:]
        CVYX = CV[26:, :26]
        CVXX = CV[:26, :26]

        nY = ObsSetsF[26:]
        muY = BkgSetsF[26:] + SSRWF[26:] + temp[26:]
        muX = BkgSetsF[:26] + SSRWF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY - muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        if Asimov:
            nX = BkgSetsF[:26] + SigSetsF[:26]
        else:
            nX = ObsSetsF[:26]
        TS = np.dot(
            np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25, :25])), nX[:25] - muXC[:25]
        )
    else:
        if Asimov:
            nXY = BkgSetsF + SigSetsF
        else:
            nXY = ObsSetsF
        muXY = BkgSetsF + SSRWF + temp
        XV = nXY - muXY
        if RemoveOverflow:
            XV[25], XV[51], XV[77], XV[103], XV[114], XV[125], XV[136] = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        TS = np.dot(np.dot(XV, inv(CV)), XV)

    return TS


MCT = np.load(f"{muB_inclusive_data_path}/MuB_NuE_True.npy")
MuB_True_BinEdges = [
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

'''def Decay_muB_OscChi2(
    Um4sq,
    gm4,
    temp,
    constrained=False,
    RemoveOverflow=False,
    sigReps=None,
    Asimov=False,
):
    """Calculates the chi-squared from the full covariance matrix,
    allowing for oscillated backgrounds (oscillating as a function of *reconstructed* neutrino energy)

    "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
    Default for our analyses will be "False"

    "RemoveOverflow" allows for discarding the last (overflow) bin of each sample when calculating the test statistic

    "Asimov" allows for determining the Asimov sensitivity expectation instead of the data-derived constraint

    "sigReps" allows for replacement of the different signal samples (nu_e CC FC/PC, nu_mu CC FC/PC) instead of re-weighting the reconstructed-energy distributions
    This allows for including oscillations as a function of *true* neutrino energy.
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))

    if sigReps is not None:
        if len(sigReps) != 7:
            print("Signal Replacement Vector Needs to have 7 Elements!")
            return 0
    else:
        sigReps = [None for k in range(7)]

    SSRW = []
    for SI in range(len(Sets)):
        if sigReps[SI] is None:
            ST = SigTypes[SI]
            BE = BEdges[SI]

            if ST == "nue":
                RWVec = [1.0 for kk in range(len(BE) - 1)]
            elif ST == "numu":
                RWVec = [
                    DecayPmmAvg(BE[kk], BE[kk + 1], LMBT, gm4, Um4sq)
                    for kk in range(len(BE) - 1)
                ]
            elif ST == "NCPi0" or ST == "numuPi0":
                RWVec = [1.0 for kk in range(len(BE) - 1)]

            SSRW.append(SigSets[SI] * RWVec)
        else:
            SSRW.append(sigReps[SI])

    SSRWF = np.concatenate(SSRW)
    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SSRWF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = (
                FCov[ii][jj]
                * (SSRWF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)
                * (SSRWF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
            )
    CV = CVSyst + CVStat

    if constrained:
        CVYY = CV[26:, 26:]
        CVXY = CV[:26, 26:]
        CVYX = CV[26:, :26]
        CVXX = CV[:26, :26]

        nY = ObsSetsF[26:]
        muY = BkgSetsF[26:] + SSRWF[26:] + temp[26:]
        muX = BkgSetsF[:26] + SSRWF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY - muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        if Asimov:
            nX = BkgSetsF[:26] + SigSetsF[:26]
        else:
            nX = ObsSetsF[:26]
        TS = np.dot(
            np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25, :25])), nX[:25] - muXC[:25]
        )
    else:
        if Asimov:
            nXY = BkgSetsF + SigSetsF
        else:
            nXY = ObsSetsF
        muXY = BkgSetsF + SSRWF + temp
        XV = nXY - muXY
        if RemoveOverflow:
            XV[25], XV[51], XV[77], XV[103], XV[114], XV[125], XV[136] = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        TS = np.dot(np.dot(XV, inv(CV)), XV)

    return TS
'''


def Decay_muB_OscChi2(
    theta,
    temp,
    constrained=False,
    RemoveOverflow=False,
    sigReps=None,
    Asimov=False,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=True,
    energy_degradation=True,
    helicity="conserving",
):
    """Calculates the chi-squared from the full covariance matrix,
    allowing for oscillated backgrounds (oscillating as a function of *reconstructed* neutrino energy)

    "constrained" is an option of whether to apply the Covariance-Matrix-Constraint method on the nu_e CC fully-contained sample
    Default for our analyses will be "False"

    "RemoveOverflow" allows for discarding the last (overflow) bin of each sample when calculating the test statistic

    "Asimov" allows for determining the Asimov sensitivity expectation instead of the data-derived constraint

    "sigReps" allows for replacement of the different signal samples (nu_e CC FC/PC, nu_mu CC FC/PC) instead of re-weighting the reconstructed-energy distributions
    This allows for including oscillations as a function of *true* neutrino energy.

    oscillations: bool, optional
         whether to include oscillations in the flavor transition probability, by default True.
         If False, then Losc goes to infinity.

     decay: bool, optional
         whether to include decay in the flavor transition probability, by default True.
         If False, then Ldec goes to infinity.

     decouple_decay : bool, optional
         whether to decouple the decay rate like in deGouvea's model, by default False.
         If True, then the decay rate is independent of the mixing angles and always into nu_e states.

    disappearance: bool, optional
        whether to include nu_e and nu_mu disappearance, by default True.
        If False, Pmm = 1, Pee = 1

    energy_degradation: bool, optional
        whether to include energy degradation in disappearance channel, by default True.
        If False, return to usual disappearance probability

    helicity: str, optional
        whether to include conserving or flipping helicity, by default "conserving".
    """
    CVStat = np.zeros(np.shape(FCov))
    CVSyst = np.zeros(np.shape(FCov))
    # Load the Sterile class from param_scan
    sterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        helicity=helicity,
    )
    if sigReps is not None:
        if len(sigReps) != 7:
            print("Signal Replacement Vector Needs to have 7 Elements!")
            return 0
    else:
        sigReps = [None for k in range(7)]

    SSRW = []
    RWVec = []
    for SI in range(len(Sets)):
        if sigReps[SI] is None:
            ST = SigTypes[SI]
            BE = BEdges[SI]

            if ST == "nue":
                RWVec = [1.0 for kk in range(len(BE) - 1)]
                if disappearance:
                    RWVec = [
                        sterile.PeeAvg(BE[kk], BE[kk + 1], LMBT)
                        for kk in range(len(BE) - 1)
                    ]
                    if energy_degradation:
                        RWVec = (
                            sterile.EnergyDegradation(
                                SigSets[SI],
                                BE,
                                which_channel="Pee",
                                which_experiment="microboone",
                            )
                            / SigSets[SI]
                        )
                    if not decay and oscillations:
                        RWVec = [
                            sterile.PeeoscAvg(BE[kk], BE[kk + 1], LMBT)
                            for kk in range(len(BE) - 1)
                        ]
            elif ST == "numu":
                RWVec = [1.0 for kk in range(len(BE) - 1)]
                if disappearance:
                    RWVec = [
                        sterile.PmmAvg(BE[kk], BE[kk + 1], LMBT)
                        for kk in range(len(BE) - 1)
                    ]
                    if energy_degradation:
                        RWVec = (
                            sterile.EnergyDegradation(
                                SigSets[SI],
                                BE,
                                which_channel="Pmm",
                                which_experiment="microboone",
                            )
                            / SigSets[SI]
                        )
                    if not decay and oscillations:
                        RWVec = [
                            sterile.PmmoscAvg(BE[kk], BE[kk + 1], LMBT)
                            for kk in range(len(BE) - 1)
                        ]
            elif ST == "NCPi0" or ST == "numuPi0":
                RWVec = [1.0 for kk in range(len(BE) - 1)]
            SSRW.append(RWVec * SigSets[SI])
        else:
            SSRW.append(sigReps[SI])

    SSRWF = np.concatenate(SSRW)
    for ii in range(len(SigSetsF)):
        CVStat[ii][ii] = CNPStat(ObsSetsF[ii], SSRWF[ii] + BkgSetsF[ii] + temp[ii])
        for jj in range(len(SigSetsF)):
            CVSyst[ii][jj] = (
                FCov[ii][jj]
                * (SSRWF[ii] + BkgSetsF[ii] + temp[ii] + 1.0e-2)
                * (SSRWF[jj] + BkgSetsF[jj] + temp[jj] + 1.0e-2)
            )
    CV = CVSyst + CVStat
    if constrained:
        CVYY = CV[26:, 26:]
        CVXY = CV[:26, 26:]
        CVYX = CV[26:, :26]
        CVXX = CV[:26, :26]

        nY = ObsSetsF[26:]
        muY = BkgSetsF[26:] + SSRWF[26:] + temp[26:]
        muX = BkgSetsF[:26] + SSRWF[:26] + temp[:26]

        muXC = muX + np.dot(np.dot(CVXY, inv(CVYY)), nY - muY)
        CVXXc = CVXX - np.dot(np.dot(CVXY, inv(CVYY)), CVYX)

        if Asimov:
            nX = BkgSetsF[:26] + SigSetsF[:26]
        else:
            nX = ObsSetsF[:26]
        TS = np.dot(
            np.dot(nX[:25] - muXC[:25], inv(CVXXc[:25, :25])), nX[:25] - muXC[:25]
        )
    else:
        if Asimov:
            nXY = BkgSetsF + SigSetsF
        else:
            nXY = ObsSetsF
        muXY = BkgSetsF + SSRWF + temp
        XV = nXY - muXY
        if RemoveOverflow:
            XV[25], XV[51], XV[77], XV[103], XV[114], XV[125], XV[136] = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        TS = np.dot(np.dot(XV, inv(CV)), XV)

    return TS


def MuBNuEDis(dm41, Ue4Sq):
    """Function for reweighting MicroBooNE nu_e spectra in terms of true energy instead of reconstructed energy"""
    PeeRW = []
    for k in range(len(MCT)):
        RWFact = PeeAvg(
            MuB_True_BinEdges[k], MuB_True_BinEdges[k + 1], LMBT, dm41, Ue4Sq
        )
        PeeRW.append(MCT[k] * RWFact)

    PCNuE = GBPC_NuE.miniToMicro(PeeRW)
    PCNuE = np.insert(PCNuE, 0, [0.0])
    PCNuE = np.append(PCNuE, 0.0)

    FCNuE = GBFC_NuE.miniToMicro(PeeRW)
    FCNuE = np.insert(FCNuE, 0, [0.0])
    FCNuE = np.append(FCNuE, 0.0)

    return [FCNuE, PCNuE]


def DecayMuBNuEDis(
    theta,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=True,
    energy_degradation=True,
    helicity="conserving",
):
    """Function for reweighting MicroBooNE nu_e spectra in terms of true energy instead of reconstructed energy"""

    # Load the Sterile class from param_scan
    sterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        helicity=helicity,
    )
    PeeRW = []
    # MCT is MiniBooNE truth level distribution from 2018. That's why it needs to be rescaled when unfolding
    for k in range(len(MCT)):
        PeeRW.append(MCT[k])
    if disappearance:
        # reset PeeRW
        PeeRW = []
        for k in range(len(MCT)):
            PeeRW.append(
                MCT[k]
                * sterile.PeeAvg(MuB_True_BinEdges[k], MuB_True_BinEdges[k + 1], LMBT)
            )
        if energy_degradation:
            PeeRW = sterile.EnergyDegradation(
                MCT,
                MuB_True_BinEdges,
                which_channel="Pee",
                which_experiment="microboone",
            )
        if not decay and oscillations:
            for k in range(len(MCT)):
                PeeRW[k] = MCT[k] * sterile.PeeoscAvg(
                    MuB_True_BinEdges[k], MuB_True_BinEdges[k + 1], LMBT
                )
    PeeRW2 = copy.deepcopy(PeeRW)
    PCNuE = GBPC_NuE.miniToMicro(PeeRW)
    PCNuE = np.insert(PCNuE, 0, [0.0])
    PCNuE = np.append(PCNuE, 0.0)

    FCNuE = GBFC_NuE.miniToMicro(PeeRW2)
    FCNuE = np.insert(FCNuE, 0, [0.0])
    FCNuE = np.append(FCNuE, 0.0)

    return [FCNuE, PCNuE]


def DecayMuBNuMuDis(
    theta,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=True,
    energy_degradation=True,
    helicity="conserving",
):
    """Function for reweighting MicroBooNE nu_mu spectra in terms of true energy instead of reconstructed energy"""

    # Load the Sterile class from param_scan
    sterile = Sterile(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        helicity=helicity,
    )
    PmmRW_FC = []
    PmmRW_PC = []
    for k in range(len(NuMuCC_TrueEDist_FC)):
        PmmRW_FC.append(NuMuCC_TrueEDist_FC[k])
        PmmRW_PC.append(NuMuCC_TrueEDist_PC[k])
    if disappearance:
        # reset PmmRW_FC and PmmRW_PC
        PmmRW_FC = []
        PmmRW_PC = []
        for k in range(len(NuMuCC_TrueEDist_FC)):
            PmmRW_FC.append(
                NuMuCC_TrueEDist_FC[k]
                * sterile.PmmAvg(MuB_BinEdges_NuMu[k], MuB_BinEdges_NuMu[k + 1], LMBT)
            )
            PmmRW_PC.append(
                NuMuCC_TrueEDist_PC[k]
                * sterile.PmmAvg(MuB_BinEdges_NuMu[k], MuB_BinEdges_NuMu[k + 1], LMBT)
            )
        if energy_degradation:
            PmmRW_FC = sterile.EnergyDegradation(
                NuMuCC_TrueEDist_FC,
                MuB_BinEdges_NuMu,
                which_channel="Pmm",
                which_experiment="microboone",
            )
            PmmRW_PC = sterile.EnergyDegradation(
                NuMuCC_TrueEDist_PC,
                MuB_BinEdges_NuMu,
                which_channel="Pmm",
                which_experiment="microboone",
            )
        if not decay and oscillations:
            for k in range(len(NuMuCC_TrueEDist_FC)):
                PmmRW_FC[k] = NuMuCC_TrueEDist_FC[k] * sterile.PmmoscAvg(
                    MuB_BinEdges_NuMu[k], MuB_BinEdges_NuMu[k + 1], LMBT
                )
                PmmRW_PC[k] = NuMuCC_TrueEDist_PC[k] * sterile.PmmoscAvg(
                    MuB_BinEdges_NuMu[k], MuB_BinEdges_NuMu[k + 1], LMBT
                )
    RecoDist_FC_0 = np.dot(NuMuCC_MigMat_FC, PmmRW_FC)
    RecoDist_PC_0 = np.dot(NuMuCC_MigMat_PC, PmmRW_PC)

    RecoDist_FC = []
    RecoDist_PC = []
    for j in range(25):
        RecoDist_FC.append(0.5 * (RecoDist_FC_0[2 * j] + RecoDist_FC_0[2 * j + 1]))
        RecoDist_PC.append(0.5 * (RecoDist_PC_0[2 * j] + RecoDist_PC_0[2 * j + 1]))
    RecoDist_FC.append(np.sum(RecoDist_FC_0[50:]))
    RecoDist_PC.append(np.sum(RecoDist_PC_0[50:]))

    FCEvts = [RecoDist_FC[kk] * NuMuCC_Eff_FC[kk] for kk in range(len(NuMuCC_Eff_FC))]
    PCEvts = [RecoDist_PC[kk] * NuMuCC_Eff_PC[kk] for kk in range(len(NuMuCC_Eff_PC))]

    return [FCEvts, PCEvts]
