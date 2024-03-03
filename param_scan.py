import numpy as np
import copy
from scipy import integrate
from scipy.special import expi
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
from importlib.resources import open_text
import numba

RHE = False
UFMB = True
GBPC = unfolder.MBtomuB(
    analysis="1eX_PC",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)
GBFC = unfolder.MBtomuB(
    analysis="1eX",
    remove_high_energy=RHE,
    unfold=UFMB,
    effNoUnfold=True,
    which_template="2020",
)

# Load the MiniBooNE MC from data release
MB_Ereco_unfold_bins = micro.bin_edges_reco
MB_Ereco_official_bins = micro.bin_edges * 1e-3
MB_Ereco_official_bins_numu = micro.bin_edges_numu * 1e-3
e_prod_e_int_bins = np.linspace(0, 3, 51)  # GeV


L_micro = 0.4685  # MicroBooNE Baseline length in kilometers
L_mini = 0.545  # MiniBooNE Baseline length in kilometers

Ereco, Etrue, Length, Weight = mini.apps.get_MC_from_data_release(mode='fhc', year='2020')
Enumu_reco, Enumu_true, Length_numu, Weight_numu = mini.apps.get_MC_from_data_release_numu(mode='fhc', year='2022')




"""
    Create a distribution of interaction energy for every production energy
    based on the energy distribution of the daughter neutrinos (eqn 2.3&2.4 in 1911.01447)

    e_prod: parent neutrino energy
    n_replications: number of interaction energy bins per production energy

"""
@numba.jit(nopython=True)
def replicate(x, n):
    return np.repeat(x, n)

@numba.jit(nopython=True)
def create_e_daughter(e_prod, n_replications=10):
    # e_prod: parent neutrino energy
    de = e_prod / n_replications
    e_daughter = np.linspace(de / 2, e_prod - de / 2, n_replications)
    return e_daughter

@numba.jit(nopython=True)
def create_Etrue_and_Weight_int(etrue, n_replications=10):
    # For every Etrue, create a list of possible daughter neutrino energy
    Etrue_daughter = np.empty((etrue.size, n_replications))
    for i in range(etrue.size):
        Etrue_daughter[i] = create_e_daughter(etrue[i], n_replications=n_replications)

    Etrue_extended = np.repeat(etrue, n_replications)

    return Etrue_extended, Etrue_daughter.flatten()

@numba.jit(nopython=True)
def numba_histogram(a, bin_edges, weights):
    ''' 
    Custom weighted histogram function from Numba's page
    https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
    '''
    hist = np.zeros((len(bin_edges)-1,), dtype=np.float64)

    for i, x in enumerate(a.flat):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += weights[i]

    return hist, bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    n = bin_edges.shape[0] - 1

    # Find the bin index using binary search
    left = 0
    right = n

    while left < right:
        mid = (left + right) // 2
        if x >= bin_edges[mid] and x < bin_edges[mid + 1]:
            return mid
        elif x < bin_edges[mid]:
            right = mid
        else:
            left = mid + 1

    return None

def fast_histogram(data, bins, weights):
    return numba_histogram(data, bins, weights)

def create_grid_of_params(g, m4, Ue4Sq, Um4Sq):
    paramlist_decay = np.array(np.meshgrid(g, m4, Ue4Sq, Um4Sq)).T.reshape(-1, 4)
    return [{"g": g, "m4": m4, "Ue4Sq": Ue4Sq, "Um4Sq": Um4Sq} for g, m4, Ue4Sq, Um4Sq in paramlist_decay]


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

        # Some derived quantities
        self.m4_in_GeV = self.m4 * 1e-9
        self.Us4Sq = 1 - self.Ue4Sq - self.Um4Sq  # Sterile mixing squared
        self.decouple_decay = decouple_decay
        self.oscillations = oscillations
        self.decay = decay

        # Some higher level variables
        self.Losc_0 = const.get_decay_rate_in_cm(1 / (2 * np.pi / (self.m4_in_GeV)) ) * 1e-5 if self.oscillations else np.inf
        self.Gamma_0 = self.GammaRestFrame()
        self.Ldec_0 = const.get_decay_rate_in_cm(self.Gamma_0) * 1e-5 if self.decay else 1e10
 
        # Vectorizing some class function
        self.PmmAvg_vec = np.vectorize(self.PmmAvg)
        self.PeeAvg_vec = np.vectorize(self.PeeAvg)
        self.PmmAvg_vec_deGouvea = np.vectorize(self.PmmAvg_deGouvea)

    def GammaRestFrame(self):
        """Decay rate in GeV, Etrue -- GeV"""
        if not self.decay:
            return 0
        else:
            if self.decouple_decay:
                return self.g**2  * self.m4_in_GeV / (16 * np.pi)
            else:
                return (
                    self.Us4Sq
                    * (1 - self.Us4Sq)
                    * (self.g ** 2 * self.m4_in_GeV)
                    / (16 * np.pi)
                )

    def GammaLab(self, E4):
        """Decay rate in GeV, Etrue -- GeV"""
        return (self.m4_in_GeV / E4) * self.Gamma_0

    def Ldec(self, E4):
        """Lab  frame decay length in km, E4 -- GeV"""
        return self.Ldec_0  / (self.m4_in_GeV / E4)
        
    def Losc(self, E4):
        """Oscillation length in km, E4 -- GeV"""
        return self.Losc_0 * (E4 / self.m4_in_GeV)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fosc(Length, Losc, Ldec):
        return (
            4
            * np.sin(np.pi / 2 * Length / Losc) ** 2
            * np.exp(-Length / Ldec / 2) + (1 - np.exp(-Length / Ldec / 2)) ** 2
        )
    
    def Fosc(self, E4, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fosc(Length, self.Losc(E4), self.Ldec(E4))
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fdec(Length, Ldec):
        return (1 - np.exp(-Length / Ldec))
    
    def Fdecay(self, E4, Edaughter, Length):
        """Decay probability function, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fdec(Length, self.Ldec(E4)) * Sterile.dPdecaydX(E4, Edaughter)

    def FdecayAvg(self, Emin, Emax, Length):
        """dPdecaydX --> 1"""
        integrand = lambda E4: (1 - np.exp(-Length / self.Ldec(E4)))
        return integrate.quad(integrand, Emin, Emax)[0] / (Emax - Emin)

    def FoscAvg(self, Emin, Emax, Length):
        integrand = lambda E4: self.Fosc(E4, Length)
        # return integrate.quad(integrand, Emin, Emax, )[0] / (Emax - Emin)
        x = np.linspace(Emin+1e-6, Emax, 1000)
        return np.sum(self.Fosc(x, Length)*(x[1]-x[0]))/(Emax - Emin)

    def FoscAna(self, Emin, Emax, Length):
        """here we evaluate the integral analytically"""
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        a = Length * self.m4**2 * 1e3 / (4 * 197.3269804)
        b = (
            Length
            * self.Us4Sq
            * (1 - self.Us4Sq)
            * (self.g * self.m4_in_GeV) ** 2
            / (32 * np.pi)
        ) / (1e-5 * 197.3269804e-16)
        z = (
            1
            / (Emax - Emin)
            * (
                (
                    Emax
                    + np.exp(-(2 * b) / Emax) * Emax
                    - 2 * np.exp(-b / Emax) * Emax * np.cos((2 * a) / Emax)
                    + (2j * a - b) * expi((2j * a - b) / Emax)
                    + 2 * b * expi(-(2 * b) / Emax)
                    - (2j * a + b) * expi(-(2j * a + b) / Emax)
                )
                - (
                    (
                        Emin
                        + np.exp(-(2 * b) / Emin) * Emin
                        - 2 * np.exp(-b / Emin) * Emin * np.cos((2 * a) / Emin)
                        + (2j * a - b) * expi((2j * a - b) / Emin)
                        + 2 * b * expi(-(2 * b) / Emin)
                        - (2j * a + b) * expi(-(2j * a + b) / Emin)
                    )
                )
            )
        )
        return z.real

    def FdecayAna(self, Emin, Emax, Length):
        """here we evaluate the integral analytically"""
        cst = (
            Length
            * self.Us4Sq
            * (1 - self.Us4Sq)
            * (self.g * self.m4_in_GeV) ** 2
            / (16 * np.pi)
        ) / (1e-5 * 197.3269804e-16)
        return (
            1
            / (Emax - Emin)
            * (
                (Emax - np.exp(-cst / Emax) * Emax - cst * expi(-cst / Emax))
                - (Emin - np.exp(-cst / Emin) * Emin - cst * expi(-cst / Emin))
            )
        )

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

    def Pme_old(self, E4, Length):
        """The original appearance probability"""
        return (
            4 * self.Um4Sq * self.Ue4Sq * np.sin(1.267 * self.m4**2 * Length / E4) ** 2
        )

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
    
    # ----------------------------------------------------------------
    # de Gouvea's model
    # ----------------------------------------------------------------
    def Pme_deGouvea(self, E4, Edaughter, Length):
        return self.Um4Sq * (1 - np.exp(-Length / (2*self.Ldec(E4))) ) * Sterile.dPdecaydX(E4, Edaughter)
    
    def PmmAvg_deGouvea(self, Emin, Emax, Length):
        integrand = lambda E4: (1 - self.Um4Sq)**2 + self.Um4Sq**2 * np.exp(-Length / (2*self.Ldec(E4)))
        return integrate.quad(integrand, Emin, Emax)[0] / (Emax - Emin)

    # ----------------------------------------------------------------
    # DECAY AND OSC PROBABILITIES IN DISAPPEARANCE ENERGY DEGRADATION
    # ----------------------------------------------------------------
    def Pmmdecay(self, Emin, Emax, Eintmin, Eintmax, Length, noffset=0):
        # decay term in Pmm, Emin and Emax are E4 bin edges
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        if Emax < 1:
            n = 2 + noffset
        else:
            n = 1 + noffset
        pdecay = (
            self.Um4Sq
            * self.FdecayAna(Emin, Emax, Length)
            * ((Eintmax**2 - Eintmin**2) / (Emax * Emin))
            * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
        )
        # ((Eintmax**2 - Eintmin**2)/(Emax*Emin)) factor is to account for the decay rate scaling with Eint/E4 -- gives the fraction of
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmmosc(self, Emin, Emax, Length):
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        # osc term in Pmm, does not involve energy degradation
        return 1 - self.Um4Sq * (1 - self.Um4Sq) * self.FoscAna(Emin, Emax, Length)

    def Peedecay(self, Emin, Emax, Eintmin, Eintmax, Length, noffset=0):
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        # decay term in Pee, Emin and Emax are E4 bin edges
        if Emax < 1:
            n = 2 + noffset
        else:
            n = 1 + noffset
        pdecay = (
            self.Ue4Sq
            * self.FdecayAna(Emin, Emax, Length)
            * ((Eintmax**2 - Eintmin**2) / (Emax * Emin))
            * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
        )
        # pdecay = self.Ue4Sq * self.FdecayAvg(Emin, Emax, Length) * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
        # ((Eintmax**2 - Eintmin**2)/(Emax*Emin)) factor is to account for the decay rate scaling with Eint/E4 -- gives the fraction of
        # events in this bin
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Peeosc(self, Emin, Emax, Length):
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        # osc term in Pee, does not involve energy degradation
        return 1 - self.Ue4Sq * (1 - self.Ue4Sq) * self.FoscAna(Emin, Emax, Length)

    # ----------------------------------------------------------------
    # DISAPPEARANCE PROBABILITIES WITHOUT ENERGY DEGRADATION
    # ----------------------------------------------------------------
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

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def dPdecaydX(Eparent, Edaughter):
        """The probability of daughter neutrino energy"""

        decay_w_base = Edaughter / Eparent

        return decay_w_base

    # def Pdecay_binned_avg(self, E4_bin_edges, fixed_Length=L_micro):
    #     """E4_bin_edges -- array in GeV, Length -- Kilometers"""

    #     # NOTE: I guess we also have to update this to include oscillations etc.
    #     # My impression is that there's probably an easier way to use the same functions above to calculate the average,
    #     # instead of rewrite the formulae already integrated. Maybe enough to do a quad by hand?

    #     de = np.diff(E4_bin_edges)
    #     el = E4_bin_edges[:-1]

    #     # # NOTE: We should check our fits are independent of this choice!!
    #     el[el == 0] = 1e-3  # 1 MeV regulator
    #     er = E4_bin_edges[1:]

    #     # exponential argument
    #     x = -1.267 * (4 * self.GammaLab(1) * fixed_Length)

    #     return (
    #         1
    #         / de
    #         * (
    #             (er * np.exp(x / er) - x * expi(x / er))
    #             - (el * np.exp(x / el) - x * expi(x / el))
    #         )
    #     )

    def EnergyDegradation(self, Etrue_dist, Etrue_bins, which_channel):
        R_deg = np.zeros((len(Etrue_dist), len(Etrue_dist)))
        R_osc = []
        # degradation piece
        for k in range(len(Etrue_dist)):
            for i in range(k + 1):
                Pdecay = 1
                if which_channel == "Pee":
                    Pdecay = self.Peedecay(
                        Etrue_bins[k],
                        Etrue_bins[k + 1],
                        Etrue_bins[i],
                        Etrue_bins[i + 1],
                        L_micro,
                        noffset=0,
                    )
                elif which_channel == "Pmm":
                    Pdecay = self.Pmmdecay(
                        Etrue_bins[k],
                        Etrue_bins[k + 1],
                        Etrue_bins[i],
                        Etrue_bins[i + 1],
                        L_micro,
                        noffset=0,
                    )
                R_deg[k][i] = Pdecay * Etrue_dist[i]
        R_sum = np.sum(R_deg, axis=0)

        # oscillation piece
        for i in range(len(Etrue_dist)):
            Peeosc = self.Peeosc(Etrue_bins[i], Etrue_bins[i + 1], L_micro)
            Pmmosc = self.Pmmosc(Etrue_bins[i], Etrue_bins[i + 1], L_micro)
            if which_channel == "Pee":
                R_osc.append(Peeosc * Etrue_dist[i])
            if which_channel == "Pmm":
                R_osc.append(Pmmosc * Etrue_dist[i])

        R_tot = R_sum + R_osc

        return R_tot

def MiniBooNEChi2_deGouvea(theta, oscillations=False, decay=True, decouple_decay=True, n_replications=10):
    """
    Returns the MicroBooNE chi2 for deGouvea's model
    """

    g = theta["g"]
    m4 = theta["m4"]
    Ue4Sq = theta["Ue4Sq"]
    Um4Sq = theta["Um4Sq"]

    sterile = Sterile(theta, oscillations=oscillations, decay=decay, decouple_decay=decouple_decay)

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int(etrue=Etrue, n_replications=n_replications)

    # replicating entries of the MC data release -- baseline L and weight
    Length_ext = replicate(Length, n=n_replications)
    Weight_ext = replicate(Weight / n_replications, n = n_replications)

    # Flavor transition probabilities -- Assuming nu4 decays only into nue
    Pme = sterile.Pme_deGouvea(Etrue_parent, Etrue_daughter, Length_ext)

    Weight_decay = Weight_ext * Pme

    # Calculate the MiniBooNE chi2
    MBSig_for_MBfit = np.dot(
        fast_histogram(Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_decay)[0],
        mini.apps.migration_matrix_official_bins_nue_11bins,
    )

    # Average disappearance in each bin of MB MC data release
    #P_avg = sterile.Pdecay_binned_avg(MB_Ereco_official_bins_numu, fixed_Length=L_micro)
    #P_mumu_avg = (1 - Um4Sq) ** 2 + Um4Sq**2 * P_avg

    #MB_chi2 = mini.fit.chi2_MiniBooNE_2020(MBSig_for_MBfit, Pmumu=P_mumu_avg, Pee=1)
    P_mumu_avg_deGouvea = sterile.PmmAvg_vec_deGouvea(
            MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_micro
    )
    MC_numu_bkg_total_w_dis_deGouvea = mini.MC_numu_bkg_tot * P_mumu_avg_deGouvea

    # Calculate MiniBooNE chi2
    MB_chi2 = mini.fit.chi2_MiniBooNE(
        MC_nue_app=MBSig_for_MBfit,
        MC_nue_dis=None,
        MC_numu_dis=None,
        year="2018",
    )

    return [g, m4, Ue4Sq, Um4Sq, MB_chi2]
# --------------------------------------------------------------------------------
def DecayReturnMicroBooNEChi2(
    theta,
    oscillations=True,
    decay=False,
    decouple_decay=False,
    disappearance=False,
    energy_degradation=False,
    use_numu_MC=False,
    n_replications=10,

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

    # Our new physics class -- for deGouvea's model, we fix m4 = 1 eV, and identify g = gm4.
    sterile = Sterile(
        theta, oscillations=oscillations, decay=decay, decouple_decay=decouple_decay
    )

    # Replicating events for multiple daughter neutrino energies
    Etrue_parent, Etrue_daughter = create_Etrue_and_Weight_int(etrue=Etrue, n_replications=n_replications)
    
    Ereco_ext = replicate(Ereco, n=n_replications) 
    Length_ext = replicate(Length, n=n_replications)
    Weight_ext = replicate(Weight / n_replications, n=n_replications)
    

    '''
        re-normalizing the muon flux
        the MC comes from the MiniBooNE prediction which 
        is informed by their nu_mu CC data
    '''
    
    # Flavor transition probabilities
    Pme = sterile.Pme(Etrue_parent, Etrue_daughter, Length_ext)
    Weight_nue_app = Weight_ext * Pme

    # Calculate the MiniBooNE chi2
    if not decay and oscillations:
        # NOTE: Using Ereco from MC for oscillation-only
        MC_nue_app = fast_histogram(
            Ereco_ext,
            weights=Weight_nue_app,
            bins=MB_Ereco_official_bins,
        )[0]
    else:
        # Migrate nue signal from Etrue to Ereco with 11 bins
        MC_nue_app = np.dot(
            fast_histogram(
                Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_app
            )[0],
            mini.apps.migration_matrix_official_bins_nue_11bins,
        )

    # Average disappearance in each bin of MB MC data release
    if disappearance:
        Weight_nue_flux = mini.apps.reweight_MC_to_nue_flux(Etrue_parent, Weight_ext)
        Weight_nue_dis = Weight_nue_flux * sterile.Pee(
            Etrue_parent, Etrue_daughter, Length_ext
        )
        if (not decay) and oscillations:
            MC_nue_bkg_intrinsic = fast_histogram(
                Ereco_ext,
                weights=Weight_nue_flux,
                bins=MB_Ereco_official_bins,
            )[0]
            MC_nue_bkg_intrinsic_osc = fast_histogram(
                Ereco_ext,
                weights=Weight_nue_dis,
                bins=MB_Ereco_official_bins,
            )[0]
        else:
            # Migrate nue signal from Etrue to Ereco with 11 bins
            MC_nue_bkg_intrinsic = np.dot(
                np.MC_nue_bkg(
                    Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_flux
                )[0],
                migration_matrix_official_bins,
            )
            MC_nue_bkg_intrinsic_osc = np.dot(
                fast_histogram(
                    Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_nue_dis
                )[0],
                migration_matrix_official_bins,
            )

        # Final MC prediction for nu_e sample (w/ oscillated intrinsics)
        MC_nue_bkg_total_w_dis = (
            mini.MC_nue_bkg_tot - MC_nue_bkg_intrinsic + MC_nue_bkg_intrinsic_osc
        )

        # NUMU DISAPPEARANCE
        if use_numu_MC:
            Enumu_true_parent, Enumu_true_daughter = create_Etrue_and_Weight_int(etrue=Enumu_true, n_replications=n_replications)
            Enumu_reco_ext = replicate(Enumu_reco, n=n_replications)
            Length_numu_ext = replicate(Length_numu, n=n_replications)
            Weight_numu_ext = replicate(Weight_numu / n_replications, n=n_replications)


            Weight_numu_dis = Weight_numu_ext * sterile.Pmm(
            Enumu_true_parent, Enumu_true_daughter, Length_numu_ext
        )
            MC_numu_bkg_total_w_dis = fast_histogram(
                Enumu_reco_ext,
                weights=Weight_numu_dis,
                bins=MB_Ereco_official_bins_numu,
            )[0]

            Weight_numu_dis = Weight_numu_ext * sterile.Pmm(
            Enumu_true_parent, Enumu_true_daughter, Length_numu_ext
        )

            # # Migrate nue signal from Etrue to Ereco with 11 bins
            # MC_numu_bkg_total_w_dis = np.dot(
            # fast_histogram(
            #     Etrue_daughter, bins=e_prod_e_int_bins, weights=Weight_numu_dis
            # )[0],
            # mini.apps.migration_matrix_official_bins_nue_11bins,
            # )

        else: 
            # NOTE: Averaged
            # Final MC prediction for nu_mu sample (w/ oscillated numus)
            P_mumu_avg = sterile.PmmAvg_vec(
                MB_Ereco_official_bins_numu[:-1], MB_Ereco_official_bins_numu[1:], L_mini
            )
            MC_numu_bkg_total_w_dis = mini.MC_numu_bkg_tot * P_mumu_avg

        # Calculate MiniBooNE chi2
        MB_chi2 = mini.fit.chi2_MiniBooNE(
            MC_nue_app=MC_nue_app,
            MC_nue_dis=MC_nue_bkg_total_w_dis,
            MC_numu_dis=MC_numu_bkg_total_w_dis,
            year="2020",
        )

    else:
        MB_chi2 = mini.fit.chi2_MiniBooNE(MC_nue_app, year="2018")

    # NOTE: SKIPPING ENERGY DEGRATION FOR NOW
    # if energy_degradation:
    #     # MiniBooNE energy degradation
    #     # Questionable, MC file is meant for Pme channel. Not sure if it can be used for numu and nue disappearance.
    #     Ree_true = sterile.EnergyDegradation(
    #         fast_histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
    #         e_prod_e_int_bins,
    #         "Pee",
    #     )
    #     Rmm_true = sterile.EnergyDegradation(
    #         fast_histogram(Etrue, bins=e_prod_e_int_bins, weights=Weight)[0],
    #         e_prod_e_int_bins,
    #         "Pmm",
    #     )
    #     migration_matrix_pee = create_reco_migration_matrix(nue_bin_edges)
    #     migration_matrix_pmm = create_reco_migration_matrix(numu_bin_edges)
    #     Ree_reco = np.dot(Ree_true, migration_matrix_pee)
    #     Rmm_reco = np.dot(Rmm_true, migration_matrix_pmm)
    #     MB_chi2 = mini.fit.chi2_MiniBooNE_2020(
    #         MBSig_for_MBfit, Rmumu=Rmm_reco, Ree=Ree_reco
    #     )

    # Calculate the MicroBooNE chi2 by unfolding
    # MBSig_for_unfolding = np.dot(
    #     (fast_histogram(Etrue_parent, bins=e_prod_e_int_bins, weights=Weight_decay)[0]),
    #     migration_matrix_unfolding_bins,
    # )
    MBSig_for_unfolding = fast_histogram(Ereco_ext, weights=Weight_nue_app, bins=MB_Ereco_official_bins)[0]
    MBSig_for_unfolding2 = copy.deepcopy(MBSig_for_unfolding)
    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig_for_unfolding)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig_for_unfolding2)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
    )
    # \nu_e disappearance signal replacement
    NuEReps = DecayMuBNuEDis(
        theta,
        oscillations=oscillations,
        decay=decay,
        decouple_decay=decouple_decay,
        disappearance=disappearance,
        energy_degradation=energy_degradation,
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
        disappearance=disappearance,
        energy_degradation=energy_degradation,
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
        disappearance=disappearance,
        energy_degradation=energy_degradation,
    )

    return [g, m4, Ue4Sq, Um4Sq, MB_chi2, MuB_chi2, MuB_chi2_Asimov]