import numpy as np
import numba
from scipy import integrate
from scipy.special import expi
from . import const
from . import L_micro, L_mini


from pathlib import Path

local_dir = Path(__file__).parent


# --------------------------------------------------------------------------------
class Sterile:
    def __init__(
        self,
        theta,
        oscillations=True,
        decay=True,
        decouple_decay=False,
        CP=1,
        helicity="conserving",
    ):
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

        CP: int, optional
            Whether to consider neutrino or antineutrino, by default +1
            CP = +1 neutrinos
            CP = -1 antineutrinos
            NOTE: So far, makes no difference

        helicity: str, optional
            conserving: Ed/Ep
            flipping: 1 - Ed/Ep

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
        self.Losc_0 = (
            const.get_decay_rate_in_cm(1 / (2 * np.pi / (self.m4_in_GeV))) * 1e-5
            if self.oscillations
            else np.inf
        )
        self.Gamma_0 = self.GammaRestFrame()
        self.Ldec_0 = (
            const.get_decay_rate_in_cm(self.Gamma_0) * 1e-5 if self.decay else 1e10
        )

        # Vectorizing some class function
        self.PmmAvg_vec_deGouvea = np.vectorize(self.PmmAvg_deGouvea)

        # Load MiniBooNE detector efficiency data
        self.pathdata = "MiniTools/include/miniboone_eff/eg_effs.dat"
        self.eff_data = np.loadtxt(self.pathdata)
        self.eff_bin_edges = self.eff_data[:, 0] / 1000  # GeV
        self.eff = self.eff_data[:, 1]

        self.CP = CP
        self.helicity = helicity
        if self.helicity == "conserving":
            self.dPdecaydX = Sterile.dPdecaydX_conserving
            self.dPdecaydX_Avg = Sterile.dPdecaydX_Avg_conserving
        elif self.helicity == "flipping":
            self.dPdecaydX = Sterile.dPdecaydX_flipping
            self.dPdecaydX_Avg = Sterile.dPdecaydX_Avg_flipping
        else:
            raise ValueError('helicity must be "conserving" or "flipping"')

    def GammaRestFrame(self):
        """Decay rate in GeV, Etrue -- GeV"""
        if not self.decay:
            return 0
        else:
            if self.decouple_decay:
                return self.g**2 * self.m4_in_GeV / (16 * np.pi)
            else:
                return (
                    self.Us4Sq
                    * (1 - self.Us4Sq)
                    * (self.g**2 * self.m4_in_GeV)
                    / (16 * np.pi)
                )

    def GammaLab(self, E4):
        """Decay rate in GeV, Etrue -- GeV"""
        return (self.m4_in_GeV / E4) * self.Gamma_0

    def Ldec(self, E4):
        """Lab  frame decay length in km, E4 -- GeV"""
        return self.Ldec_0 / (self.m4_in_GeV / E4)

    def Losc(self, E4):
        """Oscillation length in km, E4 -- GeV"""
        return self.Losc_0 * (E4 / self.m4_in_GeV)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fosc_crossterm(Length, Losc, Ldec):
        return 1 - np.exp(-Length / Ldec / 2) * np.cos(np.pi * Length / Losc)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fosc(Length, Losc, Ldec):
        return (
            1
            - 2 * np.exp(-Length / Ldec / 2) * np.cos(np.pi * Length / Losc)
            + np.exp(-Length / Ldec)
        )

    def Fosc_crossterm(self, E4, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fosc_crossterm(Length, self.Losc(E4), self.Ldec(E4))

    def Fosc(self, E4, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fosc(Length, self.Losc(E4), self.Ldec(E4))

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fdec(Length, Ldec):
        return 1 - np.exp(-Length / Ldec)

    def Fdecay(self, E4, Edaughter, Length):
        """Decay probability function, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fdec(Length, self.Ldec(E4)) * self.dPdecaydX(E4, Edaughter)

    def FdecayAvg(self, Emin, Emax, Length):
        """dPdecaydX --> 1"""
        x = np.linspace(Emin, Emax, 100)
        dx = x[1] - x[0]
        if len(Length) > 1:
            xx, ll = np.meshgrid(x, Length)
            return np.sum(self.Fdecay(xx, ll), axis=1) * dx / (Emax - Emin)
        else:
            return np.sum(self.Fdecay(x, Length) * (x[1] - x[0])) / (Emax - Emin)

    def FoscAvg_numerical(self, Emin, Emax, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        x = np.linspace(Emin, Emax, 100, endpoint=True)
        dx = x[1] - x[0]
        if len(Length) > 1:
            xx, ll = np.meshgrid(x, Length)
            return np.sum(self.Fosc(xx, ll), axis=1) * dx / (Emax - Emin)
        else:
            return np.sum(self.Fosc(x, Length) * (x[1] - x[0])) / (Emax - Emin)

    def FoscAvg_crossterm_numerical(self, Emin, Emax, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        x = np.linspace(Emin, Emax, 100, endpoint=True)
        dx = x[1] - x[0]
        if len(Length) > 1:
            xx, ll = np.meshgrid(x, Length)
            return np.sum(self.Fosc_crossterm(xx, ll), axis=1) * dx / (Emax - Emin)
        else:
            return np.sum(self.Fosc_crossterm(x, Length) * (x[1] - x[0])) / (
                Emax - Emin
            )

    def FoscAvg_crossterm_analytical(self, Emin, Emax, Length):
        """here we evaluate the integral analytically"""

        # NOTE: This is actually slower than the numerical one because of the scipy special func expi

        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001

        a = Length / self.Ldec_0 * self.m4_in_GeV
        b = np.pi * Length / self.Losc_0 * self.m4_in_GeV

        res = (Emax - Emin) * (1 + 0 * 1j)
        res += np.exp(-a / Emin / 2) * Emin * np.cos(b / Emin)
        res += -np.exp(-a / Emax / 2) * Emax * np.cos(b / Emax)
        res += 0.25 * (
            -(a - b * 2j) * expi(-((a - b * 2j) / Emax / 2))
            - (a + b * 2j) * expi(-((a + b * 2j) / Emax / 2))
            + (a - b * 2j) * expi(-((a - b * 2j) / Emin / 2))
            + (a + b * 2j) * expi(-((a + b * 2j) / Emin / 2))
        )
        return res.real / (Emax - Emin)

    def FoscAvg_analytical(self, Emin, Emax, Length):
        """here we evaluate the integral analytically"""
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001

        a = Length / self.Ldec_0 * self.m4_in_GeV
        b = np.pi * Length / self.Losc_0 * self.m4_in_GeV

        res = (Emax - Emin) * (1 + 0 * 1j)
        res += np.exp(-a / Emax) * Emax - np.exp(-a / Emin) * Emin
        res += a * expi(-a / Emax) - a * expi(-a / Emin)
        res += 0.5 * (
            -4 * np.exp(-a / (2 * Emax)) * Emax * np.cos(b / Emax)
            + 4 * np.exp(-a / (2 * Emin)) * Emin * np.cos(b / Emin)
            - (a - 2j * b) * expi(-((a - 2j * b) / (2 * Emax)))
            - (a + 2j * b) * expi(-((a + 2j * b) / (2 * Emax)))
            + (a - 2j * b) * expi(-((a - 2j * b) / (2 * Emin)))
            + (a + 2j * b) * expi(-((a + 2j * b) / (2 * Emin)))
        )

        return res.real / (Emax - Emin)

    def FdecAvg_analytical(self, E4min, E4max, Length):
        """Average of the decay function in a given E4 bin"""
        if E4min == 0.0:
            E4min = 0.000001
        if E4max == 0.0:
            E4max = 0.000001

        a = Length / self.Ldec_0 * self.m4_in_GeV

        res = E4max - E4min
        res -= E4max * np.exp(-a / E4max) - E4min * np.exp(-a / E4min)
        res -= a * (expi(-a / E4max) - expi(-a / E4min))
        return res.real / (E4max - E4min)

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

    def Pmedecay(self, E4, Edaughter, Length, exp="miniboone"):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Um4Sq
            * self.Fdecay(E4, Edaughter, Length)
            * DegradationCorrection(Edaughter, E4, exp)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmeosc(self, E4, Length):
        # Oscillation term
        posc = self.Um4Sq * self.Ue4Sq * self.Fosc(E4, Length)
        return posc

    def Peedecay(self, E4, Edaughter, Length, exp="miniboone"):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Ue4Sq
            * self.Fdecay(E4, Edaughter, Length)
            * DegradationCorrection(Edaughter, E4, exp)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Peeosc(self, E4, Length):
        # Oscillation term
        return (
            1
            - 2 * self.Ue4Sq * self.Fosc_crossterm(E4, Length)
            + self.Ue4Sq**2 * self.Fosc(E4, Length)
        )

    def Pmmdecay(self, E4, Edaughter, Length, exp="miniboone"):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Um4Sq
            * self.Fdecay(E4, Edaughter, Length)
            * DegradationCorrection(Edaughter, E4, exp)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmmosc(self, E4, Length):
        # Oscillation term
        return (
            1
            - 2 * self.Um4Sq * self.Fosc_crossterm(E4, Length)
            + self.Um4Sq**2 * self.Fosc(E4, Length)
        )

    # ----------------------------------------------------------------
    # de Gouvea's model
    # ----------------------------------------------------------------
    def Pme_deGouvea(self, E4, Edaughter, Length):
        return (
            self.Um4Sq
            * (1 - np.exp(-Length / (2 * self.Ldec(E4))))
            * self.dPdecaydX(E4, Edaughter)
        )

    def PmmAvg_deGouvea(self, Emin, Emax, Length):
        return integrate.quad(
            lambda E4: (1 - self.Um4Sq) ** 2
            + self.Um4Sq**2 * np.exp(-Length / (2 * self.Ldec(E4))),
            Emin,
            Emax,
        )[0] / (Emax - Emin)

    # ----------------------------------------------------------------
    # DECAY AND OSC PROBABILITIES IN DISAPPEARANCE ENERGY DEGRADATION
    # ----------------------------------------------------------------
    def PmmdecayAvg(
        self, Ebins, e4_index, eint_index, Length, which_experiment, noffset=0
    ):
        Eintmin, Eintmax = Ebins[eint_index], Ebins[eint_index + 1]
        Emin, Emax = Ebins[e4_index], Ebins[e4_index + 1]
        E0 = Ebins[0]
        # decay term in Pee, Emin and Emax are E4 bin edges
        pdecay = (
            self.Um4Sq
            * self.FdecAvg_analytical(Emin, Emax, Length)
            * self.dPdecaydX_Avg(E0, Emin, Emax, Eintmin, Eintmax)
            * DegradationCorrection(
                (Eintmin + Eintmax) / 2,
                (Emin + Emax) / 2,
                which_experiment,
                noffset=noffset,
            )
        )
        # ((Eintmax**2 - Eintmin**2)/(Emax*Emin)) factor is to account for the decay rate scaling with Eint/E4
        # It gives the fraction of events in this bin
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def PmmoscAvg(self, Emin, Emax, Length):
        # osc term in Pmm, does not involve energy degradation
        return (
            1
            - 2 * self.Um4Sq * self.FoscAvg_crossterm_analytical(Emin, Emax, Length)
            + self.Um4Sq**2 * self.FoscAvg_analytical(Emin, Emax, Length)
        )

    def PeedecayAvg(
        self, Ebins, e4_index, eint_index, Length, which_experiment, noffset=0
    ):
        Eintmin, Eintmax = Ebins[eint_index], Ebins[eint_index + 1]
        Emin, Emax = Ebins[e4_index], Ebins[e4_index + 1]
        E0 = Ebins[0]
        # decay term in Pee, Emin and Emax are E4 bin edges
        pdecay = (
            self.Ue4Sq
            * self.FdecAvg_analytical(Emin, Emax, Length)
            * self.dPdecaydX_Avg(E0, Emin, Emax, Eintmin, Eintmax)
            * DegradationCorrection(
                (Eintmin + Eintmax) / 2,
                (Emin + Emax) / 2,
                which_experiment,
                noffset=noffset,
            )
        )
        # It gives the fraction of events in this bin
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def PeeoscAvg(self, Emin, Emax, Length):
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001

        # osc term in Pee, does not involve energy degradation
        return (
            1
            - 2 * self.Ue4Sq * self.FoscAvg_crossterm_analytical(Emin, Emax, Length)
            + self.Ue4Sq**2 * self.FoscAvg_analytical(Emin, Emax, Length)
        )

    # ----------------------------------------------------------------
    # DISAPPEARANCE PROBABILITIES WITHOUT ENERGY DEGRADATION
    # ----------------------------------------------------------------
    def PmmAvg(self, Emin, Emax, Length):
        """
        Averaged Disappearance probability, E4 -- GeV, Length -- km
        E4 and Edaughter are approximated to be equal, since the discrepancy is suppressed by mixing squared
        """
        # Decay term
        pdecay = self.Um4Sq * self.FdecAvg_analytical(Emin, Emax, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.PmmoscAvg(Emin, Emax, Length)
        return pdecay + posc

    def PeeAvg(self, Emin, Emax, Length):
        """
        Averaged Disappearance probability, E4 -- GeV, Length -- km
        E4 and Edaughter are approximated to be equal, since the discrepancy is suppressed by mixing squared
        """
        # Decay term
        pdecay = self.Ue4Sq * self.FdecAvg_analytical(Emin, Emax, Length)
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)

        # Oscillation term
        posc = self.PeeoscAvg(Emin, Emax, Length)
        return pdecay + posc

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def dPdecaydX_flipping(Eparent, Edaughter):
        """The probability of daughter neutrino energy

        1/Gamma * (dGamma/dEdaughter) = 2 * (1 - dEdaughter/dEparent)

        NOTE: factor of 2 is to ensure the above is normalized to 1.
        """
        return 2 * (1 - Edaughter / Eparent)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def dPdecaydX_conserving(Eparent, Edaughter):
        """The probability of daughter neutrino energy

        1/Gamma * (dGamma/dEdaughter) = 2 * (dEdaughter/dEparent)

        NOTE: factor of 2 is to ensure the above is normalized to 1.
        """
        return 2 * (Edaughter / Eparent)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def dPdecaydX_Avg_conserving(E0, Emin, Emax, Eintmin, Eintmax):
        if Emin == E0 or Emax == E0:
            return 1
        else:
            return (Eintmax**2 - Eintmin**2) / ((Emax - E0) * (Emin + E0))

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def dPdecaydX_Avg_flipping(E0, Emin, Emax, Eintmin, Eintmax):
        if Emin == E0 or Emax == E0:
            return 1
        else:
            return (
                ((Emax + Emin) - (Eintmax + Eintmin))
                * (Eintmax - Eintmin)
                / (Emax - E0)
                / (Emin - E0)
            )

    def EnergyDegradation(
        self, R_in_Enutrue, Etrue_bins_edge, which_channel, which_experiment
    ):
        n_bins = len(R_in_Enutrue)
        R_deg = np.zeros((n_bins, n_bins))
        R_osc = []
        L_avg = L_micro if which_experiment == "microboone" else L_mini

        # degradation piece
        for k in range(n_bins):
            for i in range(k + 1):

                if which_channel == "Pee":
                    Pdecay = self.PeedecayAvg(
                        Etrue_bins_edge, k, i, L_avg, which_experiment, noffset=0
                    )
                elif which_channel == "Pmm":
                    Pdecay = self.PmmdecayAvg(
                        Etrue_bins_edge, k, i, L_avg, which_experiment, noffset=0
                    )
                else:
                    raise ValueError(
                        f"Channel {which_channel} not recognzied. Valid options: 'Pmm' and 'Pee'."
                    )

                R_deg[k][i] = (
                    Pdecay * R_in_Enutrue[k]
                )  # k indexes parent energy, i indexes daughter energy

        R_sum = np.sum(R_deg, axis=0)

        # oscillation piece
        for i in range(n_bins):
            if which_channel == "Pee":
                Peeosc = self.PeeoscAvg(
                    Etrue_bins_edge[i], Etrue_bins_edge[i + 1], L_avg
                )
                R_osc.append(Peeosc * R_in_Enutrue[i])
            elif which_channel == "Pmm":
                Pmmosc = self.PmmoscAvg(
                    Etrue_bins_edge[i], Etrue_bins_edge[i + 1], L_avg
                )
                R_osc.append(Pmmosc * R_in_Enutrue[i])
            else:
                raise ValueError(
                    f"Channel {which_channel} not recognzied. Valid options: 'Pmm' and 'Pee'."
                )

        R_tot = R_sum + R_osc

        return R_tot

    """
    def MiniEff(self, E):
        if E < 0.15:
            return 0.00001
        if E > 2.0:
            return 0.026
        mask = numba_histogram(a = np.array([E]), bin_edges=np.array(self.eff_bin_edges),weights=[1])[0]
        pos = np.nonzero(mask)[0]
        return self.eff[pos[0]]
    
    
    def MiniEffApp(self, E):
        "Here E is an array, Eparent or Edaughter"
        # Initialize the efficiency array
        eff = [1]*len(E)
        for i in range(len(E)):
            if E[i] < 0.15:
                eff[i] = 0.00001
            elif E[i] > 2.0:
                eff[i] = 0.026
            else:
                #mask = param_scan.numba_histogram(a = np.array([E[i]]), bin_edges=np.array(self.eff_bin_edges), weights=[1])[0]
                mask = np.histogram(E[i], bins=self.eff_bin_edges)[0]
                pos = np.nonzero(mask)[0]
                eff[i] = self.eff[pos[0]]
        return eff
    """


def MiniEff(x):
    conditions = (
        [x < 0.15]
        + [(x >= 0.15 + 0.1 * i) & (x < 0.25 + 0.1 * i) for i in range(9)]
        + [(x >= 1.05) & (x < 1.2)]
        + [(x >= 1.2 + 0.2 * j) & (x < 1.4 + 0.2 * j) for j in range(4)]
        + [x >= 2.0]
    )
    functions = [
        0.00001,
        0.089,
        0.135,
        0.139,
        0.131,
        0.123,
        0.116,
        0.106,
        0.102,
        0.095,
        0.089,
        0.082,
        0.073,
        0.067,
        0.052,
        0.026,
    ]
    return np.piecewise(x, conditions, functions)


f_sigma = np.load(
    local_dir.joinpath("InclusiveTools/f_sigma.npy"),
    allow_pickle=True,
).item()


def Xsec(E):
    """Cross section in cm^2, E -- GeV"""
    return f_sigma(E)


def DegradationCorrection(Edaughter, E4, exp, noffset=0):

    if exp == "miniboone":
        return Xsec(Edaughter) / Xsec(E4) * MiniEff(Edaughter) / MiniEff(E4)
    elif exp == "microboone":

        if Edaughter < 1:
            n = 1 + noffset
        else:
            n = 0 + noffset

        xsec_nu4 = Xsec(E4)
        return (
            np.where(xsec_nu4 > 0, Xsec(Edaughter) / xsec_nu4, 0)
            * (Edaughter / E4) ** n
        )
    else:
        raise ValueError(f"Experiment {exp} not recognized.")
