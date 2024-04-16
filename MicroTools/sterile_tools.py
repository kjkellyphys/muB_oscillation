import numpy as np
import numba
from scipy import integrate
from scipy.special import expi
from . import const
from . import L_micro, L_mini


# --------------------------------------------------------------------------------
class Sterile:
    def __init__(
        self, theta, oscillations=True, decay=True, decouple_decay=False, CP=1
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
        self.PmmAvg_vec = np.vectorize(self.PmmAvg)
        self.PeeAvg_vec = np.vectorize(self.PeeAvg)
        self.PmmAvg_vec_deGouvea = np.vectorize(self.PmmAvg_deGouvea)

        # Load MiniBooNE detector efficiency data
        self.pathdata = "MiniTools/include/miniboone_eff/eg_effs.dat"
        self.eff_data = np.loadtxt(self.pathdata)
        self.eff_bin_edges = self.eff_data[:, 0] / 1000  # GeV
        self.eff = self.eff_data[:, 1]

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
    def _Fosc(Length, Losc, Ldec):
        return (
            4 * np.sin(np.pi / 2 * Length / Losc) ** 2 * np.exp(-Length / Ldec / 2)
            + (1 - np.exp(-Length / Ldec / 2)) ** 2
        )

    def Fosc(self, E4, Length):
        """Prob of oscillation, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fosc(Length, self.Losc(E4), self.Ldec(E4))

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _Fdec(Length, Ldec):
        return 1 - np.exp(-Length / Ldec)

    def Fdecay(self, E4, Edaughter, Length):
        """Decay probability function, E4 -- GeV, Length -- Kilometers"""
        return Sterile._Fdec(Length, self.Ldec(E4)) * Sterile.dPdecaydX(E4, Edaughter)

    def FdecayAvg(self, Emin, Emax, Length):
        """dPdecaydX --> 1"""
        return integrate.quad(
            lambda E4: (1 - np.exp(-Length / self.Ldec(E4))), Emin, Emax
        )[0] / (Emax - Emin)

    def FoscAvg(self, Emin, Emax, Length):
        # integrand = lambda E4: self.Fosc(E4, Length)
        # return integrate.quad(integrand, Emin, Emax, )[0] / (Emax - Emin)
        x = np.linspace(Emin + 1e-6, Emax, 1000)
        return np.sum(self.Fosc(x, Length) * (x[1] - x[0])) / (Emax - Emin)

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
                - (
                    Emin
                    - np.exp(-cst / (Emin + 1e-10)) * Emin
                    - cst * expi(-cst / (Emin + 1e-10))
                )
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

    def Pmedecay(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Um4Sq
            * Sterile._Fdec(Length, self.Ldec(E4))
            * Sterile.dPdecaydX(E4, Edaughter)
            * Edaughter
            / E4
            * self.MiniEff(Edaughter)
            / self.MiniEff(E4)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmeosc(self, E4, Length):
        # Oscillation term
        posc = self.Um4Sq * self.Ue4Sq * self.Fosc(E4, Length)
        return posc

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
            # overlap of daughter with nu_mu state
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
        # print(posc, pdecay)
        return 1 + pdecay - posc

    def Peedecay(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Ue4Sq
            * Sterile._Fdec(Length, self.Ldec(E4))
            * Sterile.dPdecaydX(E4, Edaughter)
            * Edaughter
            / E4
            * self.MiniEff(Edaughter)
            / self.MiniEff(E4)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Ue4Sq / (1 - self.Us4Sq)
        return pdecay

    def Peeosc(self, E4, Length):
        # Oscillation term
        posc = self.Ue4Sq * (1 - self.Ue4Sq) * self.Fosc(E4, Length)
        return 1 - posc

    def Pmmdecay(self, E4, Edaughter, Length):
        """Flavor transition probability, E4 -- GeV, Edaughter -- GeV, Length -- km"""
        # Decay term
        # pdecay = self.Um4Sq * self.Fdecay(E4, Edaughter, Length)
        # degradation * xsec * efficiency
        pdecay = (
            self.Um4Sq
            * Sterile._Fdec(Length, self.Ldec(E4))
            * Sterile.dPdecaydX(E4, Edaughter)
            * Edaughter
            / E4
            * self.MiniEff(Edaughter)
            / self.MiniEff(E4)
        )
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def Pmmosc(self, E4, Length):
        # Oscillation term
        posc = self.Um4Sq * (1 - self.Um4Sq) * self.Fosc(E4, Length)
        return 1 - posc

    # ----------------------------------------------------------------
    # de Gouvea's model
    # ----------------------------------------------------------------
    def Pme_deGouvea(self, E4, Edaughter, Length):
        return (
            self.Um4Sq
            * (1 - np.exp(-Length / (2 * self.Ldec(E4))))
            * Sterile.dPdecaydX(E4, Edaughter)
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
        pdecay = 1
        # decay term in Pmm, Emin and Emax are E4 bin edges
        if which_experiment == "microboone":
            if Eintmax < 1:  # Should this be Emax or Eintmax? We should discuss.
                n = 2 + noffset
            else:
                n = 1 + noffset
            pdecay = (
                self.Um4Sq
                * self.FdecayAna(Emin, Emax, Length)
                * ((Eintmax**2 - Eintmin**2) / ((Emax - E0) * (Emax + E0)))
                * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
            )
        elif which_experiment == "miniboone":
            pdecay = (
                self.Um4Sq
                * self.FdecayAna(Emin, Emax, Length)
                * ((Eintmax**2 - Eintmin**2) / ((Emax - E0) * (Emax + E0)))
                * ((Eintmin + Eintmax) / (Emin + Emax))
                * (
                    (self.MiniEff(Eintmin) + self.MiniEff(Eintmax))
                    / (self.MiniEff(Emin) + self.MiniEff(Emax))
                )
            )
        # ((Eintmax**2 - Eintmin**2)/(Emax*Emin)) factor is to account for the decay rate scaling with Eint/E4
        if not self.decouple_decay:
            # overlap of daughter with nu_e state
            pdecay *= self.Us4Sq * self.Um4Sq / (1 - self.Us4Sq)
        return pdecay

    def PmmoscAvg(self, Emin, Emax, Length):
        if Emin == 0.0:
            Emin = 0.000001
        if Emax == 0.0:
            Emax = 0.000001
        # osc term in Pmm, does not involve energy degradation
        return 1 - self.Um4Sq * (1 - self.Um4Sq) * self.FoscAna(Emin, Emax, Length)

    def PeedecayAvg(
        self, Ebins, e4_index, eint_index, Length, which_experiment, noffset=0
    ):
        Eintmin, Eintmax = Ebins[eint_index], Ebins[eint_index + 1]
        Emin, Emax = Ebins[e4_index], Ebins[e4_index + 1]
        E0 = Ebins[0]
        pdecay = 1
        # decay term in Pee, Emin and Emax are E4 bin edges
        if which_experiment == "microboone":
            if Eintmax < 1:  # Should this be Emax or Eintmax? We should discuss.
                n = 2 + noffset
            else:
                n = 1 + noffset
            pdecay = (
                self.Ue4Sq
                * self.FdecayAna(Emin, Emax, Length)
                * ((Eintmax**2 - Eintmin**2) / ((Emax - E0) * (Emax + E0)))
                * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
            )
        elif which_experiment == "miniboone":
            pdecay = (
                self.Um4Sq
                * self.FdecayAna(Emin, Emax, Length)
                * ((Eintmax**2 - Eintmin**2) / ((Emax - E0) * (Emax + E0)))
                * ((Eintmin + Eintmax) / (Emin + Emax))
                * (
                    (self.MiniEff(Eintmin) + self.MiniEff(Eintmax))
                    / (self.MiniEff(Emin) + self.MiniEff(Emax))
                )
            )
        # pdecay = self.Ue4Sq * self.FdecayAvg(Emin, Emax, Length) * ((Eintmin + Eintmax) / (Emin + Emax)) ** n
        # ((Eintmax**2 - Eintmin**2)/(Emax*Emin)) factor is to account for the decay rate scaling with Eint/E4 -- gives the fraction of
        # events in this bin
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
        # NOTE: factor of 2 is to acconunt for the decay rate scaling with Edaughter/Eparent
        return decay_w_base * 2

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

    def EnergyDegradation(
        self, Etrue_dist, Etrue_bins, which_channel, which_experiment
    ):
        R_deg = np.zeros((len(Etrue_dist), len(Etrue_dist)))
        R_osc = []
        # degradation piece
        for k in range(len(Etrue_dist)):
            for i in range(k + 1):
                Pdecay = 1

                if which_channel == "Pee":
                    Pdecay = self.PeedecayAvg(
                        Etrue_bins, k, i, L_micro, which_experiment, noffset=0
                    )
                elif which_channel == "Pmm":
                    Pdecay = self.PmmdecayAvg(
                        Etrue_bins, k, i, L_micro, which_experiment, noffset=0
                    )
                else:
                    raise ValueError(
                        f"Channel {which_channel} not recognzied. Valid options: 'Pmm' and 'Pee'."
                    )

                R_deg[k][i] = (
                    Pdecay * Etrue_dist[k]
                )  # k indexes true energy, i indexes degraded energy

        R_sum = np.sum(R_deg, axis=0)

        # oscillation piece
        for i in range(len(Etrue_dist)):
            if which_channel == "Pee":
                Peeosc = self.PeeoscAvg(Etrue_bins[i], Etrue_bins[i + 1], L_micro)
                R_osc.append(Peeosc * Etrue_dist[i])
            elif which_channel == "Pmm":
                Pmmosc = self.PmmoscAvg(Etrue_bins[i], Etrue_bins[i + 1], L_micro)
                R_osc.append(Pmmosc * Etrue_dist[i])
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

    def MiniEff(self, x):
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
