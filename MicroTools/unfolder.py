import numpy as np
import scipy.special as spec
import scipy.integrate as integ

from MicroTools import *

class MBtomuB:
    def __init__(self, analysis='1eX', remove_high_energy=False, unfold=True, effNoUnfold=False):
        """This class converts an excess in MiniBooNE 2018 to the corresponding signal in MicroBooNE
        
           Parameters:
           analysis           ---- MicroBooNE analysis. Possible values are '1eX' [inclusive, fully containted], 
                                   '1eX_PC' [inclusive, partially contained], 'CCQE' [1e1p quasielastic]
           remove_high_energy ---- The official MicroBooNE analysis removes all excess with neutrino energy greater than
                                   800 MeV. If this option is set to True, this will also be enforced. Default: False
           unfold             ---- Perform unfolding? If True, the input to miniToMicro must be the number of events 
                                   *as a function of reconstructed neutrino energy*. If False, the input must be the 
                                   number of events *as a function of true neutrino energy*
           effNoUnfold        ---- ***Only relevant if unfold=False***
                                   Does the input include MiniBooNE efficiencies? If True, the input to miniToMicro
                                   must be the number of events after MiniBooNE efficiencies. If False, the input
                                   must be the number of events before MiniBooNE efficiencies.
        """
        if analysis not in ['1eX', '1eX_PC', 'CCQE']:
            raise NotImplementedError("The analysis "+analysis+" is not among the implemented analyses: [1eX, 1eX_PC, CCQE]")
        self._remove_high_energy = remove_high_energy
        self._unfold = unfold
        self._effnounfold = effNoUnfold
            
        ## Set up data needed for unfolding ##
        self._migration = np.loadtxt(f"{path_unfolding_data}Migration_matrix_paper_bins.dat") # MiniBooNE migration matrix
        self.MB_eff = np.sum(self._migration, axis=0)
        self._MB_MC = np.loadtxt(f"{path_unfolding_data}MC_truth.dat") # MiniBooNE nu_e background. To be used as a prior for unfolding

        
        self._bin_edges_rec_MB = np.array([200,  300,  375,  475,  550,  675,  800,  950,  1100,  1250,  1500,  3000]) # MiniBooNE bin edges [MeV]
        self._bin_edges_true = np.array([200, 250, 300, 350, 400, 450, 500, 600, 800, 1000, 1500, 2000, 2500, 3000]) # Bin edges after unfolding [MeV]
        
        self._bin_widths_rec_MB = np.diff(self._bin_edges_rec_MB)
        self._bin_widths_true = np.diff(self._bin_edges_true)
        self._bin_centers_true = (self._bin_edges_true[1:] + self._bin_edges_true[:-1])/2
        
        ## Set up MicroBooNE parameters ##
        self._relative_targets = 85 / 818 # 85 ton [MicroBooNE] vs 818 ton [MiniBooNE]

        # Efficiency, energy bins, energy resolution, and fudge efficiency
        if analysis=='CCQE':
            self._relative_exposure = 6.67/12.84 # 6.67e20 POT [MicroBooNE] vs 12.84e20 POT [MiniBooNE 2018]
            # Efficiency as obtained by comparing Fig. 1 in arXiv:2110.13978 with their MC binned in true energy
            self._efficiency = np.array([0.01888007, 0.06836754, 0.04918433, 0.07322091,
                                         0.05812577, 0.0611779, 0.05131235, 0.04774184,
                                         0.03731489, 0.01934908, 0.00426763, 0.00098117, 0.00010148])
            # We multiply by the ratio in nu_e bkg between MiniBooNE and MicroBooNE. This takes into account the different cross-sections
            self._efficiency *= np.array([1.20982679, 0.65337331, 1.10746906, 0.81288558,
                                          1.03066457, 0.90310029, 0.99587775, 1.06028056,
                                          1.03312786, 1.12102814, 1.16614954, 1.16061468, 1.15696717])
            self._bin_edges_rec_microB = np.linspace(200, 1200, 11)
            self._fudge = np.ones(len(self._bin_edges_rec_microB)-1) # No fudge factors here! :-)

            self._smearing_matrix_microB = np.loadtxt(f"{path_unfolding_data}Migration_CCQE.dat")

        elif analysis=='1eX':
            self._relative_exposure = 6.369/12.84 # 6.369 POT [MicroBooNE] vs vs 6.46e20 POT [MiniBooNE 2012]
            # Efficiency as obtained by comparing Fig. 1 in arXiv:2110.13978 with their migration matrix binned in true energy
            self._efficiency = np.array([0.50314466, 0.54882715, 0.64019214, 0.72976845, 0.81755608, 0.90826832,
                                         1.04872359, 1.15368084, 1.22829937, 1.21671845, 1.18336647, 1.03151614,
                                         0.90760706])

            self._bin_edges_rec_microB = np.linspace(100, 2500, 25)
            self._fudge = np.array([0.24356887, 0.24804412, 0.25286889, 0.25622915, 0.26522795, 0.28182341,
                                    0.28308206, 0.27329645, 0.27794804, 0.28826545, 0.29780453, 0.29065536,
                                    0.29850116, 0.29434252, 0.29046458, 0.29612374, 0.28790173, 0.28431594,
                                    0.27852108, 0.27849889, 0.26675044, 0.25958135, 0.2632961, 0.25717796])
            self._smearing_matrix_microB = np.loadtxt(f"{path_unfolding_data}Migration_1eX.dat")
        elif analysis=='1eX_PC':
            self._relative_exposure = 6.369/12.84 # 6.369 POT [MicroBooNE] vs vs 6.46e20 POT [MiniBooNE 2012]
            # Efficiency as obtained by comparing Fig. 1 in arXiv:2110.13978 with their migration matrix binned in true energy
            self._efficiency = np.array([0.01514732, 0.03039166, 0.06088035, 0.09020686, 0.11837119, 0.15031587,
                                         0.20390339, 0.30090062, 0.44590193, 0.60864919, 0.80104988, 0.89476233,
                                         0.9417476])
            self._bin_edges_rec_microB = np.linspace(100, 2500, 25)
            self._fudge = np.array([0.27324992, 0.27151672, 0.28002696, 0.27200681, 0.29572426, 0.29276621,
                                    0.28861726, 0.2792527, 0.30009332, 0.30262111, 0.29996619, 0.30985736,
                                    0.30009521, 0.27908594, 0.30160061, 0.30353468, 0.29630258, 0.29723266,
                                    0.29916651, 0.30065366, 0.30708662, 0.31920767, 0.32554718, 0.33372028])
            self._smearing_matrix_microB = np.loadtxt(f"{path_unfolding_data}Migration_1eX_PC.dat")            

    def miniToMicro(self, mini_nue):
        """Convert excess in MiniBooNE 2018 to the corresponding signal in MicroBooNE
           If unfold was set to True:
               The input is the number of excess events in each MiniBooNE bin.
               The binning used must be the standard MiniBooNE binning. I.e., the bin edges are:
                 [200 MeV, 300 MeV, 375 MeV, 475 MeV, 550 MeV, 675 MeV, 800 MeV, 950 MeV, 1100 MeV, 1250 MeV, 1500 MeV, 3000 MeV]
           If unfold was set to False:
               The input is the number of excess events in MiniBooNE in the following *true neutrino energy* bins
                 [200 MeV, 250 MeV, 300 MeV, 350 MeV, 400 MeV, 450 MeV, 500 MeV, 600 MeV, 800 MeV, 1000 MeV, 1500 MeV, 2000 MeV, 2500 MeV, 3000 MeV]
               Notice that this is the number of events **without** selection efficiencies. The MiniBooNE efficiencies as a function of *true* neutrino
               energy are in the member variable MB_eff.

           To see the MicroBooNE bin edges used for the output, check the member variable _bin_edges_rec_microB
        """

        if(self._unfold):
            # Use D'Agostini's method to unfold the MiniBooNE excess to true neutrino energy
            u = self._MB_MC * self._bin_widths_true
            for i in range(3):
                A_times_u = np.multiply(self._migration, u)
                M = np.multiply(A_times_u.T, 1/np.sum(A_times_u, axis=1)).T
                M = np.multiply(M, 1/self.MB_eff)
                u = np.matmul(M.T, mini_nue)

            u = np.where(u<0, 0, u) # If the excess is negative, set it to 0
        else:
            u = mini_nue
            if self._effnounfold:
                u /= self.MB_eff
            
        ## Transform to MicroBooNE ##
        # Set to 0 everything above 800 MeV?
        if self._remove_high_energy:
            u[8:] = 0
            
        # Multiply by the relative exposure, number of targets, and efficiency
        u *= self._relative_exposure * self._relative_targets * self._efficiency

        # Smear in energy
        u = np.matmul(self._smearing_matrix_microB, u)

        # Introduce the fudge factors
        u *= self._fudge

        return u
