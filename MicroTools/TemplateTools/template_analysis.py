import os
import sys
import numpy as np
import scipy

from MicroTools import *
from MicroTools import unfolder 
from MicroTools.InclusiveTools import inclusive
from MicroTools.TemplateTools import mcmc_signal
from MicroTools.TemplateTools import miniboone_fit as mbfit

'''
Templates

   This class handles the template analysis for a given MCMC run
        
        case: may be "signal" or "background", depending on whether the MCMC
        was run with mcmc_signal or mcmc_background 
        
        nbins: how many bins to vary wrt nominal template (starting from lowest energy)
        
        name: string for labeling the current case of interest

'''
class Templates():
    
    # run MCMC
    def __init__(self, case="signal", nbins = 5, name=''):
        
        self.case = case
        self.name = name
        self.nbins = nbins
        self.template_path = f'{PATH_MCMC_CHAINS}/miniboone_signal_chain_{self.nbins}_bins_example.dat'

        # properties of nominal template
        self.nominal_nevents = 360
        self.nominal_template = mbfit.mb_nue_analysis_data - mbfit.mb_nue_analysis_predicted_background
        self.nominal_template[mbfit.bin_centers>800] = 0.0

        # properties of sterile best fit "template"
        strl = [4.07399, 3.28421, 2.93342, 2.223875, 1.78651, 1.301688, 1.07318, 0.8050, 0.544665, 0.3872957, 0.0817373]
        SterileBF = mbfit.bin_width * strl
        self.mb_sterile_best_fit = SterileBF-mbfit.mb_nue_analysis_predicted_background
        self.mb_sterile_best_fit[self.mb_sterile_best_fit<0]=0


    ''' 
        Generates the templates with a MCMC run
        
        Currently only uses signal model-indepenent method.
        FIX ME to include background method.
    '''
    def generate_templates(self):

        mcmc_signal.run_mcmc(nwalkers=50, 
                            nsteps=3000, 
                            threads=4, 
                            burnin_steps=2000, 
                            nbins=self.nbins, 
                            model='signal',
                            appendix=name)


    # load the templates produced from MCMC
    def load_templates(self, force_path='', shrink=False):
        
        if not force_path:
            self.mcmc_chain = np.genfromtxt(self.template_path,unpack=True)
        else:
            self.mcmc_chain = np.genfromtxt(force_path,unpack=True)
        
        if shrink:
            self.mcmc_chain = self.mcmc_chain[np.random.randint(len(templates), size=min(discard,len(templates)))]

        # compute the ratio of events of new tempalte with nominal one
        self.signal_strength = np.sum(self.mcmc_chain, axis=1)/self.nominal_nevents

        # MiniBooNE average neutrino energy
        self.avg_energy = np.matmul(self.mcmc_chain, mbfit.bin_centers)/self.signal_strength/self.nominal_nevents

        # number of templates
        self.N_templates = np.size(self.signal_strength)


    def unfold(self, **args):

        # unfolder all tempaltes 
        self.converter = unfolder.MBtomuB(**args)

        # unfold all tempaltes 
        self.micro_templates = np.empty(0)
        for template in self.mcmc_chain:
            template[template<0] = 0.0
            self.micro_templates = np.append(self.micro_templates, np.append(0.0, self.converter.miniToMicro(template)))
        
        self.micro_templates = np.reshape(self.micro_templates, (np.shape(self.mcmc_chain)[0], 25))

        self.micro_nominal = np.append(0.0, self.converter.miniToMicro(self.nominal_template))

        # compute the ratio of events of new tempalte with nominal one at MicroBooNE
        self.mub_signal_strength = np.sum(self.micro_templates, axis=1)/np.sum(self.micro_nominal)

        # MicroBoonE average neutrino energy
        self.mub_avg_energy = np.matmul(self.micro_templates,inclusive.BinCenter*1e3)/np.sum(self.micro_nominal)


    def compute_chi2s(self):
        
        # MiniBooNE chi2s
        self.mb_x = np.empty(0)
        self.mb_chi2s = np.empty(0)
        for mb_template, mub_template in zip(self.mcmc_chain, self.micro_templates):
            self.mb_x = np.append(self.mb_x, np.sum(mb_template)/self.nominal_nevents)
            self.mb_chi2s = np.append(self.mb_chi2s, mbfit.CalculateChi2SignalExpectations(mb_template))

        
        # MicroBooNE chi2s
        self.mub_x = np.empty(0)
        self.mub_chi2s = np.empty(0)
        for template in self.micro_templates:
            self.mub_x = np.append(self.mub_x, np.sum(template)/self.nominal_nevents)
            self.mub_chi2s = np.append(self.mub_chi2s, inclusive.Chi2_Inclusive(template))

        self.mb_delta_chi2s = self.mb_chi2s - np.min(self.mb_chi2s)
        self.mub_delta_chi2s = self.mub_chi2s - np.min(self.mub_chi2s)

    # Compute MiniBooNE pvalue
    def compute_MB_pval(self, ndf=4.7):
        self.mb_pval=np.empty(0)
        for template in self.mcmc_chain:
            self.mb_pval = np.append(self.mb_pval, mcmc_signal.get_pvalue(template, ndf=ndf))        

        self.mask_80 = (self.mb_pval < scipy.stats.chi2.isf(0.8, ndf))
        self.mask_10 = (self.mb_pval < scipy.stats.chi2.isf(0.1, ndf))
        self.mask_0p1 = (self.mb_pval < scipy.stats.chi2.isf(0.01, ndf))






