import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import scipy
import emcee
import sys
import tqdm
import time

from MicroTools import *
from MicroTools.TemplateTools import miniboone_fit as mbfit
from MicroTools.InclusiveTools import inclusive
from MicroTools import unfolder 

NEVENTS_NOMINAL = 360

number_of_backgrounds = 3
number_of_bins = 11

NuEM = [0.26881717, 0.43322491, 0.56813788, 0.50590962, 0.49705416, 0.49174142,
        0.40728966, 0.33094358, 0.21159324, 0.15568566, 0.]


NuEKp = [0.07271037, 0.1222734,  0.17862495, 0.23069437, 0.23060864, 0.20047603,
         0.22411117, 0.19365187, 0.16835922, 0.13554328, 0.04772517]


NuEK0 = [0.04481658, 0.04755179, 0.04398599, 0.07189589, 0.05540792, 0.08263944,
         0.05338303, 0.05004523, 0.02752273, 0.03580412, 0.]

Pi0 = [2.05191139, 1.14640005, 0.68066036, 0.35258565, 0.30266261, 0.15816352,
       0.16398261, 0.13065389, 0.09875218, 0.04886478, 0.01253574]

Delta = [0.39175775, 0.55264774, 0.52771141, 0.22134511, 0.10582993, 0.04187509,
         0.005233,   0.,         0.,         0.,         0.]

Dirt = [0.24410324, 0.17854509, 0.12780866, 0.10610038, 0.06358978, 0.02739863,
        0.02747221, 0.01703119, 0.,         0.,         0.,]

Other = [0.53574058, 0.39505153, 0.26406301, 0.2078761,  0.17094773, 0.06413129,
         0.04629203, 0.02970044, 0.,         0.,         0.]

Sterile = [4.07399, 3.28421, 2.93342, 2.223875, 1.78651, 1.301688, 1.07318, 0.8050, 0.544665, 0.3872957, 0.0817373]

BinBound = [200.0,  300.0,  375.0,  475.0,  550.0,  675.0,  800.0,  950.0,  1100.0,  1250.0,  1500.0,  3000.0]


def RelativeNorm():
    WBkg =np.zeros(number_of_bins)
    for i in range(number_of_bins): 
        WBkg[i] = (BinBound[i+1] - BinBound[i]) * (Other[i] + Dirt[i] + Delta[i] + Pi0[i] + NuEK0[i] + NuEKp[i] + NuEM[i])/mbfit.mb_nue_analysis_predicted_background[i]

def CalculateNewBackgrounds(n_background_vector):
    background_neutrino =np.zeros(number_of_bins)

    eta0 = (SumDa - n_background_vector[0] * SumDl - n_background_vector[1] * SumPi - n_background_vector[2] * SumEn - SumExcs)/SumOD
    if eta0 < 0 :
        return background_neutrino
    
    for i in range(number_of_bins): 
        background_neutrino[i] = ((BinBound[i+1] - BinBound[i])/WBkg[i]) * ((Other[i] + Dirt[i]) * eta0 + Delta[i] * n_background_vector[0] + Pi0[i] * n_background_vector[1] + (NuEK0[i] + NuEKp[i] + NuEM[i])  * n_background_vector[2])

    return background_neutrino

def GetLikelihood(parameters, background_model=False):

    if background_model:

        bkg_prediction = CalculateNewBackgrounds(parameters)
        global_chi2 = 0.0
        global_chi2 += mbfit.CalculateChi2BackgroundExpectations(bkg_prediction)
    else:
         chi2_mb = mbfit.CalculateChi2SignalExpectations(parameters)
    
    # # Get microboone chi2
    # template = n_signal_vector
    # template[template<0] = 0 
    # mub_signal_vector = np.append(0.0, converter.miniToMicro(template))
    # chi2_mub = inclusive.Chi2_Inclusive(mub_signal_vector)
    
    return -chi2_mb/2. # converting from chi^2 to gaussian likelihood
    

def lnprior(parameters, background_model=False):
    if np.any(parameters < 0):
        return -np.inf
    return np.sum(np.log(1/np.sqrt(parameters)))

def lnprob(parameters, nbins=11, background_model=False):

    # assign leftover high energy bins to have zero excess
    if nbins < 11 and not background_model:
        parameters = np.append(parameters, np.zeros(11-nbins))

    lp = lnprior(parameters, background_model=background_model)

    if not np.isfinite(lp):
        return -np.inf
    
    L = GetLikelihood(parameters, background_model=background_model)

    if np.isfinite(L):
        return lp + L
    else:
        return -np.inf

### MCMC business
def run_mcmc(nwalkers=50, 
            nsteps = 3000, 
            threads=4, 
            burnin_steps = 1000, 
            nbins=11, 
            background_model=False,
            appendix=''):
    tt = time.time()
    print("Initializing walkers")

    assert nbins <= 11, "Asked for too many bins, miniboone data only has 11 Enu bins."

    if background_model:
        p0_base = [1.0]*number_of_backgrounds
        p0_std = [0.1]*number_of_backgrounds

    else:
        p0_base = np.ones(nbins)*50
        p0_std = [5]*nbins

    assert(len(p0_base) == len(p0_std))
    ndim = len(p0_base)
    p0 = np.random.normal(p0_base, p0_std, size=[nwalkers, ndim])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads, args=[nbins,background_model])

    print("Running burn-in")
    assert nwalkers > 2*ndim, "Number of walkers too low, ensure nwalkers > 2*ndims"
    

    pos, prob, state = sampler.run_mcmc(p0, burnin_steps)
    sampler.reset() 
    pos, prob, state = sampler.run_mcmc(p0, nsteps)

    # for _ in tqdm.tqdm(sampler.sample(pos, iterations=burnin_steps), total=burnin_steps):
    #     pass
    # sampler.reset() 

    # print("Running proper mcmc")
    # for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
    #     pass
    # print("Time elapsed", time.time()-tt)

    # samples = sampler.chain[:, nwalkers:, :].reshape((-1, ndim))
    samples = sampler.get_chain(discard=burnin_steps, flat=True)#, thin=15)
    
    if background_model:
    
        complete_templates = sampler.flatchain.T

    else:
        if nbins < 11:
            complete_templates = np.concatenate((sampler.flatchain, np.zeros((np.shape(sampler.flatchain)[0],11-nbins))), axis=1).T
        else:
            complete_templates = sampler.flatchain.T

    if not appendix:
        if background_model:
            appendix='bkg'
        else:
            appendix='free'
    
    np.savetxt(f"{PATH_MCMC_CHAINS}/miniboone_signal_chain_{nbins}_bins_{appendix}.dat", complete_templates)

    # labels=[f'{mbfit.bin_edges[i]} MeV - {mbfit.bin_edges[i+1]} MeV' for i in range(nbins)]
    labels=[f'bin {i}' for i in range(nbins)]
    
    fig = corner.corner(samples, show_titles=True,  plot_datapoints=False,
                                title_kwargs={"fontsize": 11}, 
                                smooth=True,labels=labels)
    fig.savefig(f"{PATH_MCMC_CHAINS}/miniboone_signal_triangle_{nbins}_bins.png")
    plt.close()

    return sampler



######################  low-level functions #######################

def new_template(x, n=2):
    # assert x >= 1, "You tried upscaling, only downscaling allowed."
    # the nominal LEE
    BF_template = mbfit.mb_nue_analysis_data-mbfit.mb_nue_analysis_predicted_background
    # downscale the first n bins
    downscaled_bins = BF_template[:n]/x
    # distribute the difference evenly
    recipients = range(n,len(BF_template))
    bin_size_correction = mbfit.bin_width[recipients]/np.sum(mbfit.bin_width[recipients])
    # bin_size_correction = BF_template[recipients]/np.sum(BF_template[recipients])
    BF_template[recipients] += np.sum(BF_template[:n]-downscaled_bins)*bin_size_correction
    BF_template[:n] = downscaled_bins
    return BF_template

def get_templates_for_all_scalings(xrange, n=2):
    chi2s = []
    scales= np.linspace(*xrange, 1000)
    for x in scales:
        chi2s.append(mbfit.CalculateChi2SignalExpectations(new_template(x, n = n)))
        # chi2s.append(get_pvalue(new_template(x, n = n)))
    return scales, chi2s


def get_pvalue(template, ndf=8.7):
    chi2 = mbfit.CalculateChi2SignalExpectations(template)
    return scipy.stats.chi2.sf(chi2, ndf)

def get_pvalue_muboone(chi2):
    return scipy.stats.chi2.sf(chi2, 25 - 11)

def filter_samples(samples, p_threshold = 0.1):
    filtered = np.array([])
    for template in samples:
        if get_pvalue(template) > p_threshold: #and template[0]>mbfit.mb_nominal_excess[0]:
            filtered = np.append(filtered,template)
    return np.reshape(filtered, (int(len(filtered)/11), 11))
    
def get_max_avgenu_templates(samples, n=4):
    avgenus = np.sum((samples.T*mbfit.bin_centers).T/np.sum(samples, axis=0),axis=0)
    ind = np.argpartition(avgenus, -n)[-n:]
    return avgenus[ind], samples.T[ind]
