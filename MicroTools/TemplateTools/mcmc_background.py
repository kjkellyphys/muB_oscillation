import matplotlib
matplotlib.use('Agg')
import numpy as np
import emcee
import sys
import tqdm
import time
import argparse
import scipy.stats as stats

from . import miniboone_neutrino_improved_fit as mbfit




SumExcs = 0
SumOD = 0
SumDl = 0
SumPi = 0
SumEn = 0
SumDa = 0
def SumBkg():
    global SumExcs
    global SumOD
    global SumDl
    global SumPi
    global SumEn
    global SumDa
    
    SumExcs =  np.sum(mbfit.mb_nominal_excess)

    for i in range(number_of_bins):
       SumOD += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * (Other[i] + Dirt[i])
       SumDl += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * Delta[i]
       SumPi += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * Pi0[i]
       SumEn += ((BinBound[i+1] - BinBound[i])/WBkg[i]) * (NuEK0[i] + NuEKp[i] + NuEM[i])
       SumDa += mbfit.mb_nue_analysis_data[i]

       
def CalculateNewBackgrounds(n_background_vector):
    background_neutrino =np.zeros(number_of_bins)

    eta0 = (SumDa - n_background_vector[0] * SumDl - n_background_vector[1] * SumPi - n_background_vector[2] * SumEn - SumExcs)/SumOD
    if eta0 < 0 :
        return background_neutrino
    
    for i in range(number_of_bins): 
        background_neutrino[i] = ((BinBound[i+1] - BinBound[i])/WBkg[i]) * ((Other[i] + Dirt[i]) * eta0 + Delta[i] * n_background_vector[0] + Pi0[i] * n_background_vector[1] + (NuEK0[i] + NuEKp[i] + NuEM[i])  * n_background_vector[2])

        

    return background_neutrino


def lnprior(n_background_vector):
    if any(n<0 for n in n_background_vector):
        return -np.inf
    return 0.0

def lnprob(theta):
            
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + GetLikelihood(theta)

#if __name__ == "__main__":
def run_mcmc_background(nwalkers=50, nsteps = 50000, threads=4, burnin_steps = 20000, nbins=11):
    
    RelativeNorm()
    SumBkg()
    
    tt = time.time()
    print("Initializing walkers")
    #nwalkers = args.walkers
    #nwalkers = nwalkers
    
    p0_base = [1.0]*number_of_backgrounds
    p0_std = [0.1]*number_of_backgrounds
    assert(len(p0_base) == len(p0_std))
    ndim = len(p0_base)
    p0 = np.random.normal(p0_base, p0_std, size=[nwalkers, ndim])


    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=args.threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads)

    print("Running burn-in")

    pos, prob, state = sampler.run_mcmc(p0, 1)
    #burnin_steps = args.burnin
    for _ in tqdm.tqdm(sampler.sample(pos, iterations=burnin_steps), total=burnin_steps):
        pass
    sampler.reset()
        
    #nsteps = args.steps

    for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
        pass
    print("Time elapsed", time.time()-tt)

    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

    np.savetxt("./miniboone_background_chain_NoAbs.dat",sampler.flatchain)

    import corner
    fig = corner.corner(samples)
    fig.savefig("./miniboone_background_triangle_new.png")

    
