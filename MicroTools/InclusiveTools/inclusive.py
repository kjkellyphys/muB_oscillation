import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize
import os

from MicroTools import *

BkgVec = np.load(f"{muB_inclusive_data_path}/TotalBackground.npy") #Expected total background rate in the inclusive channel, 25 bins between 0 and 2.5 GeV
LEEVec = np.load(f"{muB_inclusive_data_path}/LEE_x1_Expectation.npy") #Expected Low-Energy-Excess rate (full strength) in the inclusive channel, 25 bins between 0 and 2.5 GeV
DatVec = np.load(f"{muB_inclusive_data_path}/ObservedData.npy") #Observed inclusive-channel data, 25 bins between 0 and 2.5 GeV
SigCorr = np.load(f"{muB_inclusive_data_path}/SigCorr_PD.npy") #Correlation matrix amongst [25, 25] signal bins

for kk in range(25):
    SigCorr[kk][kk] = 1.0
PoC_Uncertainty = np.load(f"{muB_inclusive_data_path}/PostConstraint_Uncertainty.npy") #Diagonal uncertainty ratio after applying background-channel constraints

BinEdge = np.linspace(0, 2.5, 26)
BinCenter = (BinEdge[:-1] + BinEdge[1:])/2
BinWidth = (BinEdge[1:] - BinEdge[:-1])

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
        return mi/2.0
    else:
        return 3.0/(1.0/ni + 2.0/mi)

def PLL(ni, mi):
    """Poissonian Log-Likelihood
       Arguments:
          ni {int} -- Observation in bin i
          mi {float} -- Model expectation in bin i
       Returns
          [float] -- Test-statistic -2*negative-log-likelihood in bin i
    """
    if mi < 0.0:
        return 1e100
    if mi == 0.0:
        return 1e100
    elif ni == 0.0:
        return 2.0*mi
    else:
        return -2.0*(-mi + ni + ni*np.log(mi/ni))

def Chi2_Inclusive(temp):
    """Returns MicroBooNE Chi-Squared for Inclusive Channel, given model template temp
    """
    v0 = DatVec - (BkgVec + temp)
    MV = BkgVec + temp
    nbins = 25 #Using full range of MicroBooNE Inclusive Channel. Alternatively, nbins = 6 uses [0, 600] MeV range
    sc0 = np.zeros((nbins, nbins))
    for jj in range(nbins):
        for kk in range(nbins):
            sc0[jj][kk] += np.sqrt(((PoC_Uncertainty[jj]-1.0)*MV[jj])*((PoC_Uncertainty[kk]-1.0)*MV[kk]))*SigCorr[jj][kk]
            if jj == kk:
                sc0[jj][kk] += CNPStat(DatVec[jj], MV[jj])

    return np.dot(np.dot(v0, inv(sc0)), v0)

def Chi2_Inclusive_Asimov(temp):
    """Returns MicroBooNE Chi-Squared Asimov-Expected Sensitivity for Inclusive Channel, given model template temp
    """
    v0 = BkgVec - (BkgVec + temp)
    MV = BkgVec + temp
    nbins = 25 #Using full range of MicroBooNE Inclusive Channel. Alternatively, nbins = 6 uses [0, 600] MeV range
    sc0 = np.zeros((nbins, nbins))
    for jj in range(nbins):
        for kk in range(nbins):
            sc0[jj][kk] += np.sqrt(((PoC_Uncertainty[jj]-1.0)*MV[jj])*((PoC_Uncertainty[kk]-1.0)*MV[kk]))*SigCorr[jj][kk]
            if jj == kk:
                sc0[jj][kk] += CNPStat(BkgVec[jj], MV[jj])

    return np.dot(np.dot(v0, inv(sc0)), v0)

def Poisson_Inclusive(AiVec, *info):
    """Returns MicroBooNE Chi-Squared for Inclusive Channel, given model template temp
    """
    DV0, BV0, T0, nbinsP = info[0]
    v0 = DV0 - (BV0 + T0)
    MV = BV0 + T0

    if np.min(AiVec) < -1.0 or np.max(AiVec) > 10.0:
        return 5000.0

    #Calculate the test statistic with the (modified) signal expectation
    TS0 = np.sum([PLL(DV0[kk], np.max([0.0, (1.0 + AiVec[kk])*MV[kk]])) for kk in range(nbinsP)])

    sc0 = np.zeros((nbinsP, nbinsP))
    for jj in range(nbinsP):
        for kk in range(nbinsP):
            sc0[jj][kk] += np.sqrt(((PoC_Uncertainty[jj]-1.0)*(1.0 + AiVec[jj])*MV[jj])*((PoC_Uncertainty[kk]-1.0)*(1.0 + AiVec[kk])*MV[kk]))*SigCorr[jj][kk]
    
    MV2 = [(AiVec[kk])*MV[kk] for kk in range(nbinsP)]
    systterm = np.dot(np.dot(MV2, inv(sc0)), MV2)
    weaksyst = np.sum([AiVec[kk]**2 for kk in range(nbinsP)])

    return TS0 + systterm + weaksyst

def Chi2_Inclusive_Poisson(temp, nbinsP):
    """Minimizes Poisson_Inclusive() over the nuisance parameters
    """
    AiVec1 = np.zeros((nbinsP, nbinsP))
    for kk in range(nbinsP):
        AiVec1[kk][kk] = 0.1
    zeroVec0 = np.zeros(nbinsP)
    res0 = minimize(Poisson_Inclusive, zeroVec0, args=[DatVec, BkgVec, temp, nbinsP], options={'disp':False, 'direc':AiVec1}, tol=0.0001, method='Powell')
    return res0.fun
