import numpy as np
import copy
import sys
import os
toolsdir = os.path.realpath(__file__)[:-len('InclusiveTools/osc_inclusive_osc_bkg_decay.py')]
sys.path.append(toolsdir)
import unfolder
from multiprocessing import Pool, Value, Lock
import itertools
RHE = False
UFMB = False
GBPC = unfolder.MBtomuB(analysis='1eX_PC', remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True)
GBFC = unfolder.MBtomuB(analysis='1eX', remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True)

from inclusive_osc_tools import DecayPmmAvg, Decay_muB_OscChi2, DecayMuBNuMuDis

from MicroTools import *

MiniBooNE_Signal = np.loadtxt(f"{mb_data_osctables}/miniboone_numunuefullosc_ntuple_reweighted.dat")
MB_Ereco_Bins = [0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.800, 1.000, 1.500, 2.000, 2.500, 3.000]
LMBT = 0.4685 # Baseline length in kilometers
Ereco = MiniBooNE_Signal[:, 0]  # MeV
Etrue = MiniBooNE_Signal[:, 1]  # MeV
Length = MiniBooNE_Signal[:, 2] / 100  # meters
Weight = MiniBooNE_Signal[:, 3]

lock = Lock()
number = Value('i', 0)

e_prod_e_int_bins = np.linspace(0, 3, 51)
# Set up a migration matrix that maps Etrue to Ereco with shape of (50,13)
h0_unnorm = np.histogram2d(Etrue, Ereco, bins=[e_prod_e_int_bins, MB_Ereco_Bins], weights = Weight)[0]
migration_matrix = copy.deepcopy(h0_unnorm)
# Normalizing matrix elements w.r.t. to the interacting energy
for j in range(len(e_prod_e_int_bins)-1):
    row_sum = np.sum(h0_unnorm[j])
    if row_sum < 0.0:
        print("negative row?")
    if row_sum == 0.0:
        continue
    migration_matrix[j] = h0_unnorm[j]/row_sum

def e_prod_to_e_int(e_prod_test, weight_test, n_replications = 10):
    """
    Create a distribution of interaction energy for every production energy
    based on the energy distribution of the daughter neutrinos (eqn 2.3&2.4 in 1911.01447)
    Args:
        e_prod_test: production energy
        weight_test: corresponding weights
        n_replications: number of interaction energy bins per production energy

    Returns:
        e_int_bin_centers: bin centers of interaction energy bins
        bin_weights: weights of interaction energy bins
    """
    e_int_bin_edges = np.linspace(0, e_prod_test, n_replications+1)
    e_int_bin_centers = (e_int_bin_edges[1:] + e_int_bin_edges[:-1])/2.
    bin_weights = weight_test*np.array([(1 + 2*j)/n_replications**2 for j in range(n_replications)])
    return np.transpose([e_int_bin_centers, bin_weights])

#--------------------------------------------------------------------------------
#                 Setting up Parameter Scan -- 2-dimensions
#--------------------------------------------------------------------------------
# Range of gm4 for sterile decay model scanned over 0.1 to 100 in 30 steps
gm4Min, gm4Max ,ngm4 = -1.0, 2.0, 30
gm4Vec = [10**(gm4Min + (gm4Max-gm4Min)/ngm4*j) for j in range(ngm4+1)]
gm4IVec = [ii for ii in range(len(gm4Vec))]

# Range of |U_{\mu4}|^2 scanned over 1e-4 to 1 in 30 steps
lMMin, lMMax, nlM = -4.0, 0.0, 30
MVec = [10**(lMMin + (lMMax-lMMin)/nlM*j) for j in range(nlM+1)]

np.save(f'{path_osc_data}/gm4_Um4sq_PVs', np.asanyarray([gm4Vec,MVec],dtype=object))
paramlist_decay = list(itertools.product(gm4Vec, MVec))
#--------------------------------------------------------------------------------

# Here we define a function that returns the Chi2 under the assumption of sterile decay model
def DecayReturnMicroBooNEChi2(theta):
    gm4, Um4Sq = theta
    #Weighted decay appearance probability from eqn 2.8 in 1911.01447
    Pme_weighted = [(Um4Sq * (1 - np.exp(-1.267*gm4 ** 2 * Length[i] / (32 * np.pi * Etrue[i])))) * Weight[i] for i in
                    range(MiniBooNE_Signal.shape[0])]
    # bin centers(GeV) and bin weights of interaction energy for each production energy
    decay_e_prod, decay_e_weights = np.transpose(np.concatenate([e_prod_to_e_int(Etrue[i]/1000, Pme_weighted[i]) for i in range(len(Etrue))]))
    # dotting with the migration matrix to map to a distribution binning over reconstructed energy
    # Note: normalized to \sin^2(2\theta_{\mu e}) = 1
    MBSig0 = np.dot((np.histogram(decay_e_prod, bins=e_prod_e_int_bins, weights=decay_e_weights)[0]), migration_matrix)
    MBSig = []
    # Reweight event rate in each bin
    for k in range(len(MBSig0)):
        RWFact = 1/DecayPmmAvg(MB_Ereco_Bins[k], MB_Ereco_Bins[k + 1], LMBT, gm4, Um4Sq)
        MBSig.append(MBSig0[k]*RWFact)
    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = GBFC.miniToMicro(MBSig)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)
    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    uBPC = GBPC.miniToMicro(MBSig)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])
    # \nu_mu disappearance signal replacement
    NuMuReps = DecayMuBNuMuDis(gm4, Um4Sq)

    MuBResult = Decay_muB_OscChi2(Um4Sq, gm4, uBtemp, constrained=False, sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None], RemoveOverflow=True)
    MuBResult_Asimov = Decay_muB_OscChi2(Um4Sq, gm4, uBtemp, constrained=False, sigReps=[None, None, NuMuReps[0], NuMuReps[1], None, None, None], RemoveOverflow=True, Asimov=True)

    with lock:
        if number.value % 1000 == 0:
            print([number.value, theta, len(paramlist_decay), MuBResult])
        number.value += 1

    return [MuBResult, MuBResult_Asimov, uBFC]

if __name__ == '__main__':
    #Designed to run in parallel. Set the argument of "Pool" to 1 to disable this.
    pool = Pool()
    res = pool.map(DecayReturnMicroBooNEChi2, paramlist_decay)
    np.save(f'{path_osc_data}/App_gm4_Um4sq_migrated', res)

