import numpy as np
import unfolder
from multiprocessing import Pool, Value, Lock
import itertools
import os
RHE = False
UFMB = False
GBPC = unfolder.MBtomuB(analysis='1eX_PC', remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True)
GBFC = unfolder.MBtomuB(analysis='1eX', remove_high_energy=RHE, unfold=UFMB, effNoUnfold=True)

from inclusive_osc_tools import muB_OscChi2, PmmAvg, MuBNuEDis, MuBNuMuDis

from MicroTools import * 

#Files from Pedro -- event rates for MiniBooNE for various \Delta m_{41}^2 for \sin^2(2\theta_{\mu e}) = 1
MiniBooNE_Signal_PANM_True = np.loadtxt(f"{mb_data_osctables}/dm-MB-events-table-TrueEnu.dat")
MB_True_Bins = [0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.800, 1.000, 1.500, 2.000, 2.500, 3.000]
LMBT = 0.4685 #Baseline length in kilometers

lock = Lock()
number = Value('i', 0)

#--------------------------------------------------------------------------------
#                 Setting up Parameter Scan -- 3-dimensions
#--------------------------------------------------------------------------------
dm41Vec = np.unique(np.transpose(MiniBooNE_Signal_PANM_True)[0])
dm41IVec = [ii for ii in range(len(dm41Vec))]

#Range of \sin^2(2\theta_{ee}) = 4|U_{e4}|^2(1-|U_{e4}|^2) to scan
lEMin, lEMax, nlE = -3.0, 0.0, 30
EVec = [10**(lEMin + (lEMax-lEMin)/nlE*j) for j in range(nlE+1)]
EVec = np.insert(EVec, 0, [0.0])

#Range of \sin^2(2\theta_{\mu\mu}) = 4|U_{\mu4}|^2(1-|U_{\mu4}|^2) to scan
lMMin, lMMax, nlM = -3.0, 0.0, 30
MVec = [10**(lMMin + (lMMax-lMMin)/nlM*j) for j in range(nlM+1)]
MVec = np.insert(MVec, 0, [0.0])

np.save(f'{path_osc_data}/Dm41_SSq2EE_SSq2MM_PVs', np.asanyarray([dm41Vec, EVec, MVec], dtype=object))

paramlist = list(itertools.product(dm41IVec, EVec, MVec))
#--------------------------------------------------------------------------------

def ReturnMicroBooNEChi2(theta):
    dm41I, ssq2ee, ssq2mm = theta

    Ue4Sq = 0.5*(1.0 - np.sqrt(1.0 - ssq2ee))
    Um4Sq = 0.5*(1.0 - np.sqrt(1.0 - ssq2mm))

    if Ue4Sq + Um4Sq >= 1.0:
        return [1.0e4, 1.0e4]

    MB0 = MiniBooNE_Signal_PANM_True[dm41I]
    dm41 = MB0[0]
    MBSig0 = MB0[1:]

    MBSig = []
    for k in range(len(MBSig0)):
        RWFact = 4.0*Ue4Sq*Um4Sq/PmmAvg(MB_True_Bins[k], MB_True_Bins[k+1], LMBT, dm41, Um4Sq)
        MBSig.append(MBSig0[k]*RWFact)

    uBFC = GBFC.miniToMicro(MBSig)
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    uBPC = GBPC.miniToMicro(MBSig)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    NuEReps = MuBNuEDis(dm41Vec[dm41I], Ue4Sq)
    NuMuReps = MuBNuMuDis(dm41Vec[dm41I], Um4Sq)

    MuBResult = muB_OscChi2(Ue4Sq, Um4Sq, dm41, uBtemp, constrained=False, sigReps=[NuEReps[0], NuEReps[1], NuMuReps[0], NuMuReps[1], None, None, None], RemoveOverflow=True)
    MuBResult_Asimov = muB_OscChi2(Ue4Sq, Um4Sq, dm41, uBtemp, constrained=False, sigReps=[NuEReps[0], NuEReps[1], NuMuReps[0], NuMuReps[1], None, None, None], RemoveOverflow=True, Asimov=True)

    with lock:
        if number.value % 1000 == 0:
            print([number.value, theta, len(paramlist), MuBResult])
        number.value += 1

    return [MuBResult, MuBResult_Asimov]

if __name__ == '__main__':
    #Designed to run in parallel. Set the argument of "Pool" to 1 to disable this.
    pool = Pool()
    res = pool.map(ReturnMicroBooNEChi2, paramlist)
    np.save(f'{path_osc_data}/FullAnalysis_AppDis_SSq2EE_SSq2MM', res)
