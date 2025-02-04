import numpy as np

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from OscTools import unfolder

GBPC_NuE = unfolder.MBtoLAr(
    analysis="1eX_PC",
    remove_high_energy=False,
    unfold=False,
    effNoUnfold=False,
    which_template="2018",
)
GBFC_NuE = unfolder.MBtoLAr(
    analysis="1eX",
    remove_high_energy=False,
    unfold=False,
    effNoUnfold=False,
    which_template="2018",
)


Sets = [
    "nueCC_FC_",
    "nueCC_PC_",
    "numuCC_FC_",
    "numuCC_PC_",
    "numuCCpi0_FC_",
    "numuCCpi0_PC_",
    "NCpi0_",
]
LEEStr, SigStr, BkgStr, ObsStr = "LEE.npy", "Sig.npy", "Bkg.npy", "Obs.npy"

##################################################################
# Inclusive analysis
muB_inclusive_data_path = "OscTools.include.muB_data.inclusive_data"
muB_inclusive_datarelease_path = f"{muB_inclusive_data_path}.DataRelease"


LEESets = [
    np.load(files(muB_inclusive_datarelease_path).joinpath(si + LEEStr).open("rb"))
    for si in Sets
]
SigSets = [
    np.load(files(muB_inclusive_datarelease_path).joinpath(si + SigStr).open("rb"))
    for si in Sets
]
BkgSets = [
    np.load(files(muB_inclusive_datarelease_path).joinpath(si + BkgStr).open("rb"))
    for si in Sets
]
ObsSets = [
    np.load(files(muB_inclusive_datarelease_path).joinpath(si + ObsStr).open("rb"))
    for si in Sets
]

LEESetsF = np.concatenate(LEESets)
SigSetsF = np.concatenate(SigSets)
BkgSetsF = np.concatenate(BkgSets)
ObsSetsF = np.concatenate(ObsSets)

FCov = np.load(
    files(muB_inclusive_datarelease_path)
    .joinpath("MuBInclusive_FracCov_Square.npy")
    .open("rb")
)

SigTypes = ["nue", "nue", "numu", "numu", "numuPi0", "numuPi0", "NCPi0"]
BEdges0 = [0.0 + 0.1 * j for j in range(26)]
BEdges0.append(10.0)
Pi0BEdges0 = [0.0 + 0.1 * j for j in range(11)]
Pi0BEdges0.append(10.0)
BEdges = [BEdges0, BEdges0, BEdges0, BEdges0, Pi0BEdges0, Pi0BEdges0, Pi0BEdges0]


###########
# Numu data
NuMuCC_TrueEDist_FC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("TrueEDist_numuCC_FC.dat").open()
)
NuMuCC_MigMat_FC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("MigMat_numuCC_FC.dat").open()
)
NuMuCC_Eff_FC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("Efficiency_numuCC_FC.dat").open()
)

NuMuCC_TrueEDist_PC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("TrueEDist_numuCC_PC.dat").open()
)
NuMuCC_MigMat_PC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("MigMat_numuCC_PC.dat").open()
)
NuMuCC_Eff_PC = np.loadtxt(
    files(f"{muB_inclusive_data_path}").joinpath("Efficiency_numuCC_PC.dat").open()
)

MuB_BinEdges_NuMu = [0.0 + 0.05 * j for j in range(61)]
