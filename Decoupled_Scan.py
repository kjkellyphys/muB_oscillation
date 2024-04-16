import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

from multiprocessing import Pool
from functools import partial

import MicroTools as micro
import MicroTools.plot_tools as pt

import MiniTools.param_scan as param_scan
from MiniTools.param_scan import fast_histogram
from tqdm.notebook import tqdm  # Note the change here

# from tqdm.auto import tqdm

from ipywidgets import IntProgress
from IPython.display import display


def run_scan_osc_dec(kwargs, filename, Npoints=10, path_results="fit_data/"):

    gvec = np.geomspace(1e-2, 3.5, Npoints)
    mvec = np.geomspace(1e-1, 1e2, Npoints)
    # Ue4Sq = np.geomspace(1e-4, 0.49, Npoints)
    Umu4Sq = np.geomspace(1e-4, 0.49, Npoints)
    # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
    paramlist = param_scan.create_grid_of_params(g=gvec, m4=mvec, Ue4Sq=0, Um4Sq=Umu4Sq)

    # Pure oscillation method
    func_scan = partial(param_scan.DecayReturnMicroBooNEChi2, **kwargs)

    with Pool() as pool:
        # res = pool.map(func_scan, paramlist)
        res = np.array(
            list(tqdm(pool.imap(func_scan, paramlist), total=len(paramlist)))
        )

    param_scan.write_pickle(f"{path_results}/{filename}", res)
    return res


# Common attributes to all osc+dec scans
kwargs_common = {
    "oscillations": True,
    "decay": True,
    "decouple_decay": False,
    "include_antineutrinos": True,
    "n_replications": 10,
}

# Appearance only
kwargs_apponly = {
    "disappearance": False,
    "use_numu_MC": False,
    "undo_numu_normalization": False,
    **kwargs_common,
}

# Include disappearance and energy degradation
kwargs_std = {
    "disappearance": True,
    "use_numu_MC": True,
    "energy_degradation": True,
    "undo_numu_normalization": False,
    **kwargs_common,
}

# Include disappearance, but no energy degradation
kwargs_noed = {
    "disappearance": True,
    "use_numu_MC": True,
    "energy_degradation": False,
    "undo_numu_normalization": False,
    **kwargs_common,
}

# Oscillation only, no antineutrinos
kwargs_nobar_osc = {
    "oscillations": True,
    "decay": False,
    "decouple_decay": False,
    "disappearance": True,
    "energy_degradation": False,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    "n_replications": 10,
    "include_antineutrinos": False,
}

# osc+decay, no antineutrinos
kwargs_nobar = {
    "oscillations": True,
    "decay": True,
    "decouple_decay": False,
    "disappearance": True,
    "energy_degradation": False,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    "n_replications": 10,
    "include_antineutrinos": False,
}

# deGouvea's case
kwargs_deGouvea = {
    "oscillations": True,
    "decay": True,
    "decouple_decay": True,
    "disappearance": True,
    "energy_degradation": True,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    "n_replications": 10,
    "include_antineutrinos": True,
}

if __name__ == "__main__":
    _ = run_scan_osc_dec(kwargs_deGouvea, "TZ_decoupled", Npoints=30)
