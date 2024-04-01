import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import param_scan


def run_scan(
    kwargs, filename, Npoints=10, path_results="fit_data/", sin2theta_scan=False
):

    # Range of mixings scanned
    dm_Vec = np.geomspace(np.sqrt(1e-1), np.sqrt(1e5), Npoints)
    g_Vec = np.geomspace(1e-2, 10, Npoints)

    if sin2theta_scan:
        s2thetaSq = np.geomspace(1e-4, 0.5, Npoints)
        Umu4Sq = 0.5
        # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
        paramlist = param_scan.create_grid_of_params_sin2theta(
            g=g_Vec, m4=dm_Vec, sin2thetaSq=s2thetaSq, Um4Sq=Umu4Sq
        )

    else:
        Ue4Sq = np.geomspace(1e-4, 0.5, Npoints)
        Umu4Sq = np.geomspace(1e-4, 0.5, Npoints)
        # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
        paramlist = param_scan.create_grid_of_params(
            g=g_Vec, m4=dm_Vec, Ue4Sq=Ue4Sq, Um4Sq=Umu4Sq
        )

    # Pure oscillation method
    func_scan = partial(param_scan.DecayReturnMicroBooNEChi2, **kwargs)

    with Pool() as pool:
        # res = np.array(list(pool.map(func_scan, paramlist)))
        res = np.array(
            list(tqdm(pool.imap(func_scan, paramlist), total=len(paramlist)))
        )
    param_scan.write_pickle(f"{path_results}/{filename}", res)
    return res


def run_scan_gfixed(
    kwargs,
    filename,
    gfixed=2.5,
    Npoints=10,
    path_results="fit_data/",
    sin2theta_scan=False,
):

    # Range of mixings scanned
    dm_Vec = np.geomspace(np.sqrt(1e-1), np.sqrt(1e5), Npoints)
    g_Vec = gfixed

    if sin2theta_scan:
        s2thetaSq = np.geomspace(1e-4, 0.5, Npoints)
        Umu4Sq = 0.5
        # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
        paramlist = param_scan.create_grid_of_params_sin2theta(
            g=g_Vec, m4=dm_Vec, sin2thetaSq=s2thetaSq, Um4Sq=Umu4Sq
        )

    else:
        Ue4Sq = np.geomspace(1e-3, 0.5, Npoints)
        Umu4Sq = np.geomspace(1e-4, 0.5, Npoints)
        # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
        paramlist = param_scan.create_grid_of_params(
            g=g_Vec, m4=dm_Vec, Ue4Sq=Ue4Sq, Um4Sq=Umu4Sq
        )

    # Pure oscillation method
    func_scan = partial(param_scan.DecayReturnMicroBooNEChi2, **kwargs)

    with Pool() as pool:
        # res = np.array(list(pool.map(func_scan, paramlist)))
        res = np.array(
            list(tqdm(pool.imap(func_scan, paramlist), total=len(paramlist)))
        )
    param_scan.write_pickle(f"{path_results}/{filename}", res)
    return res


def run_scan_gfixed_Ue4SQRfixed(
    kwargs,
    filename,
    gfixed=2.5,
    Ue4SQRfixed=0.05,
    Npoints=10,
    path_results="fit_data/",
    sin2theta_scan=False,
):

    # Range of mixings scanned
    dm_Vec = np.geomspace(np.sqrt(1e-1), np.sqrt(1e5), Npoints)
    Umu4Sq = np.geomspace(1e-4, 0.5, Npoints)

    # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
    paramlist = param_scan.create_grid_of_params(
        g=gfixed, m4=dm_Vec, Ue4Sq=Ue4SQRfixed, Um4Sq=Umu4Sq
    )

    # Pure oscillation method
    func_scan = partial(param_scan.DecayReturnMicroBooNEChi2, **kwargs)

    with Pool() as pool:
        # res = np.array(list(pool.map(func_scan, paramlist)))
        res = np.array(
            list(tqdm(pool.imap(func_scan, paramlist), total=len(paramlist)))
        )

    param_scan.write_pickle(f"{path_results}/{filename}", res)
    return res


# Common attributes to all osc_only scans
kwargs_common = {
    "oscillations": True,
    "decay": True,
    "decouple_decay": False,
    "energy_degradation": True,
    "include_antineutrinos": True,
    "n_replications": 10,
}

# Full and Standard case
kwargs_std = {
    "disappearance": True,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    **kwargs_common,
}


n = 30

# 4D scans
_ = run_scan(kwargs_std, "MH_decay_test_30", Npoints=n)

n = 40
# 3D scans
_ = run_scan_gfixed(kwargs_std, "MH_decay_gfixed_2.5_40", Npoints=n, gfixed=2.5)
_ = run_scan_gfixed(kwargs_std, "MH_decay_gfixed_1.0_40", Npoints=n, gfixed=1.0)


n = 60
# 2D scans
_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_2.5_Ue4SQRfixed_0.10",
    Npoints=n,
    gfixed=2.5,
    Ue4SQRfixed=0.10,
)

_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_2.5_Ue4SQRfixed_0.05",
    Npoints=n,
    gfixed=2.5,
    Ue4SQRfixed=0.05,
)
_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_2.5_Ue4SQRfixed_0.01",
    Npoints=n,
    gfixed=2.5,
    Ue4SQRfixed=0.01,
)


_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_1_Ue4SQRfixed_0.10",
    Npoints=n,
    gfixed=1,
    Ue4SQRfixed=0.10,
)

_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_1_Ue4SQRfixed_0.05",
    Npoints=n,
    gfixed=1,
    Ue4SQRfixed=0.05,
)
_ = run_scan_gfixed_Ue4SQRfixed(
    kwargs_std,
    "MH_decay_gfixed_1_Ue4SQRfixed_0.01",
    Npoints=n,
    gfixed=1,
    Ue4SQRfixed=0.01,
)
