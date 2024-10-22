import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import MicroTools.param_scan as param_scan


def run_scan_4D(
    kwargs, filename, Npoints=10, path_results="fit_data/", sin2theta_scan=False
):

    # Range of mixings scanned
    dm_Vec = np.geomspace(np.sqrt(1e-1), np.sqrt(1e5), Npoints)
    g_Vec = np.geomspace(1e-2, 10, Npoints)
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


def run_scan_gfixed_3D(
    kwargs,
    filename,
    gfixed=2.5,
    Npoints=10,
    dmSq_range=(1e-1, 1e5),
    Ue4Sq_range=(1e-3, 0.25),
    Umu4Sq_range=(2e-4, 0.25),
    path_results="fit_data/",
):

    # Range of mixings scanned
    if len(Npoints) == 1:
        dm_Vec = np.geomspace(np.sqrt(dmSq_range[0]), np.sqrt(dmSq_range[1]), Npoints)
        Ue4Sq = np.geomspace(Ue4Sq_range[0], Ue4Sq_range[1], Npoints)
        Umu4Sq = np.geomspace(Umu4Sq_range[0], Umu4Sq_range[1], Npoints)
    else:
        dm_Vec = np.geomspace(
            np.sqrt(dmSq_range[0]), np.sqrt(dmSq_range[1]), Npoints[0]
        )
        Ue4Sq = np.geomspace(Ue4Sq_range[0], Ue4Sq_range[1], Npoints[1])
        Umu4Sq = np.geomspace(Umu4Sq_range[0], Umu4Sq_range[1], Npoints[2])
    g_Vec = gfixed

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


def run_scan_gfixed_Ue4SQRfixed_2D(
    kwargs,
    filename,
    gfixed=2.5,
    Ue4SQRfixed=0.05,
    Npoints=10,
    path_results="fit_data/",
    dmSq_range=(0.5e-1, 2e4),
    Umu4Sq_range=(2e-4, 0.25),
):

    # Range of mixings scanned
    dm_Vec = np.geomspace(np.sqrt(dmSq_range[0]), np.sqrt(dmSq_range[1]), Npoints)
    Umu4Sq = np.geomspace(Umu4Sq_range[0], Umu4Sq_range[1], Npoints)

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


def run_scan_gfixed_dmfixed_2D(
    kwargs,
    filename,
    gfixed=2.5,
    dmSq_fixed=1e8,
    Npoints=10,
    path_results="fit_data/",
    Ue4Sq_range=(1e-3, 0.25),
    Umu4Sq_range=(2e-4, 0.25),
):

    # Range of mixings scanned
    Ue4Sq = np.geomspace(Ue4Sq_range[0], Ue4Sq_range[1], Npoints)
    Umu4Sq = np.geomspace(Umu4Sq_range[0], Umu4Sq_range[1], Npoints)

    # Cartesian product of grid -- already imposes unitarity and pertubatirbity of g
    paramlist = param_scan.create_grid_of_params(
        g=gfixed, m4=np.sqrt(dmSq_fixed), Ue4Sq=Ue4Sq, Um4Sq=Umu4Sq
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

kwargs_osconly = {
    "oscillations": True,
    "decay": False,
    "decouple_decay": False,
    "energy_degradation": False,
    "include_antineutrinos": True,
    "n_replications": 1,
}

# Full and Standard case
kwargs_std = {
    "disappearance": True,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    **kwargs_common,
}

# Full and Standard case
kwargs_std_osconly = {
    "disappearance": True,
    "use_numu_MC": True,
    "undo_numu_normalization": False,
    **kwargs_osconly,
}


if __name__ == "__main__":

    ###################
    MOCK_SCAN = False
    N_MOCK = 5

    # # 4D scans
    # n = 30 if not MOCK_SCAN else N_MOCK
    # _ = run_scan_4D(kwargs_std, f"MH_decay_4D_{n}", Npoints=n)

    # n = 40 if not MOCK_SCAN else N_MOCK
    # # 3D scans
    # _ = run_scan_gfixed_3D(
    #     kwargs_std, f"MH_decay_gfixed_2.5_3D_{n}", Npoints=n, gfixed=2.5
    # )
    # _ = run_scan_gfixed_3D(
    #     kwargs_std, f"MH_decay_gfixed_1.0_3D_{n}", Npoints=n, gfixed=1.0
    # )

    n = 60 if not MOCK_SCAN else N_MOCK
    # kwargs_std["helicity"] = "flipping"
    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_2.5_Ue4SQRfixed_0.05_2D_{n}_flipping",
    #     Npoints=n,
    #     gfixed=2.5,
    #     Ue4SQRfixed=0.05,
    # )

    UE4SQR_IceCube = 0.05
    # g2_over_pi_values = np.linspace(0, 4, 9, endpoint=True)
    # 2D scans
    # for g2op in g2_over_pi_values:
    #     if g2op > 0:
    #         _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #             kwargs_std,
    #             f"MH_decay_goverpi_{g2op:.2f}_Ue4SQRfixed_{UE4SQR_IceCube:.3f}_2D_{n}",
    #             Npoints=n,
    #             gfixed=np.sqrt(g2op * np.pi),
    #             Ue4SQRfixed=UE4SQR_IceCube,
    #         )

    _ = run_scan_gfixed_Ue4SQRfixed_2D(
        kwargs_std_osconly,
        f"MH_decay_goverpi_0_Ue4SQRfixed_{UE4SQR_IceCube:.3f}_2D_{n}",
        Npoints=n,
        gfixed=0,
        Ue4SQRfixed=UE4SQR_IceCube,
    )

    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_2.5_Ue4SQRfixed_0.05_2D_{n}",
    #     Npoints=n,
    #     gfixed=2.5,
    #     Ue4SQRfixed=0.05,
    # )
    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_2.5_Ue4SQRfixed_0.01_2D_{n}",
    #     Npoints=n,
    #     gfixed=2.5,
    #     Ue4SQRfixed=0.01,
    # )

    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_1_Ue4SQRfixed_0.10_2D_{n}",
    #     Npoints=n,
    #     gfixed=1,
    #     Ue4SQRfixed=0.10,
    # )

    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_1_Ue4SQRfixed_0.05_2D_{n}",
    #     Npoints=n,
    #     gfixed=1,
    #     Ue4SQRfixed=0.05,
    # )
    # _ = run_scan_gfixed_Ue4SQRfixed_2D(
    #     kwargs_std,
    #     f"MH_decay_gfixed_1_Ue4SQRfixed_0.01_2D_{n}",
    #     Npoints=n,
    #     gfixed=1,
    #     Ue4SQRfixed=0.01,
    # )
