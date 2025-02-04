import numpy as np
from scipy.interpolate import interp1d

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def get_reweighter_nu_to_antinu_Enu_1D(generator="GENIE_v3_02_00"):
    # these are sigma_nu / sigma_nubar ratios
    enu, R = np.genfromtxt(
        files("OscTools.include.antinus_data")
        .joinpath(f"{generator}_nubar_ratio.dat")
        .open(),
        unpack=True,
    )

    return interp1d(enu, 1 / R, fill_value=1.0, bounds_error=False)
