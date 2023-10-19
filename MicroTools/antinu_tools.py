import numpy as np
from scipy.interpolate import interp1d

from pathlib import Path

local_dir = Path(__file__).parent


def get_reweighter_nu_to_antinu_Enu_1D(generator="GENIE_v3_02_00"):
    # these are sigma_nu / sigma_nubar ratios
    enu, R = np.genfromtxt(
        local_dir / f"antinus_data/{generator}_nubar_ratio.dat", unpack=True
    )

    return interp1d(enu, 1 / R, fill_value=1.0, bounds_error=False)
