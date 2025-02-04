import numpy as np
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

# For more convenient imports
# from . import antinu_tools
# from . import const
# from . import plot_tools
# from . import sterile_tools
# from . import unfolder
# from . import mini_tools
# from . import apps


local_dir = Path(__file__).parent

MeVToGeV = 1.0e-3

##################################################################
# Our oscillation results and other oscillation limits
path_plots = f"plots/"
path_osc_data = f"{local_dir}/osc_data/"
path_osc_app = f"{path_osc_data}/numu_to_nue/"
path_osc_numudis = f"{path_osc_data}/numu_dis/"
path_osc_nuedis = f"{path_osc_data}/nue_dis/"

L_micro = 0.4685  # MicroBooNE Baseline length in kilometers
L_mini = 0.545  # MiniBooNE Baseline length in kilometers
L_SBND = 0.110  # MicroBooNE Baseline length in kilometers
L_ICARUS = 0.600  # MiniBooNE Baseline length in kilometers

Mass_mini = 818
Mass_micro = 85
Mass_ICARUS = 476
Mass_SBND = 112

# MiniBooNE POTs in 1e20 units
mini_POTs = {"2012": 6.46, "2018": 12.84, "2020": 18.75}
