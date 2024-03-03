import numpy as np
from importlib.resources import open_text
from MiniTools import fit
from MiniTools import apps

MC_nue_bkg_tot = np.genfromtxt(
    open_text(
        f"MiniTools.include.MB_data_release_2020.fhcmode",
        f"miniboone_nuebgr_lowe.txt",
    )
)
MC_numu_bkg_tot = np.genfromtxt(
    open_text(
        f"MiniTools.include.MB_data_release_2020.fhcmode",
        f"miniboone_numu.txt",
    )
)
