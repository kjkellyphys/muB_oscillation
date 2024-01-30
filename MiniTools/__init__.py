import numpy as np
from importlib.resources import open_text

from MiniTools import fit

# bin_edges_reco = np.genfromtxt(
#     open_text("MiniTools.include.miniboone_2020", "Enu_bin_edges.dat")
# )
# bin_centers_reco = bin_edges_reco[:-1] + np.diff(bin_edges_reco) / 2.0
# bin_width_reco = np.diff(bin_edges_reco)
