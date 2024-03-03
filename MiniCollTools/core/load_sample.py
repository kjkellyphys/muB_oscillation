import numpy as np
import uproot
import os
import os.path
from . import sample_info

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
mc_path = os.path.join(base_path, "mc/")
data_path = os.path.join(base_path, "data/")
cov_path = os.path.join(base_path, "cov/")
dl_mc_path = os.path.join(base_path, "dl_mc/")
dl_data_path = os.path.join(base_path, "dl_data/")
dl_cov_path = os.path.join(base_path, "dl_cov/")
wc_mc_path = os.path.join(base_path, "wc_mc/")
wc_data_path = os.path.join(base_path, "wc_data/")
wc_cov_path = os.path.join(base_path, "wc_cov/")


def load_data_sample(f, nu):
    tree = sample_info.tree_names[nu]
    energy = np.array(f[tree + "/energy"].array())

    data = np.empty(
        len(energy),
        dtype=[
            ("ntuple_energy", energy.dtype),
        ],
    )

    data["ntuple_energy"] = energy

    sample_id = np.full(len(data), sample_info.sample_ids[nu])

    dtype = list(data.dtype.descr)
    dtype.append(("sample_id", sample_id.dtype))
    new_data = np.empty(len(sample_id), dtype=dtype)
    for name in data.dtype.names:
        new_data[name] = data[name]
    new_data["sample_id"] = sample_id

    return new_data


def load_data():
    all_data = []

    with uproot.open(os.path.join(data_path, "neutrino_mode_nue_sample.root")) as f:
        all_data.append(load_data_sample(f, "nue"))

    with uproot.open(os.path.join(data_path, "neutrino_mode_numu_sample.root")) as f:
        all_data.append(load_data_sample(f, "numu"))

    with uproot.open(os.path.join(data_path, "antineutrino_mode_nue_sample.root")) as f:
        all_data.append(load_data_sample(f, "nuebar"))

    with uproot.open(
        os.path.join(data_path, "antineutrino_mode_numu_sample.root")
    ) as f:
        all_data.append(load_data_sample(f, "numubar"))

    return np.concatenate(all_data)


def load_mc_sample(f, nu):
    tree = sample_info.tree_names[nu]

    iflux = np.array(f[tree + "/iflux"].array()).round().astype(int)
    ibkgd = np.array(f[tree + "/ibkgd"].array()).round().astype(int)
    nuchan = np.array(f[tree + "/nuchan"].array()).round().astype(int)
    inno = np.array(f[tree + "/inno"].array()).round().astype(int)
    enugen = np.array(f[tree + "/enugen"].array())
    energy = np.array(f[tree + "/energy"].array())
    nuleng = np.array(f[tree + "/nuleng"].array())
    parid = np.array(f[tree + "/parid"].array()).round().astype(int)
    wgt = np.array(f[tree + "/wgt"].array())
    ispi0 = np.array(f[tree + "/ispi0"].array()).round().astype(bool)
    isdirt = np.array(f[tree + "/isdirt"].array()).round().astype(bool)

    mc = np.empty(
        len(iflux),
        dtype=[
            ("ntuple_iflux", iflux.dtype),
            ("ntuple_ibkgd", ibkgd.dtype),
            ("ntuple_nuchan", nuchan.dtype),
            ("ntuple_inno", inno.dtype),
            ("ntuple_enugen", enugen.dtype),
            ("ntuple_energy", energy.dtype),
            ("ntuple_nuleng", nuleng.dtype),
            ("ntuple_parid", parid.dtype),
            ("ntuple_wgt", wgt.dtype),
            ("ntuple_ispi0", ispi0.dtype),
            ("ntuple_isdirt", isdirt.dtype),
        ],
    )

    mc["ntuple_iflux"] = iflux
    mc["ntuple_ibkgd"] = ibkgd
    mc["ntuple_nuchan"] = nuchan
    mc["ntuple_inno"] = inno
    mc["ntuple_enugen"] = enugen
    mc["ntuple_energy"] = energy
    mc["ntuple_nuleng"] = nuleng
    mc["ntuple_parid"] = parid
    mc["ntuple_wgt"] = wgt
    mc["ntuple_ispi0"] = ispi0
    mc["ntuple_isdirt"] = isdirt

    sample_id = sample_info.sample_ids[nu]

    iflux = mc["ntuple_iflux"]
    wgt = mc["ntuple_wgt"]
    cv_weight = np.copy(wgt)

    if nu == "nue" or nu == "numu":
        data_livetime = sample_info.livetimes["nu_data"]
    else:
        data_livetime = sample_info.livetimes["nubar_data"]

    if (
        sample_id == sample_info.sample_ids["nue"]
        or sample_id == sample_info.sample_ids["nuebar"]
    ):
        osc_pot_scale = data_livetime / sample_info.livetimes[nu + "_osc"]
        intrinsic_pot_scale = data_livetime / sample_info.livetimes[nu + "_intrinsic"]
        osc_mask = iflux == 11
        intrinsic_mask = iflux == 14
        cv_weight[osc_mask] *= osc_pot_scale
        cv_weight[intrinsic_mask] *= intrinsic_pot_scale
        good_mask = np.logical_or(osc_mask, intrinsic_mask)
    else:
        intrinsic_pot_scale = data_livetime / sample_info.livetimes[nu + "_intrinsic"]
        cv_weight *= intrinsic_pot_scale
        good_mask = np.ones(len(mc)).astype(bool)

    cv_weight[cv_weight < 0] = 0

    sample_ids = np.full(len(mc), sample_id)
    dtype = list(mc.dtype.descr)
    dtype.append(("cv_weight", cv_weight.dtype))
    dtype.append(("sample_id", sample_ids.dtype))
    new_mc = np.empty(len(cv_weight), dtype=dtype)
    for name in mc.dtype.names:
        new_mc[name] = mc[name]
    new_mc["cv_weight"] = cv_weight
    new_mc["sample_id"] = sample_ids
    new_mc = new_mc[good_mask]

    return new_mc


def load_mc():
    all_mc = {}

    with uproot.open(os.path.join(mc_path, "neutrino_mode_nue_sample.root")) as f:
        all_mc["nue"] = load_mc_sample(f, "nue")

    with uproot.open(os.path.join(mc_path, "neutrino_mode_numu_sample.root")) as f:
        all_mc["numu"] = load_mc_sample(f, "numu")

    with uproot.open(os.path.join(mc_path, "antineutrino_mode_nue_sample.root")) as f:
        all_mc["nuebar"] = load_mc_sample(f, "nuebar")

    with uproot.open(os.path.join(mc_path, "antineutrino_mode_numu_sample.root")) as f:
        all_mc["numubar"] = load_mc_sample(f, "numubar")

    return all_mc


def load_dl_templates():
    all_templates = {}

    for key in [
        "tot_constrained_bkg",
        "nue_nominal_bkg",
        "numu_fitted_bkg",
        "tot_constrained_lee",
    ]:
        all_templates[key] = np.loadtxt(
            os.path.join(dl_mc_path, "nue_1e1p_" + key + "_pred.txt")
        )
    all_templates["nue_constrained_bkg"] = (
        all_templates["tot_constrained_bkg"] - all_templates["numu_fitted_bkg"]
    )
    all_templates["tot_nominal_bkg"] = (
        all_templates["nue_nominal_bkg"] + all_templates["numu_fitted_bkg"]
    )

    return all_templates


def load_wc_templates():
    all_templates = {}

    for ds in ["nue_FC_Constr", "nue_FC", "nue_PC", "numu_FC", "numu_PC"]:
        for tkey in ["bkg", "sig_bkg"]:
            key = ds + "_" + tkey
            all_templates[key] = np.loadtxt(os.path.join(wc_mc_path, key + ".txt"))
        all_templates[ds + "_sig"] = (
            all_templates[ds + "_sig_bkg"] - all_templates[ds + "_bkg"]
        )

    for tkey in ["bkg", "sig", "sig_bkg"]:
        all_templates["nue_numu_" + tkey] = np.concatenate(
            (
                all_templates["nue_FC_" + tkey],
                all_templates["nue_PC_" + tkey],
                all_templates["numu_PC_" + tkey],
                all_templates["numu_PC_" + tkey],
            )
        )
    return all_templates


def load_dl_binned_data():
    return np.loadtxt(os.path.join(dl_data_path, "nue_1e1p_data.txt"))


def load_wc_binned_data():
    all_templates = {}
    for ds in ["nue_FC_Constr", "nue_FC", "nue_PC", "numu_FC", "numu_PC"]:
        key = ds + "_data"
        all_templates[key] = np.loadtxt(os.path.join(wc_data_path, key + ".txt"))
    all_templates["nue_numu_data"] = np.concatenate(
        (
            all_templates["nue_FC_data"],
            all_templates["nue_PC_data"],
            all_templates["numu_FC_data"],
            all_templates["numu_PC_data"],
        )
    )
    return all_templates


def load_dl_bin_edges():
    return np.loadtxt(dl_mc_path + "nue_1e1p_bins.txt") / 1e3  # Convert from MeV to GeV


def load_wc_bin_edges():
    return np.loadtxt(wc_data_path + "Ereco_bins.txt")


def load_dl_nue_mc():
    reco_energy, true_energy, weight = np.loadtxt(dl_mc_path + "nue_eventlist.txt").T
    weight[weight < 0] = 0
    mc = np.empty(
        len(reco_energy),
        dtype=[
            ("reco_energy", reco_energy.dtype),
            ("true_energy", true_energy.dtype),
            ("cv_weight", weight.dtype),
        ],
    )
    mc["reco_energy"] = reco_energy / 1e3  # Convert from MeV to GeV
    mc["true_energy"] = true_energy / 1e3  # Convert from MeV to GeV
    mc["cv_weight"] = weight
    return mc


def load_wc_smearing_matrices(true_bins, reco_bins):
    smearing_matrices = {}
    for ds in ["nue_FC", "nue_PC", "numu_FC", "numu_PC"]:
        mtx = np.loadtxt(os.path.join(wc_mc_path, ds + "_smearing_matrix.txt"))
        smearing_matrices[ds] = mtx[: len(reco_bins) - 1, : len(true_bins) - 1]
    return smearing_matrices


if __name__ == "__main__":
    mc = load_mc()
    data = load_data()
