import os
import uproot
import numpy as np

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

mc_path = os.path.join(base_path, "mc/")
data_path = os.path.join(base_path, "data/")

livetimes = {
    "nu_data": 18.75e20,
    "nubar_data": 11.27e20,
    "nue_osc": 1.327999038e20,
    "nue_intrinsic": 4.108001152e21,
    "numu_intrinsic": 4.108001152e21,
    "nuebar_osc": 3.249e21,
    "nuebar_intrinsic": 2.79e22,
    "numubar_intrinsic": 2.79e22,
}


def load_sample(f, tree):
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

    return mc


nue_tree = "h55"
numu_tree = "h56"


def load_mc():
    all_mc = []

    with uproot.open(os.path.join(mc_path, "neutrino_mode_nue_sample.root")) as f:
        mc = load_sample(f, nue_tree)
        osc_pot_scale = livetimes["nu_data"] / livetimes["nue_osc"]
        intrinsic_pot_scale = livetimes["nu_data"] / livetimes["nue_intrinsic"]

        iflux = mc["ntuple_iflux"]

        osc_mask = iflux == 11
        intrinsic_mask = iflux == 14

        wgt = mc["ntuple_wgt"]
        cv_weight = np.copy(wgt)
        cv_weight[osc_mask] *= osc_pot_scale
        cv_weight[intrinsic_mask] *= intrinsic_pot_scale

        good_mask = np.logical_or(osc_mask, intrinsic_mask)

        dtype = list(mc.dtype.descr)
        dtype.append(("cv_weight", cv_weight.dtype))
        new_mc = np.empty(len(cv_weight), dtype=dtype)
        for name in mc.dtype.names:
            new_mc[name] = mc[name]
        new_mc["cv_weight"] = cv_weight
        new_mc = new_mc[good_mask]
        all_mc.append(new_mc)

    with uproot.open(os.path.join(mc_path, "antineutrino_mode_nue_sample.root")) as f:
        mc = load_sample(f, nue_tree)
        osc_pot_scale = livetimes["nubar_data"] / livetimes["nuebar_osc"]
        intrinsic_pot_scale = livetimes["nubar_data"] / livetimes["nuebar_intrinsic"]

        iflux = mc["ntuple_iflux"]

        osc_mask = iflux == 11
        intrinsic_mask = iflux == 14

        wgt = mc["ntuple_wgt"]
        cv_weight = np.copy(wgt)
        cv_weight[osc_mask] *= osc_pot_scale
        cv_weight[intrinsic_mask] *= intrinsic_pot_scale

        good_mask = np.logical_or(osc_mask, intrinsic_mask)

        dtype = list(mc.dtype.descr)
        dtype.append(("cv_weight", cv_weight.dtype))
        new_mc = np.empty(len(cv_weight), dtype=dtype)
        for name in mc.dtype.names:
            new_mc[name] = mc[name]
        new_mc["cv_weight"] = cv_weight
        new_mc = new_mc[good_mask]
        all_mc.append(new_mc)

    with uproot.open(os.path.join(mc_path, "neutrino_mode_numu_sample.root")) as f:
        mc = load_sample(f, numu_tree)
        intrinsic_pot_scale = livetimes["nu_data"] / livetimes["numu_intrinsic"]

        iflux = mc["ntuple_iflux"]

        wgt = mc["ntuple_wgt"]
        cv_weight = np.copy(wgt)
        cv_weight *= intrinsic_pot_scale

        dtype = list(mc.dtype.descr)
        dtype.append(("cv_weight", cv_weight.dtype))
        new_mc = np.empty(len(cv_weight), dtype=dtype)
        for name in mc.dtype.names:
            new_mc[name] = mc[name]
        new_mc["cv_weight"] = cv_weight
        all_mc.append(new_mc)

    with uproot.open(os.path.join(mc_path, "antineutrino_mode_numu_sample.root")) as f:
        mc = load_sample(f, numu_tree)
        intrinsic_pot_scale = livetimes["nubar_data"] / livetimes["nuebar_intrinsic"]

        iflux = mc["ntuple_iflux"]

        wgt = mc["ntuple_wgt"]
        cv_weight = np.copy(wgt)
        cv_weight *= intrinsic_pot_scale

        dtype = list(mc.dtype.descr)
        dtype.append(("cv_weight", cv_weight.dtype))
        new_mc = np.empty(len(cv_weight), dtype=dtype)
        for name in mc.dtype.names:
            new_mc[name] = mc[name]
        new_mc["cv_weight"] = cv_weight
        all_mc.append(new_mc)

    return all_mc


all_mc = load_mc()
