import os
import uproot
import numpy as np

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

mc_path = os.path.join(base_path, "mc/")
data_path = os.path.join(base_path, "data/")

nue_energy_bins = np.array(
    [
        0.200,
        0.250,
        0.300,
        0.375,
        0.475,
        0.550,
        0.600,
        0.675,
        0.750,
        0.800,
        0.950,
        1.100,
        1.150,
        1.250,
        1.300,
        1.500,
        1.700,
        1.900,
        3.000,
    ]
)
numu_energy_bins = np.array(
    [
        0.0,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        3.0,
    ]
)


def load_data_sample(f, tree):
    energy = np.array(f[tree + "/energy"].array())

    mc = np.empty(
        len(energy),
        dtype=[
            ("ntuple_energy", energy.dtype),
        ],
    )

    mc["ntuple_energy"] = energy

    return mc


nue_tree = "h55"
numu_tree = "h56"

sample_ids = {
    "nue": 0,
    "nuebar": 1,
    "numu": 2,
    "numubar": 3,
}


def load_data():
    all_data = []

    with uproot.open(os.path.join(data_path, "neutrino_mode_nue_sample.root")) as f:
        data = load_data_sample(f, nue_tree)

        sample_id = np.full(len(data), sample_ids["nue"])

        dtype = list(data.dtype.descr)
        dtype.append(("sample_id", sample_id.dtype))
        new_data = np.empty(len(sample_id), dtype=dtype)
        for name in data.dtype.names:
            new_data[name] = data[name]
        new_data["sample_id"] = sample_id
        all_data.append(new_data)

    with uproot.open(os.path.join(data_path, "antineutrino_mode_nue_sample.root")) as f:
        data = load_data_sample(f, nue_tree)

        sample_id = np.full(len(data), sample_ids["nuebar"])

        dtype = list(data.dtype.descr)
        dtype.append(("sample_id", sample_id.dtype))
        new_data = np.empty(len(sample_id), dtype=dtype)
        for name in data.dtype.names:
            new_data[name] = data[name]
        new_data["sample_id"] = sample_id
        all_data.append(new_data)

    with uproot.open(os.path.join(data_path, "neutrino_mode_numu_sample.root")) as f:
        data = load_data_sample(f, numu_tree)

        sample_id = np.full(len(data), sample_ids["numu"])

        dtype = list(data.dtype.descr)
        dtype.append(("sample_id", sample_id.dtype))
        new_data = np.empty(len(sample_id), dtype=dtype)
        for name in data.dtype.names:
            new_data[name] = data[name]
        new_data["sample_id"] = sample_id
        all_data.append(new_data)

    with uproot.open(
        os.path.join(data_path, "antineutrino_mode_numu_sample.root")
    ) as f:
        data = load_data_sample(f, numu_tree)

        sample_id = np.full(len(data), sample_ids["numubar"])

        dtype = list(data.dtype.descr)
        dtype.append(("sample_id", sample_id.dtype))
        new_data = np.empty(len(sample_id), dtype=dtype)
        for name in data.dtype.names:
            new_data[name] = data[name]
        new_data["sample_id"] = sample_id
        all_data.append(new_data)

    return all_data


data = load_data()
