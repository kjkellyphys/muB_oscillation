import numpy as np
import sample_info
import functools
from numba import njit

original_energy_bins = {
    "nue": np.array(
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
    ),
    "numu": np.array(
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
    ),
}
original_energy_bins["nuebar"] = original_energy_bins["nue"]
original_energy_bins["numubar"] = original_energy_bins["numu"]


def rebin_edges(rebin, edges):
    rebin = np.asarray(rebin)
    mask = rebin[:-1] != rebin[1:]
    new_edges = np.concatenate([[edges[0]], edges[1:-1][mask], [edges[-1]]])
    return new_edges


nu_rebins = {
    "nue": np.array(
        #  [1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 9, 10, 11, 11, 11]
        [1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 11, 11]
    ),
    "numu": np.array(
        [-1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8, -8, 0]
    ),
}
nu_rebins["nuebar"] = nu_rebins["nue"]
nu_rebins["numubar"] = nu_rebins["numu"]

rebin_components = [
    ("osc", "nue"),
    ("intrinsic", "nue"),
    ("intrinsic", "numu"),
    ("osc", "nuebar"),
    ("intrinsic", "nuebar"),
    ("intrinsic", "numubar"),
]


nu_samples = ["nue", "numu", "nuebar", "numubar", "zero"]


def masks_by_sample(mc):
    masks = {}
    for nu, sample_id in sample_info.sample_ids.items():
        masks[nu] = mc["sample_id"] == sample_id
    return masks


def masks_by_component(mc, is_data=False):
    if is_data:
        return {"data": np.ones(len(mc)).astype(bool)}
    masks = {}
    iflux = mc["ntuple_iflux"]
    masks["osc"] = iflux == 11
    masks["intrinsic"] = np.logical_or(iflux == 14, iflux == 16)
    return masks


def masks_by_energy(sample, sample_masks):
    masks = {}
    for nu, sample_id in sample_info.sample_ids.items():
        mapping = np.digitize(sample["ntuple_energy"], original_energy_bins[nu]) - 1
        masks[nu] = np.array(
            [
                np.logical_and(mapping == i, sample_masks[nu])
                for i in range(len(original_energy_bins[nu]) - 1)
            ]
        )
    return masks


def sort_events(events, is_data=False):
    raw_ids = []

    sorted_events = np.empty(np.shape(events), dtype=events.dtype)
    raw_slices = []
    rebinned_slices = []
    collapsed_slices = []
    raw_to_rebin_slices = []
    rebin_to_collapsed_slices = []
    sample_masks = masks_by_sample(events)
    component_masks = masks_by_component(events, is_data=is_data)
    zero_mask = np.zeros(len(events)).astype(bool)
    samples = ["nue", "numu", "nuebar", "numubar"]
    if is_data:
        sample_components = {
            "nue": ["data"],
            "numu": ["data"],
            "nuebar": ["data"],
            "numubar": ["data"],
        }
    else:
        sample_components = {
            "nue": ["osc", "intrinsic"],
            "numu": ["intrinsic"],
            "nuebar": ["osc", "intrinsic"],
            "numubar": ["intrinsic"],
        }

    for sample in samples:
        rebin = nu_rebins[sample]
        zero_bins = np.ix_(rebin == 0)[0]
        zero_low_edge = original_energy_bins[sample][:-1][zero_bins]
        zero_high_edge = original_energy_bins[sample][1:][zero_bins]

        zero_mask = np.logical_and(
            sample_masks[sample],
            np.any(
                np.logical_and(
                    events["ntuple_energy"][:, None] >= zero_low_edge[None, :],
                    events["ntuple_energy"][:, None] < zero_high_edge[None, :],
                ),
                axis=1,
            ),
        )
    for sample in samples:
        sample_masks[sample] &= ~zero_mask
    sample_masks["zero"] = zero_mask

    offset = 0
    raw_count = 0
    rebinned_count = 0
    collapsed_count = 0
    for sample in samples:
        sample_mask = sample_masks[sample]

        rebin = nu_rebins[sample]
        nonzero = rebin != 0
        rebin_u, rebin_inv = np.unique(rebin[nonzero], return_inverse=True)
        i = np.argsort(np.abs(rebin_u))
        rebin_u = rebin_u[i]
        i = np.argsort(i)
        rebin_inv = i[rebin_inv]
        low_edges = original_energy_bins[sample][:-1]
        high_edges = original_energy_bins[sample][1:]

        super_energy_bins = {}
        super_energy_masks = {}
        sample_events = events[sample_mask]
        sample_events_energy = sample_events["ntuple_energy"]
        for i in range(len(rebin_inv)):
            idx = i
            res_idx = rebin_inv[i]
            if res_idx < 0:
                continue
            if res_idx not in super_energy_bins:
                super_energy_masks[res_idx] = False
                super_energy_bins[res_idx] = []
            super_energy_bins[res_idx].append(
                (idx, low_edges[nonzero][idx], high_edges[nonzero][idx])
            )
            super_energy_masks[res_idx] |= np.logical_and(
                sample_events_energy >= low_edges[nonzero][idx],
                sample_events_energy < high_edges[nonzero][idx],
            )

        for super_e_idx in sorted(super_energy_masks.keys()):
            super_e_mask = super_energy_masks[super_e_idx]
            super_e_events = sample_events[super_e_mask]
            collapsed_offset_start = offset
            rebinned_count_start = rebinned_count
            for component in sample_components[sample]:
                component_mask = component_masks[component][sample_mask][super_e_mask]
                component_events = super_e_events[component_mask]
                component_events_energy = component_events["ntuple_energy"]
                rebinned_offset_start = offset
                raw_count_start = raw_count
                for idx, low_e, high_e in super_energy_bins[super_e_idx]:
                    energy_mask = np.logical_and(
                        component_events_energy >= low_e,
                        component_events_energy < high_e,
                    )
                    n = np.count_nonzero(energy_mask)
                    sorted_events[offset : offset + n] = component_events[energy_mask]
                    raw_ids.append((sample, component, idx))
                    raw_count += 1
                    raw_slices.append(slice(offset, offset + n))
                    offset += n
                raw_to_rebin_slices.append(slice(raw_count_start, raw_count))
                rebinned_count += 1
                rebinned_slices.append(slice(rebinned_offset_start, offset))
            rebin_to_collapsed_slices.append(
                slice(rebinned_count_start, rebinned_count)
            )
            collapsed_count += 1
            collapsed_slices.append(slice(collapsed_offset_start, offset))

    raw_count_start = raw_count
    rebinned_offset_start = offset
    rebinned_count_start = rebinned_count
    collapsed_offset_start = offset
    for sample in samples:
        rebin = nu_rebins[sample]
        zero = rebin == 0
        rebin_u, rebin_inv = np.unique(rebin[zero], return_inverse=True)
        i = np.argsort(np.abs(rebin_u))
        rebin_u = rebin_u[i]
        i = np.argsort(i)
        rebin_inv = i[rebin_inv]
        low_edges = original_energy_bins[sample][:-1]
        high_edges = original_energy_bins[sample][1:]
        zero_idxs = np.ix_(zero)[0]

        for component in sample_components[sample]:
            component_mask = component_masks[component][sample_mask][
                zero_mask[sample_mask]
            ]
            component_events = super_e_events[component_mask]
            component_events_energy = component_events["ntuple_energy"]
            for idx in zero_idxs:
                low_e = low_edges[idx]
                high_e = high_edges[idx]
                energy_mask = np.logical_and(
                    component_events_energy >= low_e, component_events_energy < high_e
                )
                n = np.count_nonzero(energy_mask)
                sorted_events[offset : offset + n] = component_events[energy_mask]
                raw_ids.append((sample, component, idx))
                raw_count += 1
                raw_slices.append(slice(offset, offset + n))
                offset += n
    raw_to_rebin_slices.append(slice(raw_count_start, raw_count))
    rebinned_count += 1
    rebinned_slices.append(slice(rebinned_offset_start, offset))
    rebin_to_collapsed_slices.append(slice(rebinned_count_start, rebinned_count))
    collapsed_count += 1
    collapsed_slices.append(slice(collapsed_offset_start, offset))

    events_slices = {
        "raw": raw_slices,
        "rebinned": rebinned_slices,
        "collapsed": collapsed_slices,
    }
    transform_slices = {
        "raw_to_rebin": raw_to_rebin_slices,
        "rebin_to_collapsed": rebin_to_collapsed_slices,
    }

    raw_id_dict = dict(zip(raw_ids, range(len(raw_ids))))
    raw_id_order = []

    for sample in samples:
        for component in sample_components[sample]:
            for i in range(len(original_energy_bins[sample]) - 1):
                new_idx = raw_id_dict[(sample, component, i)]
                raw_id_order.append(new_idx)
    raw_id_order = np.array(raw_id_order)
    orig_idx_order = np.empty(len(raw_id_order), dtype=int)
    orig_idx_order[raw_id_order] = np.arange(len(raw_id_order), dtype=int)

    raw_osc = np.array([component == "osc" for sample, component, idx in raw_ids])
    raw_intrinsic = np.array(
        [component == "intrinsic" for sample, component, idx in raw_ids]
    )
    raw_nue = np.array([sample == "nue" for sample, component, idx in raw_ids])
    raw_numu = np.array([sample == "numu" for sample, component, idx in raw_ids])
    raw_nuebar = np.array([sample == "nuebar" for sample, component, idx in raw_ids])
    raw_numubar = np.array([sample == "numubar" for sample, component, idx in raw_ids])
    raw_masks = {
        "osc": raw_osc,
        "intrinsic": raw_intrinsic,
        "nue": raw_nue,
        "numu": raw_numu,
        "nuebar": raw_nuebar,
        "numubar": raw_numubar,
    }
    raw_masks = {("raw", key): value for key, value in raw_masks.items()}
    rebinned_masks = {
        ("rebinned", key[1]): np.array(
            [np.any(raw[slc]) for slc in transform_slices["raw_to_rebin"]]
        )
        for key, raw in raw_masks.items()
    }
    collapsed_masks = {
        ("collapsed", key[1]): np.array(
            [np.any(raw[slc]) for slc in transform_slices["rebin_to_collapsed"]]
        )
        for key, raw in raw_masks.items()
    }
    component_bin_masks = {}
    component_bin_masks.update(raw_masks)
    component_bin_masks.update(rebinned_masks)
    component_bin_masks.update(collapsed_masks)

    return (
        sorted_events,
        events_slices,
        transform_slices,
        orig_idx_order,
        component_bin_masks,
    )


def bin_transforms(is_data=False):
    raw_ids = []

    raw_to_rebin_slices = []
    rebin_to_collapsed_slices = []
    samples = ["nue", "numu", "nuebar", "numubar"]
    if is_data:
        sample_components = {
            "nue": ["data"],
            "numu": ["data"],
            "nuebar": ["data"],
            "numubar": ["data"],
        }
    else:
        sample_components = {
            "nue": ["osc", "intrinsic"],
            "numu": ["intrinsic"],
            "nuebar": ["osc", "intrinsic"],
            "numubar": ["intrinsic"],
        }

    raw_count = 0
    rebinned_count = 0
    collapsed_count = 0
    for sample in samples:
        rebin = nu_rebins[sample]
        nonzero = rebin != 0
        rebin_u, rebin_inv = np.unique(rebin[nonzero], return_inverse=True)
        i = np.argsort(np.abs(rebin_u))
        rebin_u = rebin_u[i]
        i = np.argsort(i)
        rebin_inv = i[rebin_inv]
        low_edges = original_energy_bins[sample][:-1]
        high_edges = original_energy_bins[sample][1:]

        super_energy_bins = {}
        for i in range(len(rebin_inv)):
            idx = i
            res_idx = rebin_inv[i]
            if res_idx < 0:
                continue
            if res_idx not in super_energy_bins:
                super_energy_bins[res_idx] = []
            super_energy_bins[res_idx].append(
                (idx, low_edges[nonzero][idx], high_edges[nonzero][idx])
            )

        for super_e_idx in sorted(super_energy_bins.keys()):
            rebinned_count_start = rebinned_count
            for component in sample_components[sample]:
                raw_count_start = raw_count
                for idx, low_e, high_e in super_energy_bins[super_e_idx]:
                    raw_ids.append((sample, component, idx, low_e, high_e))
                    raw_count += 1
                raw_to_rebin_slices.append(slice(raw_count_start, raw_count))
                rebinned_count += 1
            rebin_to_collapsed_slices.append(
                slice(rebinned_count_start, rebinned_count)
            )
            collapsed_count += 1

    raw_count_start = raw_count
    rebinned_count_start = rebinned_count
    for sample in samples:
        rebin = nu_rebins[sample]
        zero = rebin == 0
        rebin_u, rebin_inv = np.unique(rebin[zero], return_inverse=True)
        i = np.argsort(np.abs(rebin_u))
        rebin_u = rebin_u[i]
        i = np.argsort(i)
        rebin_inv = i[rebin_inv]
        low_edges = original_energy_bins[sample][:-1]
        high_edges = original_energy_bins[sample][1:]
        zero_idxs = np.ix_(zero)[0]

        for component in sample_components[sample]:
            for idx in zero_idxs:
                low_e = low_edges[idx]
                high_e = high_edges[idx]
                raw_ids.append((sample, component, idx, low_e, high_e))
                raw_count += 1
    raw_to_rebin_slices.append(slice(raw_count_start, raw_count))
    rebinned_count += 1
    rebin_to_collapsed_slices.append(slice(rebinned_count_start, rebinned_count))
    collapsed_count += 1

    transform_slices = {
        "raw_to_rebin": tuple(raw_to_rebin_slices),
        "rebin_to_collapsed": tuple(rebin_to_collapsed_slices),
    }

    raw_id_dict = dict(zip([r[:3] for r in raw_ids], range(len(raw_ids))))
    raw_id_order = []

    for sample in samples:
        for component in sample_components[sample]:
            for i in range(len(original_energy_bins[sample]) - 1):
                new_idx = raw_id_dict[(sample, component, i)]
                raw_id_order.append(new_idx)
    raw_id_order = np.array(raw_id_order)
    orig_idx_order = np.empty(len(raw_id_order), dtype=int)
    orig_idx_order[raw_id_order] = np.arange(len(raw_id_order), dtype=int)

    raw_osc = np.array([component == "osc" for sample, component, idx, _, _ in raw_ids])
    raw_intrinsic = np.array(
        [component == "intrinsic" for sample, component, idx, _, _ in raw_ids]
    )
    raw_nue = np.array([sample == "nue" for sample, component, idx, _, _ in raw_ids])
    raw_numu = np.array([sample == "numu" for sample, component, idx, _, _ in raw_ids])
    raw_nuebar = np.array(
        [sample == "nuebar" for sample, component, idx, _, _ in raw_ids]
    )
    raw_numubar = np.array(
        [sample == "numubar" for sample, component, idx, _, _ in raw_ids]
    )
    raw_masks = {
        "osc": raw_osc,
        "intrinsic": raw_intrinsic,
        "nue": raw_nue,
        "numu": raw_numu,
        "nuebar": raw_nuebar,
        "numubar": raw_numubar,
    }
    raw_masks = {("raw", key): value for key, value in raw_masks.items()}
    rebinned_masks = {
        ("rebinned", key[1]): np.array(
            [np.any(raw[slc]) for slc in transform_slices["raw_to_rebin"]]
        )
        for key, raw in raw_masks.items()
    }
    collapsed_masks = {
        ("collapsed", key[1]): np.array(
            [np.any(raw[slc]) for slc in transform_slices["rebin_to_collapsed"]]
        )
        for key, raw in raw_masks.items()
    }
    component_bin_masks = {}
    component_bin_masks.update(raw_masks)
    component_bin_masks.update(rebinned_masks)
    component_bin_masks.update(collapsed_masks)

    return transform_slices, orig_idx_order, component_bin_masks, raw_ids


def sort_events(events, bin_transforms, is_data=False):
    sorted_events = np.empty(np.shape(events), dtype=events.dtype)
    sample_masks = masks_by_sample(events)
    component_masks = masks_by_component(events, is_data=is_data)
    transform_slices, orig_idx_order, component_bin_masks, raw_ids = bin_transforms

    # all_mask = np.ones(len(events)).astype(bool)
    masks = [None, None, None]
    last = [None, None, None]
    last_id = [None, None, None]

    def update(raw_id, i):
        sample, component, idx, low_e, high_e = raw_id
        if i == 0:
            masks[i] = sample_masks[sample]
            # all_mask[masks[i]] = False
            last[i] = events[masks[i]]
        elif i == 1:
            masks[i] = component_masks[component][masks[0]]
            # all_mask[masks[i-1]][masks[i]] = False
            last[i] = last[0][masks[i]]
        elif i == 2:
            masks[i] = np.logical_and(
                last[1]["ntuple_energy"] >= low_e, last[1]["ntuple_energy"] < high_e
            )
            # all_mask[masks[i-2]][masks[i-1]][masks[i]] = False
            last[i] = last[1][masks[i]]

    slices = []

    offset = 0
    for raw_id in raw_ids:
        sample, component, idx, low_e, high_e = raw_id
        idx = (sample, component, idx)
        bad = False
        for i in range(len(idx)):
            bad |= (last_id[i] is None) or (last_id[i] != idx[i])
            if bad:
                update(raw_id, i)
                last_id[i] = idx[i]
        n = len(last[-1])
        slices.append(slice(offset, offset + n))
        offset += n
        sorted_events[slices[-1]] = last[-1]
    sorted_events = sorted_events[:offset]

    return sorted_events, tuple(slices)


@njit
def rebin_cov(cov, slices):
    n = len(slices)

    mid_cov = np.empty((n,) + np.shape(cov)[1:])
    for i in range(n):
        mid_cov[i, :] = np.sum(cov[slices[i], :], axis=0)

    new_cov = np.empty((n, n) + np.shape(cov)[2:])
    for i in range(n):
        new_cov[:, i] = np.sum(mid_cov[:, slices[i]], axis=1)

    return new_cov


@njit
def rebin_cv(cv, slices):
    n = len(slices)

    new_cv = np.empty((n,) + np.shape(cv)[1:])
    for i in range(n):
        new_cv[i] = np.sum(cv[slices[i]], axis=0)

    return new_cv


def revert_cov(cov, component_bin_masks, stage):
    mid_cov = np.zeros(np.shape(cov))
    osc = component_bin_masks[(stage, "osc")]
    intrinsic = component_bin_masks[(stage, "intrinsic")]
    nue = component_bin_masks[(stage, "nue")]
    numu = component_bin_masks[(stage, "numu")]
    nuebar = component_bin_masks[(stage, "nuebar")]
    numubar = component_bin_masks[(stage, "numubar")]
    nue_osc = np.logical_and(nue, osc)
    nue_intrinsic = np.logical_and(nue, intrinsic)
    nuebar_osc = np.logical_and(nuebar, osc)
    nuebar_intrinsic = np.logical_and(nuebar, intrinsic)

    if stage == "rebinned" or stage == "raw":
        order = [nue_osc, nue_intrinsic, numu, nuebar_osc, nuebar_intrinsic, numubar]
    elif stage == "collapsed":
        order = [nue, numu, nuebar, numubar]
    offset = 0
    for m in order:
        m[-1] = False
        n = np.count_nonzero(m)
        mid_cov[offset : offset + n] = cov[m]
        offset += n

    new_cov = np.zeros(np.shape(mid_cov))

    offset = 0
    for m in order:
        m[-1] = False
        n = np.count_nonzero(m)
        new_cov[:, offset : offset + n] = mid_cov[:, m]
        offset += n

    return new_cov


def dl_mb_mc_true_energy_binning():
    return np.linspace(0, 1.5, 100 + 1)


def wc_mb_mc_true_energy_binning():
    return np.linspace(0, 3.0, 60 + 1)


def ub_sort_mb_mc(events, true_energy_edges):
    true_mapping = np.digitize(events["ntuple_enugen"], bins=true_energy_edges) - 1
    masks = np.array([true_mapping == i for i in range(len(true_energy_edges) - 1)])

    no_mask = ~functools.reduce(np.logical_or, masks)
    masks = list(masks) + [no_mask]

    idx_sort = np.empty(len(events), dtype=int)
    idx = np.arange(len(events))

    sorted_events = np.empty(events.shape, dtype=events.dtype)
    bin_edge = 0
    bin_slices = []
    for mask in masks:
        n_events = np.count_nonzero(mask)
        bin_slices.append(slice(bin_edge, bin_edge + n_events))
        sorted_events[bin_edge : bin_edge + n_events] = events[mask]
        idx_sort[bin_edge : bin_edge + n_events] = idx[mask]
        bin_edge += n_events

    bin_slices = bin_slices[:-1]

    return sorted_events, tuple(bin_slices), idx_sort


def dl_sort_smearing_mc(events, true_energy_edges, reco_energy_edges):
    true_mapping = np.digitize(events["true_energy"], bins=true_energy_edges) - 1
    reco_mapping = np.digitize(events["reco_energy"], bins=reco_energy_edges) - 1
    # idx = np.lexsort([true_mapping, reco_mapping])
    true_masks = np.array(
        [true_mapping == i for i in range(len(true_energy_edges) - 1)]
    )
    reco_masks = np.array(
        [reco_mapping == i for i in range(len(reco_energy_edges) - 1)]
    )
    masks = []
    for reco_mask in reco_masks:
        for true_mask in true_masks:
            mask = np.logical_and(true_mask, reco_mask)
            masks.append(mask)

    no_mask = ~functools.reduce(np.logical_or, masks)
    masks = list(masks) + [no_mask]

    sorted_events = np.empty(events.shape, dtype=events.dtype)
    bin_edge = 0
    bin_slices = []
    for mask in masks:
        n_events = np.count_nonzero(mask)
        bin_slices.append(slice(bin_edge, bin_edge + n_events))
        sorted_events[bin_edge : bin_edge + n_events] = events[mask]
        bin_edge += n_events
    bin_slices = bin_slices[:-1]

    reco_bin_edge = 0
    reco_bin_slices = []
    for mask in reco_masks:
        n_events = np.count_nonzero(mask)
        reco_bin_slices.append(slice(reco_bin_edge, reco_bin_edge + n_events))
        reco_bin_edge += n_events

    return sorted_events, tuple(bin_slices), tuple(reco_bin_slices)


def main():
    pass


if __name__ == "__main__":
    main()
