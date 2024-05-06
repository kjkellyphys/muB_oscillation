import numpy as np
from scipy.stats import chi2
import copy
from math import log10, floor, erf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.tri as tri
from scipy.spatial.distance import pdist, squareform
from matplotlib import colors as mpl_colors
from matplotlib.collections import PatchCollection

import scipy
from scipy.interpolate import splprep, splev
from importlib.resources import open_text

from MicroTools.InclusiveTools.inclusive_osc_tools import (
    Decay_muB_OscChi2,
    DecayMuBNuMuDis,
    DecayMuBNuEDis,
)
import MiniTools as mini
from . import muB_inclusive_datarelease_path, bin_width
from MicroTools import param_scan

###########################
# Matheus
fsize = 11
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.3 * 2.3617)
std_axes_form = [0.18, 0.16, 0.79, 0.76]

rcparams = {
    "axes.labelsize": fsize,
    "xtick.labelsize": fsize,
    "ytick.labelsize": fsize,
    "figure.figsize": std_figsize,
    "legend.frameon": False,
    "legend.loc": "best",
}
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
matplotlib.rcParams["hatch.linewidth"] = 0.3

rcParams.update(rcparams)

# settings for Mini Figs
TOTAL_RATE = False
INCLUDE_MB_LAST_BIN = False
STACKED = False
PLOT_FAMILY = False
PATH_PLOTS = "plots/event_rates/"

PAPER_TAG = r"HKZ\,2024"


##########################
#
def get_CL_from_sigma(sigma):
    return erf(sigma / np.sqrt(2))


def get_chi2vals_w_nsigmas(n_sigmas, ndof):
    return [chi2.ppf(get_CL_from_sigma(i), ndof) for i in range(n_sigmas + 1)]


def get_chi2vals_w_CL(CLs, ndof):
    return [chi2.ppf(cl, ndof) for cl in CLs]


###########################
# Kevin
font0 = FontProperties()
font = font0.copy()
font.set_size(fsize)
font.set_family("serif")

labelfont = font0.copy()
labelfont.set_size(fsize)
labelfont.set_weight("bold")
# params= {'text.latex.preamble' : [r'\usepackage{inputenc}']}
# plt.rcParams.update(params)
legendfont = font0.copy()
legendfont.set_size(fsize)
legendfont.set_weight("bold")

redcol = "#e41a1c"
bluecol = "#1f78b5"
grncol = "#33a12c"
purcol = "#613d9b"
pinkcol = "#fc9b9a"
orcol = "#ff7f00"


def std_fig(ax_form=std_axes_form, figsize=std_figsize, rasterized=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig, ax


def double_axes_fig(
    height=0.5,
    gap=0.1,
    axis_base=[0.14, 0.1, 0.80, 0.18],
    figsize=std_figsize,
    split_y=False,
    split_x=False,
    rasterized=False,
):
    fig = plt.figure(figsize=figsize)

    if split_y and not split_x:
        axis_base = [0.14, 0.1, 0.80, 0.4 - gap / 2]
        axis_appended = [0.14, 0.5 + gap / 2, 0.80, 0.4 - gap / 2]

    elif not split_y and split_x:
        axis_appended = [0.14, 0.1, 0.4 - gap / 2, 0.8]
        axis_base = [0.14 + 0.4 + gap / 2, 0.1, 0.4 - gap / 2, 0.8]

    else:
        axis_base[-1] = height
        axis_appended = axis_base + np.array(
            [0, height + gap, 0, 1 - 2 * height - gap - axis_base[1] - 0.07]
        )

    ax1 = fig.add_axes(axis_appended, rasterized=rasterized)
    ax2 = fig.add_axes(axis_base, rasterized=rasterized)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    return fig, [ax1, ax2]


def data_plot(ax, X, Y, xerr, yerr, zorder=2, label="data", **kwargs):
    return ax.errorbar(
        X,
        Y,
        yerr=yerr,
        xerr=xerr,
        marker="o",
        markeredgewidth=0.75,
        capsize=1,
        markerfacecolor="black",
        markeredgecolor="black",
        ms=1.75,
        lw=0.0,
        elinewidth=0.75,
        color="black",
        label=label,
        zorder=zorder,
        **kwargs,
    )


def step_plot(
    ax, x, y, lw=1, color="red", label="signal", where="post", dashes=(3, 0), zorder=3
):
    return ax.step(
        np.append(x, np.max(x) + x[-1]),
        np.append(y, 0.0),
        where=where,
        lw=lw,
        dashes=dashes,
        color=color,
        label=label,
        zorder=zorder,
    )


def plot_MB_vertical_region(ax, color="dodgerblue", label=r"MiniBooNE $1 \sigma$"):
    ##########
    # MINIBOONE 2018
    matplotlib.rcParams["hatch.linewidth"] = 0.7
    y = [0, 1e10]
    NEVENTS = 381.2
    ERROR = 85.2
    xleft = (NEVENTS - ERROR) / NEVENTS
    xright = (NEVENTS + ERROR) / NEVENTS
    ax.fill_betweenx(
        y,
        [xleft, xleft],
        [xright, xright],
        zorder=3,
        ec=color,
        fc="None",
        hatch="\\\\\\\\\\",
        lw=0,
        label=label,
    )

    ax.vlines(1, 0, 1e10, zorder=3, lw=1, color=color)
    ax.vlines(xleft, 0, 1e10, zorder=3, lw=0.5, color=color)
    ax.vlines(xright, 0, 1e10, zorder=3, lw=0.5, color=color)


# Kevin align
def flushalign(ax):
    ic = 0
    for l in ax.get_yticklabels():
        if ic == 0:
            l.set_va("bottom")
        elif ic == len(ax.get_yticklabels()) - 1:
            l.set_va("top")
        ic += 1

    ic = 0
    for l in ax.get_xticklabels():
        if ic == 0:
            l.set_ha("left")
        elif ic == len(ax.get_xticklabels()) - 1:
            l.set_ha("right")
        ic += 1


# Function to find the path that connects points in order of closest proximity
def nearest_neighbor_path(points):
    # Compute the pairwise distance between points
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    ordered_points = points[path]
    return ordered_points


def get_ordered_closed_region(points, logx=False, logy=False):
    xraw, yraw = points

    # check for nans
    if np.isnan(points).sum() > 0:
        raise ValueError("NaN's were found in input data. Cannot order the contour.")

    # check for repeated x-entries -- remove them
    # x, mask_diff = np.unique(x, return_index=True)
    # y = y[mask_diff]

    if logy:
        if (yraw == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        yraw = np.log10(yraw)
    if logx:
        if (xraw == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        xraw = np.log10(xraw)

    # Transform to unit square space:
    xmin, xmax = np.min(xraw), np.max(xraw)
    ymin, ymax = np.min(yraw), np.max(yraw)

    x = (xraw - xmin) / (xmax - xmin)
    y = (yraw - ymin) / (ymax - ymin)

    points = np.array([x, y]).T
    # points_s     = (points - points.mean(0))
    # angles       = np.angle((points_s[:,0] + 1j*points_s[:,1]))
    # points_sort  = points_s[angles.argsort()]
    # points_sort += points.mean(0)

    # if np.isnan(points_sort).sum()>0:
    #     raise ValueError("NaN's were found in sorted points. Cannot order the contour.")
    # # print(points.mean(0))
    # # return points_sort
    # tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    # # u_new = np.linspace(u.min(), u.max(), len(points[:,0]))
    # x_new, y_new = splev(u, tck, der=0)
    # # x_new, y_new = splev(u_new, tck, der=0)
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    x_new, y_new = points[path].T

    x_new = x_new * (xmax - xmin) + xmin
    y_new = y_new * (ymax - ymin) + ymin

    if logx:
        x_new = 10 ** (x_new)
    if logy:
        y_new = 10 ** (y_new)
    return x_new, y_new


def interp_grid(
    x,
    y,
    z,
    fine_gridx=False,
    fine_gridy=False,
    logx=False,
    logy=False,
    method="interpolate",
    smear_stddev=False,
):
    # default
    if not fine_gridx:
        fine_gridx = 100
    if not fine_gridy:
        fine_gridy = 100

    # log scale x
    if logx:
        xi = np.geomspace(np.min(x), np.max(x), fine_gridx)
    else:
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)

    # log scale y
    if logy:
        yi = np.geomspace(np.min(y), np.max(y), fine_gridy)
    else:
        yi = np.linspace(np.min(y), np.max(y), fine_gridy)

    Xi, Yi = np.meshgrid(xi, yi)
    # if logy:
    #     Yi = 10**(-Yi)

    # triangulation
    if method == "triangulation":
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Zi = interpolator(Xi, Yi)

    elif method == "interpolate":
        Zi = scipy.interpolate.griddata(
            (x, y), z, (xi[None, :], yi[:, None]), method="linear", rescale=True
        )
    else:
        print(f"Method {method} not implemented.")

    # gaussian smear -- not recommended
    if smear_stddev:
        Zi = scipy.ndimage.filters.gaussian_filter(
            Zi, smear_stddev, mode="nearest", order=0, cval=0
        )

    return Xi, Yi, Zi


def round_sig(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def sci_notation(
    num,
    sig_digits=1,
    precision=None,
    exponent=None,
    notex=False,
    optional_sci=False,
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), sig_digits)
        if coeff == 10:
            coeff = 1
            exponent += 1
        if precision is None:
            precision = sig_digits

        if optional_sci and np.abs(exponent) < optional_sci:
            string = rf"{round_sig(num, precision)}"
        else:
            string = r"{0:.{2}f}\times 10^{{{1:d}}}".format(coeff, exponent, precision)

        if notex:
            return string
        else:
            return f"${string}$"

    else:
        return r"0"


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


###########################
def get_cmap_colors(name, ncolors, cmin=0, cmax=1, reverse=False):
    try:
        cmap = plt.get_cmap(name)
    except ValueError:
        cmap = build_cmap(name, reverse=reverse)
    return cmap(np.linspace(cmin, cmax, ncolors, endpoint=True))


def build_cmap(color, reverse=False):
    cvals = [0, 1]
    colors = [color, "white"]
    if reverse:
        colors = colors[::-1]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    return mpl_colors.LinearSegmentedColormap.from_list("", tuples)


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def make_rate_plot(rates, params, name="3+1_osc"):
    fig, ax1 = std_fig(figsize=(3.3 * 1.2, 2.1 * 1.2))
    bins = param_scan.MB_Ereco_official_bins
    bin_w = np.diff(bins)
    bin_c = bins[:-1] + bin_w / 2

    ######################################
    # MiniBooNE
    if TOTAL_RATE:
        units = 1
        ax1.set_ylabel(r"Events")
    else:
        units = 1 / bin_width
        ax1.set_ylabel(r"Events/MeV")

    nue_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_2020.combined",
            f"miniboone_nuedata_lowe.txt",
        )
    )
    nue_tot_bkg = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_2020.combined",
            f"miniboone_nuebgr_lowe.txt",
        )
    )

    Weight_nue_flux = mini.apps.reweight_MC_to_nue_flux(
        param_scan.Etrue_nue, param_scan.Weight_nue, mode="fhc"
    )

    MC_nue_bkg_intrinsic = np.dot(
        param_scan.fast_histogram(
            param_scan.Etrue_nue,
            bins=param_scan.e_prod_e_int_bins,
            weights=Weight_nue_flux,
        )[0],
        mini.apps.migration_matrix_official_bins_nue_11bins,
    )
    nue_bkg_midID = nue_tot_bkg - MC_nue_bkg_intrinsic

    # plot data
    data_plot(
        ax1,
        X=bin_c,
        Y=nue_data * units,
        xerr=bin_w / 2,
        yerr=np.sqrt(nue_data) * units,
        zorder=3,
    )

    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(nue_tot_bkg) * units,
        edgecolor="black",
        lw=0.5,
        ls=(1, (2, 1)),
        label=r"unoscillated total bkg",
        histtype="step",
        zorder=3,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(nue_bkg_midID) * units,
        edgecolor="black",
        facecolor="lightgrey",
        lw=0.5,
        label=r"misID bkg",
        histtype="stepfilled",
        zorder=2,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        # weights=(nue_tot_bkg)*units,
        weights=rates["MC_nue_bkg_total_w_dis"] * units,
        edgecolor="black",
        facecolor="peachpuff",
        lw=0.5,
        label=r"$\nu_e$ disappearance",
        histtype="stepfilled",
        zorder=1.6,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(rates["MC_nue_app"] + rates["MC_nue_bkg_total_w_dis"]) * units,
        # weights=(rates_dic_osc['MC_nue_app'] + nue_tot_bkg)*units,
        edgecolor="black",
        facecolor="lightblue",
        lw=0.5,
        linestyle=(1, (3, 0)),
        label=r"$\nu_\mu \to \nu_e$ appearance",
        histtype="stepfilled",
        zorder=1.5,
    )

    ax1.legend(fontsize=8, markerfirst=False, ncol=1)
    pval = r"$p_{\rm val}$"
    pval_str = rf"{pval} $\,= {sci_notation(mini.fit.get_pval(rates, 38-5)*100, sig_digits=2, optional_sci=2, notex=True)}\%$"
    # ax1.annotate(text=r'MiniBooNE FHC 2020 -- '+ pval_str, xy=(0.0,1.025), xycoords='axes fraction', fontsize=9)
    ax1.annotate(
        text=r"MiniBooNE FHC 2020",
        xy=(0.0, 1.025),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax1.annotate(
        text=rf'\noindent $g_\varphi = {params["g"]:.1f}$\\$m_4 = {params["m4"]:.1f}$ eV\\$|U_{{e4}}|^2 = {params["Ue4Sq"]:.2f}$\\$|U_{{\mu 4}}|^2 = {params["Um4Sq"]:.3f}$',
        xy=(0.72, 0.45),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox=dict(
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            boxstyle="square,pad=0.3",
        ),
    )
    ax1.set_xlabel(r"Reconstructed $E_\nu^{\rm QE}$ (GeV)", fontsize=9, labelpad=2.5)
    if INCLUDE_MB_LAST_BIN:
        ax1.set_xticks([0.2, 0.5, 1, 1.5, 2, 2.5, 3])
        ax1.set_xlim(0.2, 3)
    else:
        ax1.set_xticks([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
        ax1.set_xlim(0.2, 1.5)
    ax1.set_ylim(0, 8)
    # ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax1.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.annotate(
        text=PAPER_TAG,
        xy=(1, 1.025),
        xycoords="axes fraction",
        fontsize=8.5,
        ha="right",
    )
    # ax1.annotate(text=fr'{pval} $\,= {mini.fit.get_pval(rates, 38-5)*100:.1f}\%$', xy=(0.15,0.9), xycoords='axes fraction', fontsize=8.5)
    # fig.savefig(f"{PATH_PLOTS}/Mini_{name}.png", dpi=400)
    fig.savefig(f"{PATH_PLOTS}/Mini_{name}.pdf", dpi=400, bbox_inches="tight")
    return fig, ax1


def make_numu_rate_plot(rates, params, name="3+1_osc"):
    fig, ax1 = std_fig(figsize=(3.3 * 1.2, 2 * 1.2))
    bins = param_scan.MB_Ereco_official_bins_numu
    bin_w = np.diff(bins)
    bin_c = bins[:-1] + bin_w / 2

    ######################################
    # MiniBooNE
    if TOTAL_RATE:
        units = 1
        ax1.set_ylabel(r"Events")
    else:
        units = 1 / bin_w / 1e3
        ax1.set_ylabel(r"Events/MeV")

    numu_data = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_2020.combined",
            f"miniboone_numudata.txt",
        )
    )
    numu_tot_bkg = np.genfromtxt(
        open_text(
            f"MiniTools.include.MB_data_release_2020.combined",
            f"miniboone_numu.txt",
        )
    )

    # plot data
    data_plot(
        ax1,
        X=bin_c,
        Y=numu_data * units,
        xerr=bin_w / 2,
        yerr=np.sqrt(numu_data) * units,
        zorder=3,
    )

    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(numu_tot_bkg) * units,
        edgecolor="black",
        lw=0.5,
        ls=(1, (2, 1)),
        label=r"unoscillated total bkg",
        histtype="step",
        zorder=1.6,
    )

    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=rates["MC_numu_bkg_total_w_dis"] * units,
        edgecolor="black",
        facecolor="thistle",
        lw=0.5,
        label=r"$\nu_\mu$ w/ disappearance",
        histtype="stepfilled",
        zorder=1.6,
    )

    ax1.legend(fontsize=8, markerfirst=False, ncol=1)
    ax1.annotate(
        text=r"MiniBooNE FHC 2020",
        xy=(0.0, 1.025),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax1.annotate(
        text=rf'\noindent $g_\varphi = {params["g"]:.1f}$\\$m_4 = {params["m4"]:.0f}$ eV\\$|U_{{e4}}|^2 = {params["Ue4Sq"]:.2f}$\\$|U_{{\mu 4}}|^2 = {params["Um4Sq"]:.3f}$',
        xy=(0.72, 0.6),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox=dict(
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            boxstyle="square,pad=0.3",
        ),
    )
    ax1.set_xlabel(r"Reconstructed $E_\nu^{\rm QE}$ (GeV)", fontsize=9, labelpad=2.5)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 1.9])
    ax1.set_xlim(0.0, 1.9)
    ax1.set_ylim(0, 500)

    ax1.annotate(
        text=PAPER_TAG,
        xy=(1, 1.025),
        xycoords="axes fraction",
        fontsize=8.5,
        ha="right",
    )

    # fig.savefig(f"{PATH_PLOTS}/Mini_{name}_numu.png", dpi=400)
    fig.savefig(f"{PATH_PLOTS}/Mini_{name}_numu.pdf", dpi=400)
    return fig, ax1


MuBchi2_null_hyp = 93


def make_micro_rate_plot(
    rates, params, name="micro_3+1_osc", PC=False, helicity="conserving"
):
    fig, ax1 = std_fig(figsize=(3.3 * 1.2, 2 * 1.2))

    bins = np.array([0.0 + 0.1 * j for j in range(26)] + [10.0])
    bin_w = np.diff(bins)
    bin_c = bins[:-1] + bin_w / 2

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    uBFC = param_scan.GBFC.miniToMicro(rates["MC_nue_app_for_unfolding"])
    uBFC = np.insert(uBFC, 0, [0.0])
    uBFC = np.append(uBFC, 0.0)

    # MicroBooNE partially inclusive signal by unfolding MiniBooNE Signal
    MC_nue_app_for_unfolding2 = copy.deepcopy(rates["MC_nue_app_for_unfolding"])
    uBPC = param_scan.GBPC.miniToMicro(MC_nue_app_for_unfolding2)
    uBPC = np.insert(uBPC, 0, [0.0])
    uBPC = np.append(uBPC, 0.0)

    uBtemp = np.concatenate([uBFC, uBPC, np.zeros(85)])

    uB_signal = uBPC if PC else uBFC
    #     unfolding = unfolder.MBtomuB(
    #     analysis="1eX_PC" if PC else "1eX",
    #     remove_high_energy=False,
    #     unfold=True,
    #     effNoUnfold=True,
    #     which_template="2020",
    #     )

    # MicroBooNE fully inclusive signal by unfolding MiniBooNE Signal
    #     uB_signal = unfolding.miniToMicro(rates["MC_nue_app_for_unfolding"])
    #     uB_signal = np.insert(uB_signal, 0, [0.0])
    #     uB_signal = np.append(uB_signal, 0.0)

    SAMPLE = "PC" if PC else "FC"
    other_bkg = np.load(muB_inclusive_datarelease_path + f"nueCC_{SAMPLE}_Bkg.npy")
    intrinsic_bkg = np.load(muB_inclusive_datarelease_path + f"nueCC_{SAMPLE}_Sig.npy")
    data = np.load(muB_inclusive_datarelease_path + f"nueCC_{SAMPLE}_Obs.npy")

    # \nu_e disappearance signal replacement
    NuEReps = DecayMuBNuEDis(
        params,
        oscillations=True,
        decay=True,
        decouple_decay=False,
        disappearance=True,
        energy_degradation=True,
        helicity=helicity,
    )

    # \nu_mu disappearance signal replacement

    NuMuReps = DecayMuBNuMuDis(
        params,
        oscillations=True,
        decay=True,
        decouple_decay=False,
        disappearance=True,
        energy_degradation=True,
        helicity=helicity,
    )

    # MicroBooNE
    MuB_chi2 = Decay_muB_OscChi2(
        params,
        uBtemp,
        constrained=False,
        sigReps=[NuEReps[0], NuEReps[1], NuMuReps[0], NuMuReps[1], None, None, None],
        RemoveOverflow=True,
        oscillations=True,
        decay=True,
        decouple_decay=False,
        disappearance=True,
        energy_degradation=True,
        helicity=helicity,
    )

    ######################################
    if TOTAL_RATE:
        units = 1
        ax1.set_ylabel(r"Events")
    else:
        units = 1 / bin_w / 1e3
        ax1.set_ylabel(r"Events/MeV")

    # plot data
    data_plot(
        ax1,
        X=bin_c,
        Y=data * units,
        xerr=bin_w / 2,
        yerr=np.sqrt(data) * units,
        zorder=3,
    )

    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(other_bkg + intrinsic_bkg) * units,
        edgecolor="black",
        lw=0.5,
        ls=(1, (2, 1)),
        label=r"unoscillated total bkg",
        histtype="step",
        zorder=1.8,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=other_bkg * units,
        edgecolor="black",
        facecolor="lightgrey",
        lw=0.5,
        label=r"Non-$\nu_e$ bkg",
        histtype="stepfilled",
        zorder=1.7,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(other_bkg + NuEReps[1 if PC else 0]) * units,
        edgecolor="black",
        facecolor="peachpuff",
        lw=0.5,
        label=r"$\nu_e$ disappearance",
        histtype="stepfilled",
        zorder=1.6,
    )
    ax1.hist(
        bins[:-1],
        bins=bins,
        weights=(uB_signal + NuEReps[1 if PC else 0]) * units,
        edgecolor="black",
        facecolor="lightblue",
        lw=0.5,
        label=r"$\nu_e$ appearance",
        histtype="stepfilled",
        zorder=1.4,
    )

    ax1.legend(loc="upper right", fontsize=8, markerfirst=False, ncol=1)
    ax1.annotate(
        text=rf'MicroBooNE {"PC" if PC else "FC"} 2020 $\vert$ $\Delta \chi^2 = {MuB_chi2 - MuBchi2_null_hyp:.0f}$',
        xy=(0.0, 1.025),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax1.annotate(
        text=rf'\noindent $g_\varphi = {params["g"]:.1f},\, m_4 = {params["m4"]:.0f}$ eV\\$|U_{{e4}}|^2 = {params["Ue4Sq"]:.2f}\\|U_{{\mu 4}}|^2 = {params["Um4Sq"]:.3f}$',
        xy=(0.025, 0.91),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox=dict(
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            boxstyle="square,pad=0.3",
        ),
    )

    ax1.set_xlabel(r"Reconstructed $E_\nu^{\rm QE}$ (GeV)", fontsize=9, labelpad=2.5)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax1.set_xlim(0.0, 2.5)
    ax1.set_ylim(0, 0.3 if PC else 0.5)

    ax1.annotate(
        text=PAPER_TAG,
        xy=(1, 1.025),
        xycoords="axes fraction",
        fontsize=8.5,
        ha="right",
    )

    #     fig.savefig(f"{PATH_PLOTS}/Micro_{name}_{'PC' if PC else 'FC'}.png", dpi=400)
    fig.savefig(f"{PATH_PLOTS}/Micro_{name}_{'PC' if PC else 'FC'}.pdf", dpi=400)
    return fig, ax1
