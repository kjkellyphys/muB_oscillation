import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.tri as tri
from scipy.spatial.distance import pdist, squareform

import scipy
from scipy.interpolate import splprep, splev

###########################
# Matheus
fsize = 11
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.3 * 2.3617)
std_axes_form = [0.16, 0.16, 0.81, 0.76]

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
        ms=2.5,
        lw=0.0,
        elinewidth=0.75,
        color="black",
        label=label,
        zorder=zorder,
        **kwargs
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
    x, y = points

    # check for nans
    if np.isnan(points).sum() > 0:
        raise ValueError("NaN's were found in input data. Cannot order the contour.")

    # check for repeated x-entries --
    # this is an error because
    x, mask_diff = np.unique(x, return_index=True)
    y = y[mask_diff]

    if logy:
        if (y == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        sy = np.sign(y)
        ssy = (np.abs(y) < 1) * (-1) + (np.abs(y) > 1) * (1)
        y = ssy * np.log10(y * sy)
    if logx:
        if (x == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        sx = np.sign(x)
        ssx = (x < 1) * (-1) + (x > 1) * (1)
        x = ssx * np.log10(x * sx)

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

    if logx:
        x_new = sx * 10 ** (ssx * x_new)
    if logy:
        y_new = sy * 10 ** (ssy * y_new)

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
        xi = np.logspace(np.min(np.log10(x)), np.max(np.log10(x)), fine_gridx)
    else:
        xi = np.linspace(np.min(x), np.max(x), fine_gridx)

    # log scale y
    if logy:
        yi = np.logspace(np.min(np.log10(y)), np.max(np.log10(y)), fine_gridy)

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
            (x, y), z, (xi[None, :], yi[:, None]), method="linear", rescale=False
        )
    else:
        print(f"Method {method} not implemented.")

    # gaussian smear -- not recommended
    if smear_stddev:
        Zi = scipy.ndimage.filters.gaussian_filter(
            Zi, smear_stddev, mode="nearest", order=0, cval=0
        )

    return Xi, Yi, Zi