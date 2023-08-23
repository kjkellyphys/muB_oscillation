import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

from matplotlib.ticker import MaxNLocator, FixedLocator, LinearLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredText

from MicroTools import *

###########################
# Matheus 
fsize=12
fsize_annotate=10

std_figsize = (1.2*3.7,1.3*2.3617)
std_axes_form  =[0.16,0.16,0.81,0.76]

rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
                'figure.figsize':std_figsize, 
                'legend.frameon': False,
                'legend.loc': 'best'  }
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
rc('text', usetex=True)
rc('font',**{'family':'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rcParams['hatch.linewidth'] = 0.3

rcParams.update(rcparams)

###########################
# Kevin
font0 = FontProperties()
font = font0.copy()
font.set_size(fsize)
font.set_family('serif')

labelfont=font0.copy()
labelfont.set_size(fsize)
labelfont.set_weight('bold')
#params= {'text.latex.preamble' : [r'\usepackage{inputenc}']}
#plt.rcParams.update(params)
legendfont=font0.copy()
legendfont.set_size(fsize)
legendfont.set_weight('bold')

redcol='#e41a1c'
bluecol='#1f78b5'
grncol='#33a12c'
purcol='#613d9b'
pinkcol='#fc9b9a'
orcol='#ff7f00'


def std_fig(ax_form=std_axes_form, 
            figsize=std_figsize,
            rasterized=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig,ax

def double_axes_fig(height = 0.5,
                    gap = 0.1,
                    axis_base = [0.14,0.1,0.80,0.18], 
                    figsize=std_figsize,
                    split_y=False,
                    split_x=False,
                    rasterized=False):

    fig = plt.figure(figsize=figsize)

    if split_y and not split_x:
        axis_base = [0.14,0.1,0.80,0.4-gap/2]
        axis_appended = [0.14,0.5+gap/2,0.80,0.4-gap/2]
    
    elif not split_y and split_x:
        axis_appended = [0.14,0.1,0.4-gap/2,0.8]
        axis_base = [0.14+0.4+gap/2, 0.1, 0.4-gap/2, 0.8]        

    else:
        axis_base[-1] = height
        axis_appended = axis_base+np.array([0, height+gap, 0, 1 - 2*height - gap - axis_base[1] - 0.05])
        

    ax1 = fig.add_axes(axis_appended, rasterized=rasterized)
    ax2 = fig.add_axes(axis_base, rasterized=rasterized)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    return fig, [ax1, ax2]

def data_plot(ax, X, Y, xerr, yerr, zorder=2, label='data'):
    return ax.errorbar(X, Y, yerr= yerr, xerr = xerr, \
                    marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
                    markeredgecolor="black",ms=2,  lw = 0.0, elinewidth=0.8,
                    color='black', label=label, zorder=zorder)

def step_plot(ax, x, y, lw=1, color='red', label='signal', where = 'post', dashes=(3,0), zorder=3):
    return ax.step( np.append(x, np.max(x)+x[-1]),
                    np.append(y, 0.0),
                    where=where,
                    lw = lw, 
                    dashes=dashes,
                    color = color, 
                    label = label, zorder=zorder)


def plot_MB_vertical_region(ax, color='dodgerblue', label=r'MiniBooNE $1 \sigma$'):
    ##########
    # MINIBOONE 2018
    matplotlib.rcParams['hatch.linewidth'] = 0.7
    y = [0,1e10]
    NEVENTS=381.2
    ERROR = 85.2
    xleft = (NEVENTS-ERROR)/NEVENTS
    xright = (NEVENTS+ERROR)/NEVENTS
    ax.fill_betweenx(y,[xleft,xleft],[xright,xright], 
                        zorder=3,
                        ec=color, fc='None',
                        hatch='\\\\\\\\\\',
                        lw=0,
                        label=label)

    ax.vlines(1,0,1e10, zorder=3, lw=1, color=color)
    ax.vlines(xleft,0,1e10, zorder=3, lw=0.5, color=color)
    ax.vlines(xright,0,1e10, zorder=3, lw=0.5, color=color)

# Kevin align
def flushalign(ax):
    ic = 0
    for l in ax.get_yticklabels():
        if ic == 0:
            l.set_va("bottom")
        elif ic == len(ax.get_yticklabels())-1:
            l.set_va("top")
        ic += 1

    ic = 0
    for l in ax.get_xticklabels():
        if ic == 0:
            l.set_ha("left")
        elif ic == len(ax.get_xticklabels())-1:
            l.set_ha("right")
        ic += 1