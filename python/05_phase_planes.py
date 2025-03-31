#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:32:36 2025

This script generates Figures 4, S3-S5 from the manuscript

@author: Matt Ryan
"""
from BaD import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as clrs
import matplotlib.ticker as tkr


params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif",
          "font.size": 14}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %% flags
save_flag = True  # NB trying to output plots during the session may result in errors from too many plots being generated
dpi = 300
num_days_to_run = 3*100

# %% Load parameters

params = load_param_defaults()

# %% functions


def add_arrow(line, position=None, direction='right', size=15, color=None, linestyle=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()
    if linestyle is None:
        linestyle = line.get_linestyle()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    ymax = np.argmax(ydata)
    start_ind = np.argmin(np.absolute(xdata[:ymax] - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->",
                                       color=color, lw=3),
                       size=size
                       )


def get_position(x, y):
    ymax = y.max()

    pos_idx = np.where(y == ymax)
    pos = (x[pos_idx]+x[0])/2

    return pos


def create_phase_diagrams(params, p_range,
                          p="w1",
                          fixed_r0="full",
                          r0=3.28,
                          num_days_to_run=300,
                          endemic=False,
                          save_flag=False,
                          dpi=300):

    save_folder = "../img/phase_plots/"
    save_tag = p + f"_fixedR0_{fixed_r0}_endemic_{endemic}.png"

    latex_labels = {
        "w1": "$\\omega_1$",
        "w2": "$\\omega_2$",
        "w3": "$\\omega_3$",
    }

    params = params.copy()
    if not endemic:
        params["immune_period"] = np.inf
    num_models = len(p_range)
    linestyles = ["dashed",
                  "dotted",
                  "-",
                  "dashdot"]

    M = []
    for idx, pp in enumerate(p_range):
        tmp = params.copy()
        tmp[p] = pp
        tmp["transmission"] = r0/(tmp["infectious_period"] *
                                  (tmp["pA"]*tmp["qA"] + 1 - tmp["pA"]))

        M.append(bad(**tmp))
        if fixed_r0 == "disease":
            M[idx].init_cond = M[idx].set_initial_conditions(starting_B=1e-6)

        if fixed_r0 == "full":
            M[idx].update_params(**{"transmission": 1})
            R_multi = M[idx].get_reproduction_number()
            M[idx].transmission = r0/R_multi

        M[idx].run(t_end=num_days_to_run,
                   t_step=0.1)

    # %% S v T
    col = "orange"

    plt.figure()
    for idx in range(num_models):
        pos = get_position(x=M[idx].get_S(),
                           y=M[idx].results[:, Compartments.T])
        l = plt.plot(M[idx].get_S(),
                     M[idx].results[:, Compartments.T],
                     color=col,
                     linestyle=linestyles[idx],
                     label=f"{latex_labels[p]}:{p_range[idx]}")[0]
        add_arrow(l,
                  color=col,
                  position=pos)
    plt.xlabel("$S$", size=20)
    plt.xlim(0, 1)
    plt.ylabel("$T$", size=20)
    plt.ylim(0, 0.12)
    plt.legend()
    if save_flag:
        plt.savefig(f"{save_folder}phasePlot_SvT_{save_tag}",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    # %% S vs I
    col = "purple"

    plt.figure()
    for idx in range(num_models):
        pos = get_position(x=M[idx].get_S(),
                           y=M[idx].get_all_I())
        l = plt.plot(M[idx].get_S(),
                     M[idx].get_all_I(),
                     color=col,
                     linestyle=linestyles[idx],
                     label=f"{latex_labels[p]}:{p_range[idx]}")[0]
        add_arrow(l,
                  color=col,
                  position=pos)
    plt.xlabel("$S$", size=20)
    plt.xlim(0, 1)
    plt.ylabel("$O+A$", size=20)
    plt.ylim(0, 0.30)
    plt.legend()
    if save_flag:
        plt.savefig(f"{save_folder}phasePlot_SvI_{save_tag}",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    # %% B vs T
    col = "fuchsia"

    plt.figure()
    for idx in range(num_models):
        pos = get_position(x=M[idx].get_B(),
                           y=M[idx].results[:, Compartments.T])
        l = plt.plot(M[idx].get_B(),
                     M[idx].results[:, Compartments.T],
                     color=col,
                     linestyle=linestyles[idx],
                     label=f"{latex_labels[p]}:{p_range[idx]}")[0]
        add_arrow(l,
                  color=col,
                  position=pos)
    plt.xlabel("$B$", size=20)
    plt.xlim(0, 1)
    plt.ylabel("$T$", size=20)
    plt.ylim(0, 0.12)
    plt.legend()
    if save_flag:
        plt.savefig(f"{save_folder}phasePlot_BvT_{save_tag}",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()


# %% Create models
p_list = ["w"+str(x) for x in range(1, 4)]

for p in p_list:
    p_range = [0.001,
               params[p]/2.5,
               params[p],
               2.5*params[p]]
    for d in ["disease", "full"]:
        for e in [True, False]:
            create_phase_diagrams(params=params,
                                  p_range=p_range,
                                  p=p,
                                  fixed_r0=d,
                                  endemic=e,
                                  save_flag=save_flag)
