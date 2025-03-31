#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:56:01 2025

This script creates heat maps of R0 vs behaviour spread looking at different endemic prevalences.  

This cript produces Figure 6 from the main manuscript.

@author: Matt Ryan
"""
from BaD import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import pickle as pkl


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
save_plot_flag = False
generate_data_flag = False
dpi = 300


# %%

def get_ss_contour_plots(R0d_start=0.0,
                         R0d_stop=5,
                         R0d_step=0.1,
                         y_label="Behaviour reproduction number ($\\mathcal{R}_0^B$)",
                         save_plot_flag=True,
                         generate_data_flag=False,
                         **kwargs):
    args = kwargs

    v_param = args["v_param"]
    v_start = args["v_start"]
    v_stop = args["v_stop"]
    v_step = args["v_step"]
    v_range = np.arange(start=v_start, stop=v_stop+v_step, step=v_step)

    if v_param == "R0b":
        def update_v(val, pars, vv=v_param):
            ans = val*(pars["a1"] + pars["a2"])
            pars["w1"] = ans
            return pars
    elif v_param == "qT":
        def update_v(val, pars, vv=v_param):
            ans = 1-val
            pars[vv] = ans
            return pars
    else:
        def update_v(val, pars, vv=v_param):
            pars[vv] = val
            return pars

    if generate_data_flag:
        R0d_range = np.arange(start=R0d_start,
                              stop=R0d_stop+R0d_step,
                              step=R0d_step)

        params = np.meshgrid(R0d_range, v_range)

        iter_vals = np.array(params).reshape(2, len(R0d_range)*len(v_range)).T

        default_dict = load_param_defaults()

        R0_multiplier = default_dict["infectious_period"] * \
            (default_dict["pA"]*default_dict["qA"] + 1 - default_dict["pA"])

        r0_one = list()

        for idx, vv in enumerate(v_range):
            tmp_dict = default_dict.copy()
            tmp_dict["transmission"] = 1
            tmp_dict = update_v(val=vv, pars=tmp_dict)

            M_tmp = bad(**tmp_dict)

            tmp_beta = 1/M_tmp.get_reproduction_number()

            r0_one.append(tmp_beta*R0_multiplier)

        I = []
        O = []
        B = []
        T = []
        R0_list = []

        for idx, x in enumerate(iter_vals):
            tmp_dict = default_dict.copy()
            R0 = x[0]
            vv = x[1]

            tmp_dict = update_v(val=vv, pars=tmp_dict)
            tmp_dict["transmission"] = R0 / R0_multiplier

            M = bad(**tmp_dict)

            R0_list.append(M.get_reproduction_number())

            ss, _ = M.find_ss()

            I.append(ss[[Compartments.In, Compartments.Ib, Compartments.An,
                         Compartments.Ab, Compartments.T]].sum())
            O.append(
                ss[[Compartments.T, Compartments.In, Compartments.Ib]].sum())
            B.append(ss[[Compartments.Sb, Compartments.Ib, Compartments.Ab,
                         Compartments.Rb, Compartments.T, Compartments.Eb]].sum())
            T.append(ss[Compartments.T])

        I = np.array(I).reshape(params[0].shape)
        O = np.array(O).reshape(params[0].shape)
        B = np.array(B).reshape(params[0].shape)
        T = np.array(T).reshape(params[0].shape)
        R0 = np.array(R0_list).reshape(params[0].shape)

        save_data = {
            "I": I,
            "O": O,
            "B": B,
            "T": T,
            "R0": R0,
            "r0_one": r0_one,
            "params": params,
            "default_dict": default_dict
        }

        with open(f"../outputs/endemic/r0d_{v_param}_ss.pkl", "wb") as f:
            pkl.dump(save_data, f)
    else:
        with open(f"../outputs/endemic/r0d_{v_param}_ss.pkl", "rb") as f:
            dat = pkl.load(f)
        I = dat["I"]
        O = dat["O"]
        B = dat["B"]
        T = dat["T"]
        R0 = dat["R0"]
        r0_one = dat["r0_one"]
        params = dat["params"]
        default_dict = dat["default_dict"]

        R0_multiplier = default_dict["infectious_period"] * \
            (default_dict["pA"]*default_dict["qA"] + 1 - default_dict["pA"])

    if v_param == "R0b":
        y_cross = default_dict["w1"]/(default_dict["a1"] + default_dict["a2"])
    else:
        y_cross = default_dict[v_param]
    x_cross = default_dict["transmission"] * R0_multiplier

    x_label = "Behaviour-free reproduction number ($\\mathcal{R}_0^D$)"

    xx = params[0]
    yy = params[1]

    plt.figure()
    plt.title("Total infection ($O+A$)")
    im = plt.contourf(xx, yy, I, cmap=plt.cm.Reds)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, I,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_infection_total.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Observed infection ($T$)")
    im = plt.contourf(xx, yy, T, cmap=plt.cm.Greens)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, T,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_infection_observed.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Total willing to test ($B$)")
    im = plt.contourf(xx, yy, B, cmap=plt.cm.Blues)
    ctr = plt.contour(xx, yy, B,
                      colors="black",
                      alpha=0.5)

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=1))
    cbar_lvls = ctr.levels[1:-1]
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_behaviour.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Reproduction number ($\\mathcal{R}_0$)")
    im = plt.contourf(xx, yy, R0, cmap=plt.cm.viridis,
                      levels=[0, 1, 2, 3, 4, 5])
    ctr = plt.contour(xx, yy, R0,
                      colors="black",
                      alpha=0.5,
                      levels=[0, 1, 2, 3, 4, 5])

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im)
    cbar_lvls = ctr.levels[1:-1]
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    # hack to make figures the same saved size - this colour bar involves
    # no percentage signs and so is larger.
    ticks_labs = [str(round(x, 2) + 0.001)[0:4] + "   " for x in cbar_lvls]
    cbar.set_ticklabels(ticks_labs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_reproduction_number.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Symptomatic infection ($O$)")
    im = plt.contourf(xx, yy, O, cmap=plt.cm.Oranges)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O,
                      colors="black",
                      alpha=0.5,
                      levels=lvls)

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im,  format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_symptomatic.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Undetected symptomatic infection ($I$)", size=14)
    im = plt.contourf(xx, yy, O-T, cmap=plt.cm.plasma)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O-T,
                      colors="black",
                      alpha=0.5,
                      levels=lvls)

    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="black", markersize=10)
    cbar = plt.colorbar(im,  format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/endemic/r0d_{v_param}_ss_unobserved_symptomatic.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()


# %%

pars_r0b = {
    "v_param": "R0b",
    "v_start": 0,
    "v_stop": 5,
    "v_step": 0.1
}
get_ss_contour_plots(save_plot_flag=save_plot_flag,
                     generate_data_flag=generate_data_flag,
                     y_label="Social reproduction number ($\\mathcal{R}_0^B$)",
                     **pars_r0b)

# %%
pars_pT = {
    "v_param": "pT",
    "v_start": 0,
    "v_stop": 1,
    "v_step": 0.05
}
get_ss_contour_plots(save_plot_flag=save_plot_flag,
                     generate_data_flag=generate_data_flag,
                     y_label="Efficacy of test ($p_T$)",
                     **pars_pT)
# %%
pars_qT = {
    "v_param": "qT",
    "v_start": 0,
    "v_stop": 1,
    "v_step": 0.05
}
get_ss_contour_plots(save_plot_flag=save_plot_flag,
                     generate_data_flag=generate_data_flag,
                     y_label="Efficacy of isolation ($1-q_T$)",
                     **pars_qT)
