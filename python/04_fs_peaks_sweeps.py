#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:24:21 2025

@author: rya200
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
          "font.size": 12}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %% flags
save_plot_flag = True
generate_data_flag = False
dpi = 300
num_days_to_run = 100

# %%


def get_peaks_fs_plots(R0d_start=0.0,
                       R0d_stop=5,
                       R0d_step=0.05,
                       y_label="Social reproduction number ($\\mathcal{R}_0^B$)",
                       save_plot_flag=True,
                       generate_data_flag=False,
                       **kwargs):
    args = kwargs

    v_param = args["v_param"]
    v_start = args["v_start"]
    v_stop = args["v_stop"]
    v_step = args["v_step"]

    R0d_range = np.arange(start=R0d_start,
                          stop=R0d_stop+R0d_step,
                          step=R0d_step)
    v_range = np.arange(start=v_start, stop=v_stop+v_step, step=v_step)

    if v_param == "R0b":
        def update_v(val, pars, vv=v_param):
            ans = val*(pars["a1"] + pars["a2"])
            pars["w1"] = ans
            return pars
    elif v_param == "B0":
        def update_v(val, pars, vv=v_param):
            ans = (val*(pars["a1"]+pars["a2"]) - pars["a1"]*val)/(1-val)
            pars["w3"] = ans
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
        params = np.meshgrid(R0d_range, v_range)

        iter_vals = np.array(params).reshape(2, len(R0d_range)*len(v_range)).T

        default_dict = load_param_defaults()
        default_dict["immune_period"] = np.inf

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
        T = []
        O = []

        T_peak = []
        O_peak = []
        B_peak = []

        R0_list = []

        for idx, x in enumerate(iter_vals):
            R0 = x[0]
            vv = x[1]
            tmp_dict = dict(default_dict)
            tmp_dict = update_v(val=vv, pars=tmp_dict)

            tmp_dict["transmission"] = R0 / R0_multiplier

            M = bad(**tmp_dict)
            R0_list.append(M.get_reproduction_number())
            if 0.99 < R0_list[idx] < 1.5:
                M.run(t_end=2000, flag_incidence_tracking=True, t_step=0.05)
            else:
                M.run(t_end=num_days_to_run,
                      flag_incidence_tracking=True, t_step=0.05)

            M.results[M.results < 1e-6] = 0

            T.append(M.results[-1, Compartments.T_inc])
            O.append(M.results[-1, [Compartments.T_inc,
                     Compartments.In_inc, Compartments.Ib_inc]].sum())

            T_peak.append(M.results[:, Compartments.T].max())
            O_peak.append(M.get_I().max())
            B_peak.append(M.get_B().max())

        T = np.array(T).reshape(params[0].shape)
        T_peak = np.array(T_peak).reshape(params[0].shape)
        O = np.array(O).reshape(params[0].shape)
        O_peak = np.array(O_peak).reshape(params[0].shape)
        B_peak = np.array(B_peak).reshape(params[0].shape)
        R0 = np.array(R0_list).reshape(params[0].shape)

        save_data = {
            "O": O,
            "O_peak": O_peak,
            "T": T,
            "T_peak": T_peak,
            "B_peak": B_peak,
            "R0": R0,
            "r0_one": r0_one,
            "params": params,
            "default_dict": default_dict
        }

        with open(f"../outputs/epidemic/{v_param}_R0d.pkl", "wb") as f:
            pkl.dump(save_data, f)
    else:
        with open(f"../outputs/epidemic/{v_param}_R0d.pkl", "rb") as f:
            dat = pkl.load(f)
        O = dat["O"]
        O_peak = dat["O_peak"]
        B_peak = dat["B_peak"]
        T = dat["T"]
        T_peak = dat["T_peak"]
        R0 = dat["R0"]
        r0_one = dat["r0_one"]
        params = dat["params"]
        default_dict = dat["default_dict"]

        R0_multiplier = default_dict["infectious_period"] * \
            (default_dict["pA"]*default_dict["qA"] + 1 - default_dict["pA"])

    if v_param == "R0b":
        y_cross = default_dict["w1"]/(default_dict["a1"] + default_dict["a2"])
    elif v_param == "B0":
        M_tmp = bad(**default_dict)
        y_cross = 1-M_tmp.get_ss_N(0)
    else:
        y_cross = default_dict[v_param]
    x_cross = default_dict["transmission"] * R0_multiplier

    x_label = "Behaviour-free reproduction number ($\\mathcal{R}_0^D$)"

    xx = params[0]
    yy = params[1]

    plt.figure()
    plt.title("Observed final size ($T$)")
    lvls = np.arange(0.0, 0.41, 0.05)
    im = plt.contourf(xx, yy, T, cmap=plt.cm.Greens, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, T,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_observed_finalsize.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Symptomatic final size ($I+T$)")
    lvls = np.arange(0.0, 0.91, 0.15)
    im = plt.contourf(xx, yy, O, cmap=plt.cm.Oranges, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_symptomatic_finalsize.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Undetected symptomatic final size ($I$)")
    lvls = np.arange(0.0, 0.91, 0.15)
    im = plt.contourf(xx, yy, O-T, cmap=plt.cm.plasma, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O-T,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_unobserved_symptomatic_finalsize.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Observed peak (T)")
    lvls = np.arange(0.0, 0.081, 0.01)
    im = plt.contourf(xx, yy, T_peak, cmap=plt.cm.Greens, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, T_peak,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_observed_peak.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Symptomatic peak (I+T)")
    lvls = np.arange(0.0, 0.28, 0.04)
    im = plt.contourf(xx, yy, O_peak, cmap=plt.cm.Oranges, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O_peak,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_symptomatic_peak.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Undetected symptomatic peak ($I$)")
    lvls = np.arange(0.0, 0.28, 0.04)
    im = plt.contourf(xx, yy, O_peak-T_peak, cmap=plt.cm.plasma, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O_peak-T_peak,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_unobserved_symptomatic_peak.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Willing to test peak ($B$)")
    lvls = np.arange(0.0, 1.11, 0.15)
    im = plt.contourf(xx, yy, B_peak, cmap=plt.cm.Blues, levels=lvls)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, B_peak,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    plt.plot(r0_one, v_range, linestyle="dashed", color="grey")
    plt.plot([x_cross, x_cross], [y_cross, y_cross],
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_behav_peak.png",
                    bbox_inches="tight",
                    dpi=300)
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
             marker="x", color="grey", markersize=10)
    cbar = plt.colorbar(im)
    cbar_lvls = ctr.levels[1:-1]
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_plot_flag:
        plt.savefig(f"../img/epidemic/{v_param}_R0d_reproduction_number.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()
    return

# %%


pars_B0 = {
    "v_param": "B0",
    "v_start": 0.01,
    "v_stop": 0.99,
    "v_step": 0.01
}
get_peaks_fs_plots(save_plot_flag=save_plot_flag,
                   generate_data_flag=generate_data_flag,
                   y_label="Initial behaviour ($B(0)$)",
                   **pars_B0)

# %%

pars_r0b = {
    "v_param": "R0b",
    "v_start": 0,
    "v_stop": 5,
    "v_step": 0.05
}
get_peaks_fs_plots(save_plot_flag=save_plot_flag,
                   generate_data_flag=generate_data_flag,
                   y_label="Social reproduction number ($\\mathcal{R}_0^b$)",
                   **pars_r0b)
# %%
pars_pT = {
    "v_param": "pT",
    "v_start": 0,
    "v_stop": 1,
    "v_step": 0.05
}
get_peaks_fs_plots(save_plot_flag=save_plot_flag,
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
get_peaks_fs_plots(save_plot_flag=save_plot_flag,
                   generate_data_flag=generate_data_flag,
                   y_label="Efficacy of isolation ($1-q_T$)",
                   **pars_qT)
