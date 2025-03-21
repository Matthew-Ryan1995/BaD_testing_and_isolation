#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:56:01 2025

This script creates heat maps of pt vs qt spread looking at different endemic prevalences

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

# %%


def get_testing_plots(save_plot_flag=True,
                      generate_data_flag=False,
                      epidemic=False,
                      num_days_to_run=1000,
                      step=0.05,
                      **kwargs):
    start = 0
    stop = 1
    pT_range = np.arange(start=start, stop=stop+step, step=step)
    qT_range = pT_range

    if epidemic:
        title_pre = "Final size"
        save_pre = "finalSize"
        save_pre_b = "peak"
    else:
        title_pre = "Steady state"
        save_pre = "steadyState"
        save_pre_b = "steadyState"

    if generate_data_flag:
        params = np.meshgrid(pT_range, qT_range)

        iter_vals = np.array(params).reshape(2, len(pT_range)*len(qT_range)).T

        default_dict = load_param_defaults()

        O = []
        A = []
        B = []
        T = []
        r0 = []

        if epidemic:
            T_peak = []
            O_peak = []
            A_peak = []
            default_dict["immune_period"] = np.inf

        if epidemic:
            for idx, x in enumerate(iter_vals):
                tmp_dict = default_dict.copy()
                tmp_dict["pT"] = x[0]
                tmp_dict["qT"] = 1-x[1]

                M = bad(**tmp_dict)

                M.run(t_end=num_days_to_run,
                      flag_incidence_tracking=True, t_step=0.05)

                M.results[M.results < 1e-6] = 0

                T.append(M.results[-1, Compartments.T_inc])
                O.append(M.results[-1, [Compartments.T_inc,
                         Compartments.In_inc, Compartments.Ib_inc]].sum())
                A.append(M.results[-1, [Compartments.An_inc,
                         Compartments.Ab_inc]].sum())

                T_peak.append(M.results[:, Compartments.T].max())
                O_peak.append(M.get_I().max())
                A_peak.append(M.get_all_I().max())
                B.append(M.get_B().max())

                r0.append(M.get_reproduction_number())

        else:
            for idx, x in enumerate(iter_vals):
                tmp_dict = default_dict.copy()
                tmp_dict["pT"] = x[0]
                tmp_dict["qT"] = 1-x[1]

                M = bad(**tmp_dict)

                ss, _ = M.find_ss()

                O.append(
                    ss[[Compartments.In, Compartments.Ib, Compartments.T]].sum())
                A.append(ss[[Compartments.An,
                         Compartments.Ab]].sum())
                B.append(ss[[Compartments.Sb, Compartments.Ib, Compartments.Ab,
                             Compartments.Rb, Compartments.T, Compartments.Eb]].sum())
                T.append(ss[Compartments.T])
                r0.append(M.get_reproduction_number())

        O = np.array(O).reshape(params[0].shape)
        A = np.array(A).reshape(params[0].shape)
        B = np.array(B).reshape(params[0].shape)
        T = np.array(T).reshape(params[0].shape)
        r0 = np.array(r0).reshape(params[0].shape)

        save_data = {
            "O": O,
            "A": A,
            "B": B,
            "T": T,
            "r0": r0,
            "params": params,
            "default_dict": default_dict
        }

        if epidemic:
            T_peak = np.array(T_peak).reshape(params[0].shape)
            O_peak = np.array(O_peak).reshape(params[0].shape)
            A_peak = np.array(A_peak).reshape(params[0].shape)

            save_data["T_peak"] = T_peak
            save_data["O_peak"] = O_peak
            save_data["A_peak"] = A_peak

        with open(f"../outputs/testing/{save_pre}_pt_qt.pkl", "wb") as f:
            pkl.dump(save_data, f)
    else:
        with open(f"../outputs/testing/{save_pre}_pt_qt.pkl", "rb") as f:
            dat = pkl.load(f)
        O = dat["O"]
        A = dat["A"]
        B = dat["B"]
        T = dat["T"]
        r0 = dat["r0"]
        params = dat["params"]
        default_dict = dat["default_dict"]

        if epidemic:
            T_peak = dat["T_peak"]
            O_peak = dat["O_peak"]
            A_peak = dat["A_peak"]

    # %%

    xx = params[0]
    yy = params[1]

    plt.figure()
    plt.title(f"{title_pre}: symptomatic infection ($I+T$)")
    im = plt.contourf(xx, yy, O, cmap=plt.cm.Oranges)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre}_pt_qt_infection_symp.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title(f"{title_pre}: all infection ($O+A$)")
    im = plt.contourf(xx, yy, O + A, cmap=plt.cm.Reds)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O + A,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre}_pt_qt_infection_total.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title(f"{title_pre}: Observed infection ($T$)")
    im = plt.contourf(xx, yy, T, cmap=plt.cm.Greens)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, T,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre}_pt_qt_obs_infection_observed.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title(f"{title_pre}: Undetected symptomatic ($I$)")
    im = plt.contourf(xx, yy, O-T, cmap=plt.cm.plasma)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, O-T,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)  # Get contour values on plot?
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre}_pt_qt_unobserved_symptomatic.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.close()
    else:
        plt.show()

    plt.figure()
    if epidemic:
        plt.title(f"Peak: Total willing to test ($B$)")
    else:
        plt.title(f"Steady state: Total willing to test ($B$)")
    im = plt.contourf(xx, yy, B, cmap=plt.cm.Blues)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, B,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre_b}_pt_qt_behaviour.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

    if epidemic:
        plt.figure()
        plt.title(f"Peak: symptomatic infection ($I+T$)")
        im = plt.contourf(xx, yy, O_peak, cmap=plt.cm.Oranges)
        lvls = im.levels[1:-1]
        ctr = plt.contour(xx, yy, O_peak,
                          colors="black",
                          levels=lvls,
                          alpha=0.5)
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = lvls
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
        plt.ylabel("Efficacy of isolation ($1-q_T$)")
        plt.xlabel("Efficacy of test ($p_T$)")
        if save_plot_flag:
            plt.savefig(f"../img/testing/peak_pt_qt_infection_symp.png",
                        bbox_inches="tight",
                        dpi=dpi)
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title(f"Peak: all infection ($O+A$)")
        im = plt.contourf(xx, yy, A_peak, cmap=plt.cm.Reds)
        lvls = im.levels[1:-1]
        ctr = plt.contour(xx, yy, A_peak,
                          colors="black",
                          levels=lvls,
                          alpha=0.5)
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = lvls
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
        plt.ylabel("Efficacy of isolation ($1-q_T$)")
        plt.xlabel("Efficacy of test ($p_T$)")
        if save_plot_flag:
            plt.savefig(f"../img/testing/peak_pt_qt_infection_total.png",
                        bbox_inches="tight",
                        dpi=dpi)
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title(f"Peak: Observed infection ($T$)")
        im = plt.contourf(xx, yy, T_peak, cmap=plt.cm.Greens)
        lvls = im.levels[1:-1]
        ctr = plt.contour(xx, yy, T_peak,
                          colors="black",
                          levels=lvls,
                          alpha=0.5)
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = lvls
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
        plt.ylabel("Efficacy of isolation ($1-q_T$)")
        plt.xlabel("Efficacy of test ($p_T$)")
        if save_plot_flag:
            plt.savefig(f"../img/testing/peak_pt_qt_obs_infection_observed.png",
                        bbox_inches="tight",
                        dpi=dpi)
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title("Peak: Undetected symptomatic ($I$)")
        im = plt.contourf(xx, yy, O_peak-T_peak, cmap=plt.cm.plasma)
        lvls = im.levels[1:-1]
        ctr = plt.contour(xx, yy, O_peak-T_peak,
                          colors="black",
                          levels=lvls,
                          alpha=0.5)  # Get contour values on plot?
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = lvls
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
        plt.ylabel("Efficacy of isolation ($1-q_T$)")
        plt.xlabel("Efficacy of test ($p_T$)")
        if save_plot_flag:
            plt.savefig(f"../img/testing/peak_pt_qt_unobserved_symptomatic.png",
                        bbox_inches="tight",
                        dpi=300)
            plt.close()
        else:
            plt.show()

    plt.figure()
    plt.title("Reproduction number ($\\mathcal{R}_0$)")
    im = plt.contourf(xx, yy, r0, cmap=plt.cm.viridis)
    lvls = im.levels[1:-1]
    ctr = plt.contour(xx, yy, r0,
                      colors="black",
                      levels=lvls,
                      alpha=0.5)
    cbar = plt.colorbar(im)
    cbar_lvls = lvls
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    plt.ylabel("Efficacy of isolation ($1-q_T$)")
    plt.xlabel("Efficacy of test ($p_T$)")
    if save_plot_flag:
        plt.savefig(f"../img/testing/{save_pre}_pt_qt_R0.png",
                    bbox_inches="tight",
                    dpi=dpi)
        plt.close()
    else:
        plt.show()

# %%


get_testing_plots(save_plot_flag=save_plot_flag,
                  generate_data_flag=generate_data_flag,
                  epidemic=False)
get_testing_plots(save_plot_flag=save_plot_flag,
                  generate_data_flag=generate_data_flag,
                  epidemic=True)
