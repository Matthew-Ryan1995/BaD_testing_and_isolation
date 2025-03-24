#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:12:40 2023

Calculate the boundaries and regions for the steady states for a given parameter set.

@author: Matt Ryan
"""
# %% Packages

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *
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
save_plot_flag = False
generate_data_flag = False
dpi = 300

d_step = 0.01
b_step = 0.01


# %% functions

# Create steady state regions based on parameter inputs


def create_ss_region_data(input_params,
                          disease_range=[0, 5], disease_step=0.01,
                          behav_range=[0, 3], behav_step=0.01,
                          generate_data_flag=False):

    params = dict(input_params)

    save_lbl = "steady_states_regions_w1"
    append_txt = ""
    if input_params["w2"] > 0:
        save_lbl += "w2_"
    else:
        append_txt += "w2_0_"
    if input_params["w3"] > 0:
        save_lbl += "w3_"
    else:
        append_txt += "w3_0"

    # Find the line where R0 = 1
    if generate_data_flag:
        r0_b = np.arange(start=behav_range[0],
                         stop=behav_range[1] + behav_step, step=behav_step)

        R0_multiplier = params["infectious_period"] * \
            (params["pA"]*params["qA"] + 1 - params["pA"])

        r0_d = list()

        for idx, r0b in enumerate(r0_b):
            tmp_dict = params.copy()
            tmp_dict["transmission"] = 1
            tmp_dict["w1"] = r0b*(tmp_dict["a1"] + tmp_dict["a2"])

            M_tmp = bad(**tmp_dict)

            tmp_beta = 1/M_tmp.get_reproduction_number()

            r0_d.append(tmp_beta*R0_multiplier)

        r0_d = np.array(r0_d)
        # # Calculate steady state regions
        if r0_d.max() > disease_range[1]:
            disease_range[1] = r0_d.max()
        r0_d_mesh_vals = np.arange(start=disease_range[0],
                                   stop=disease_range[1] + disease_step,
                                   step=disease_step)

        grid_vals = np.meshgrid(r0_b, r0_d_mesh_vals)

        iter_vals = np.array(grid_vals).reshape(
            2, len(r0_b)*len(r0_d_mesh_vals)).T

        ss_categories = list()

        for idxx in range(len(iter_vals)):

            tmp_params = dict(params)
            tmp_r0d = iter_vals[idxx, 1]
            tmp_r0b = iter_vals[idxx, 0]

            tmp_params["w1"] = tmp_r0b * \
                (tmp_params["a1"] + tmp_params["a2"])
            if tmp_params["w1"] < 0:
                tmp_params["w1"] = 0

            pA = tmp_params["pA"]
            qA = tmp_params["qA"]
            beta = tmp_r0d * \
                (1/tmp_params["infectious_period"]) / ((pA*qA + 1 - pA))
            tmp_params["transmission"] = beta

            M = bad(**tmp_params)
            ss, _ = M.find_ss()

            ss = ss.round(6)

            B = ss[[Compartments.Sb, Compartments.Eb, Compartments.Ab,
                    Compartments.Ib, Compartments.Rb, Compartments.T]].sum()
            I = ss[[Compartments.Ab, Compartments.Ib, Compartments.An,
                    Compartments.In, Compartments.T]].sum()

            # 0 - BaD free
            # 1 - B free, D endemic
            # 2 - D free, B endemic
            # 3 - full endemic
            if B > 0:
                if I > 0:
                    ss_categories.append(3)
                else:
                    ss_categories.append(2)
            else:
                if I > 0:
                    ss_categories.append(1)
                else:
                    ss_categories.append(0)

        ss_categories = np.array(ss_categories).reshape(grid_vals[0].shape)

        save_data = {
            "grid_vals": grid_vals,
            "ss_categories": ss_categories,
            "r0_d": r0_d,
            "r0_b": r0_b
        }

        with open(f"../outputs/steady_state_regions/{save_lbl}.pkl", "wb") as f:
            pkl.dump(save_data, f)
    else:
        with open(f"../outputs/steady_state_regions/{save_lbl}.pkl", "rb") as f:
            dat = pkl.load(f)

        grid_vals = dat["grid_vals"]
        ss_categories = dat["ss_categories"]
        r0_d = dat["r0_d"]
        r0_b = dat["r0_b"]

    return grid_vals, ss_categories, (r0_d, r0_b)


# Create a plot of the steady state regions
def create_ss_plots(input_params, grid_vals, ss_categories, lines, save=False):

    lvls = [0, 0.5, 1.5, 2.5, 3.5]
    cmap = plt.cm.RdBu_r

    r0_d = lines[0]
    r0_b = lines[1]

    if input_params["w2"] > 0:
        y_line = [0, 1]
    else:
        y_line = [0, grid_vals[1].max()]

    fontname = {'fontname': "serif"}

    code_to_label = {
        # "transmission": "beta",
        "infectious_period": "gamma_inv",
        "immune_period": "nu_inv",
        "qB": "qB",
        "qT": "qT",
        "a1": "a1",
        # "B_social": "w1",
        # "B_fear": "w2",
        # "B_const": "w3",
        "a2": "a2"
    }
    code_to_latex = {
        # "transmission": "beta",
        "infectious_period": "$1/\\gamma$",
        "immune_period": "$1/\\nu$",
        "qB": "$qB$",
        "qT": "$pqT",
        "a1": "$\\alpha_1$",
        # "B_social": "w1",
        # "B_fear": "w2",
        # "B_const": "w3",
        "a2": "$\\alpha_2$"
    }
    title = "Steady state regions: "
    if np.isclose(input_params["w2"], 0):
        title += "$\\omega_2$ = 0, "
    if np.isclose(input_params["w3"], 0):
        title += "$\\omega_3$ = 0"
    title += "\n| "

    for var in code_to_latex.keys():
        if var == "N_social":
            title += "\n| "
        title += code_to_latex[var] + " = " + \
            str(np.round(input_params[var], 1)) + " | "
    if input_params["w2"] > 0:
        title += "$\\omega_2$  = " + \
            str(np.round(input_params["w2"], 1)) + " | "
    if input_params["w3"] > 0:
        title += "$\\omega_2$  = " + \
            str(np.round(input_params["w3"], 1)) + " | "

    save_lbl = "steady_states_regions_w1"
    # for var in code_to_label.keys():
    #     save_lbl += code_to_label[var] + "_" + str(input_params[var]) + "_"

    append_txt = ""
    if input_params["w2"] > 0:
        save_lbl += "w2_"
    else:
        append_txt += "w2_0_"
    if input_params["w3"] > 0:
        save_lbl += "w3_"
    else:
        append_txt += "w3_0"

    caption_txt = "Dark blue - BaD free; Light blue - B free, D endemic; \nLight red - B endemic, D free; Dark red - BaD endemic"

    plt.figure()
    # plt.title(title)
    plt.tight_layout()
    plt.contourf(grid_vals[1], grid_vals[0], ss_categories,
                 levels=lvls,  cmap=cmap)
    plt.contour(grid_vals[1], grid_vals[0], ss_categories,
                levels=[0, 1, 2, 3],
                colors="black")
    plt.plot(r0_d, r0_b, linestyle="-", color="black", linewidth=2)
    if np.isclose(input_params["w3"], 0):
        plt.plot(y_line, [1, 1], color="black", linewidth=2)
    plt.ylabel(
        "Social reproduction number ($\\mathcal{R}_0^{B}$)"
    )
    plt.xlabel(
        "Behaviour-free reproduction number ($\\mathcal{R}_0^{D}$)"
    )

    y_ticks = plt.yticks()[0]
    y_spacing = y_ticks[1] - y_ticks[0]

    # plt.text(0., -y_spacing - 0.5, caption_txt, va="top")
    x_pos = 1.5+(grid_vals[1].max() - 1)/2
    y_pos = 1+(grid_vals[0].max() - 1)/2
    if np.isclose(input_params["w3"], 0):
        plt.text(0.5, 0.5, "$E_{00}$", size=16, va="center", ha="center")
        if np.isclose(input_params["w2"], 0):
            plt.text(x_pos, 0.5, "$E_{0D}$",
                     size=16, va="center", ha="center", **fontname)
        else:
            plt.text(2.5, 0.3, "$E_{0D}$",
                     size=16, va="center", ha="center", **fontname)

    plt.text(0.5, y_pos, "$E_{B0}$", size=16,
             va="center", ha="center", family="serif")
    if not np.isclose(input_params["w3"], 0):
        plt.text(x_pos, y_pos-1, "$E_{BD}$",
                 size=16, va="center", ha="center", **fontname)
    else:
        plt.text(x_pos, y_pos, "$E_{BD}$",
                 size=16, va="center", ha="center", **fontname)

    if save:
        plt.savefig("../img/steady_state_regions/" + save_lbl + append_txt +
                    ".png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# %% Load baseline parameters
plot_params_baseline = load_param_defaults()


plot_params = dict(plot_params_baseline)

grid_vals, ss_categories, lines = create_ss_region_data(
    plot_params,
    disease_step=d_step,
    behav_step=b_step,
    generate_data_flag=generate_data_flag)
create_ss_plots(plot_params,
                grid_vals,
                ss_categories,
                lines=lines,
                save=save_plot_flag)


plot_params["w3"] = 0
grid_vals, ss_categories, lines = create_ss_region_data(
    plot_params,
    disease_step=d_step,
    behav_step=b_step,
    generate_data_flag=generate_data_flag)
create_ss_plots(plot_params,
                grid_vals,
                ss_categories,
                lines=lines,
                save=save_plot_flag)

plot_params["w2"] = 0
grid_vals, ss_categories, lines = create_ss_region_data(
    plot_params,
    disease_step=d_step,
    behav_step=b_step,
    generate_data_flag=generate_data_flag)
create_ss_plots(plot_params,
                grid_vals,
                ss_categories,
                lines=lines,
                save=save_plot_flag)


