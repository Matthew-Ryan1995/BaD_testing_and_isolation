#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 09:56:05 2025

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
dpi = 300
num_days_to_run = 100

# %%
start = 0.1
stop = 5
step = 0.01
R0_range = np.arange(start=start, stop=stop+step, step=step)


default_dict = load_param_defaults()

default_dict["immune_period"] = np.inf

default_dict["transmission"] = 1


M_default = bad(**default_dict)
R0_multiplier = M_default.get_reproduction_number()

T = []
O = []
Inc = []

T_peak = []
O_peak = []
Inc_peak = []

B_peak = []

for idx, R0 in enumerate(R0_range):
    tmp_dict = dict(default_dict)
    tmp_dict["transmission"] = R0 / R0_multiplier

    M = bad(**tmp_dict)
    if 0.99 < R0 < 1.5:
        M.run(t_end=2000, flag_incidence_tracking=True, t_step=0.05)
    else:
        M.run(t_end=num_days_to_run,
              flag_incidence_tracking=True, t_step=0.05)

    T.append(M.results[-1, Compartments.T_inc])
    O.append(M.results[-1, [Compartments.T_inc,
                            Compartments.In_inc, Compartments.Ib_inc]].sum())
    Inc.append(M.incidence[-1])

    T_peak.append(M.results[:, Compartments.T].max())
    O_peak.append(M.get_I().max())
    Inc_peak.append(M.get_all_I().max())

    B_peak.append(M.get_B().max())

# %%

plt.figure()
plt.plot(R0_range, T, label="Observed  (T)", color="green")
plt.plot(R0_range, O, label="Symptomatic (T + I)", color="orange")
plt.plot(R0_range, Inc, label="True (T + I + A)", color="red")
plt.legend()
plt.xlabel("Reproduction number ($\\mathcal{R}_0$)")
plt.ylabel("Final size")
if save_plot_flag:
    plt.savefig(f"../img/epidemic/R0_final_size.png",
                bbox_inches="tight",
                dpi=dpi)
    plt.close()
else:
    plt.show()


plt.figure()
plt.plot(R0_range, T_peak, label="Observed  (T)", color="green")
plt.plot(R0_range, O_peak, label="Symptomatic (T + I)", color="orange")
plt.plot(R0_range, Inc_peak, label="True (T + I + A)", color="red")
plt.legend()
plt.xlabel("Reproduction number ($\\mathcal{R}_0$)")
plt.ylabel("Peak infection")
if save_plot_flag:
    plt.savefig(f"../img/epidemic/R0_peak.png",
                bbox_inches="tight",
                dpi=dpi)
    plt.close()
else:
    plt.show()


plt.figure()
plt.plot(R0_range, T_peak, label="Observed  (T)", color="green")
plt.plot(R0_range, T, color="green", linestyle=":")
plt.legend()
plt.xlabel("Reproduction number ($\\mathcal{R}_0$)")
plt.ylabel("Proportion of population")
if save_plot_flag:
    plt.savefig(f"../img/epidemic/T_peak.png",
                bbox_inches="tight",
                dpi=dpi)
    plt.close()
else:
    plt.show()


plt.figure()
plt.plot(R0_range, B_peak, label="Peak intention to test  (B)", color="blue")
plt.legend()
plt.xlabel("Reproduction number ($\\mathcal{R}_0$)")
plt.ylabel("Proportion of population")
if save_plot_flag:
    plt.savefig(f"../img/epidemic/B_peak.png",
                bbox_inches="tight",
                dpi=dpi)
    plt.close()
else:
    plt.show()
