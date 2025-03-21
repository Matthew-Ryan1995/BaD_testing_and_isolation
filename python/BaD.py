#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:34:08 2023

Class definitions and helper functions for a BaD SIRS model.

TODO: Fix code to match document.  I have changed notation on I^*
@author: Matt Ryan
"""


# %% Packages/libraries
import math
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import json
from enum import IntEnum

# %% Class definitions


class Compartments(IntEnum):
    """
    for speed ups whilst maintaining readability of code
    """
    Sn = 0
    Sb = 1
    En = 2
    Eb = 3
    An = 4
    Ab = 5
    In = 6
    Ib = 7
    T = 8
    Rn = 9
    Rb = 10

    An_inc = 11
    Ab_inc = 12
    In_inc = 13
    Ib_inc = 14
    T_inc = 15
    incidence = 16


class bad(object):
    """
    Implementation of the SIR model with behaviour (indicate will test if symptomatic) or not states for each
    compartment.  Explicitly, we assume proportion and not counts.
    Currently assuming no demography, no death due to pathogen, homogenous mixing, transitions between
    behaviour/no behaviour determined by social influence and fear of disease.  Currently assuming FD-like "infection"
    process for testing with fear of disease.
    """

    def __init__(self, **kwargs):
        """
        Written by: Roslyn Hickson
        Required parameters when initialising this class.
        :param transmission: double, the transmission rate from those infectious to those susceptible.
        :param infectious_period: scalar, the average infectious period.
        :param immune_period: scalar, average Immunity period (for SIRS)
        :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S tests (c)
        :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I tests (p)
        :param N_social: double, social influence of non-testers on testers (a1)
        :param N_fear: double, Fear of disease for testers to not test (a2)
        :param B_social: double, social influence of testers on non-testers (w1)
        :param B_fear: double, Fear of disease for non-testers to become testers (w2)
        :param av_lifespan: scalar, average life span in years
        """
        args = self.set_defaults()  # load default values from json file
        self.update_params(**kwargs)  # update with user specified values

        # set initial conditions
        self.init_cond = self.set_initial_conditions()

    def set_defaults(self, filename="model_parameters.json"):
        """
        Written by: Roslyn Hickson
        Pull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return: loaded expected parameter values
        """
        with open(filename) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            json_data[key] = value["exp"]
        return json_data

    def update_params(self, **kwargs):
        args = kwargs
        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

        # error handle period->rate conversions
        if self.immune_period == 0:
            self.nu = 1e6  # if immune period zero, means quickly move back to susceptible
        elif self.immune_period == np.inf:
            self.nu = 0  # if immune period is larege, means no move back to susceptible
        else:
            self.nu = 1/self.immune_period

        if self.infectious_period == 0:
            self.gamma = 1e6
        elif self.infectious_period == np.inf:
            self.gamma = 0
        else:
            self.gamma = 1/self.infectious_period
        if self.latent_period == 0:
            self.sigma = 1e6
        elif self.latent_period == np.inf:
            self.sigma = 0
        else:
            self.sigma = 1/self.latent_period

    def rate_to_infect(self, Ib, In, An, Ab, T):
        """
        calculate force of infection, lambda

        Parameters
        ----------
        Ib : float
            current Ib(t)
        In : float
            current In(t)
        Ab : float
            current Ab(t)
        An : float
            current An(t)
        T : float
            current T(t)

        Returns
        -------
        lambda : float
            force of infection at time t.
        """
        I = In + Ib
        A = An + Ab
        return self.transmission * (I + self.qA * A + self.qT * T)

    def ss_rate_to_infect(self, O, A, T):
        """
        calculate force of infection, lambda, at steady state

        Parameters
        ----------
        O : float
            Total in O=I_n+I_b+T at steady state
        A : float
            Total in A=A_n+A_b at steady state
        T : float
            Total in T at steady state

        Returns
        -------
        lambda : float
            force of infection at time steady state
        """
        return self.transmission * (O-T + self.qA * A + self.qT * T)

    def rate_to_test(self, B, T):  # i.e. omega
        return self.w1 * (B) + self.w2 * (T) + self.w3

    def rate_to_no_test(self, N):  # i.e. alpha
        return self.a1 * (N) + self.a2

    def odes(self, t, prev_pop, phi=False, flag_track_incidence=False):
        """
        ODE set up to use spi.integrate.solve_ivp.  This defines the change in state at time t.

        Parameters
        ----------
        t : double
            time point.
        prev_pop : array
            State of the population at time t-1, in proportions.
            Assumes that it is of the form:
                [Sn, Sb, En, Eb, In, Ib, T, Rn, Rb]
        phi : a forcing function that multiplies the infection rate

        Returns
        -------
        Y : array
            rate of change of population compartments at time t.
        """

        Y = np.zeros((len(prev_pop)))
        total_pop = prev_pop[0:11].sum()
        assert np.isclose(total_pop, 1.0), "total population deviating from 1"

        B_total = (prev_pop[[Compartments.Sb, Compartments.Eb, Compartments.Ab,
                             Compartments.Ib, Compartments.T, Compartments.Rb]].sum())/total_pop
        T = (prev_pop[Compartments.T]) / total_pop

        lam = self.rate_to_infect(Ib=prev_pop[Compartments.Ib] / total_pop,
                                  In=prev_pop[Compartments.In] / total_pop,
                                  An=prev_pop[Compartments.An] / total_pop,
                                  Ab=prev_pop[Compartments.Ab] / total_pop,
                                  T=T)
        if phi:
            lam *= phi(t)

        omega = self.rate_to_test(B=B_total,
                                  T=T)
        alpha = self.rate_to_no_test(N=1.0-B_total)

        # Sn
        Y[Compartments.Sn] = -lam * prev_pop[Compartments.Sn] - omega * \
            prev_pop[Compartments.Sn] + alpha * \
            prev_pop[Compartments.Sb] + self.nu * prev_pop[Compartments.Rn]
        # Sb
        Y[Compartments.Sb] = -self.qB * lam * prev_pop[Compartments.Sb] + omega * \
            prev_pop[Compartments.Sn] - alpha * \
            prev_pop[Compartments.Sb] + self.nu * prev_pop[Compartments.Rb]
        # En
        Y[Compartments.En] = lam * prev_pop[Compartments.Sn] - self.sigma * prev_pop[Compartments.En] - \
            omega * prev_pop[Compartments.En] + \
            alpha * prev_pop[Compartments.Eb]
        # Eb
        Y[Compartments.Eb] = self.qB * lam * prev_pop[Compartments.Sb] - self.sigma * \
            prev_pop[Compartments.Eb] + omega * \
            prev_pop[Compartments.En] - alpha * prev_pop[Compartments.Eb]
        # An
        Y[Compartments.An] = self.pA * self.sigma * prev_pop[Compartments.En] - self.gamma * \
            prev_pop[Compartments.An] - omega * \
            prev_pop[Compartments.An] + alpha * prev_pop[Compartments.Ab]
        # Ab
        Y[Compartments.Ab] = self.pA * self.sigma * prev_pop[Compartments.Eb] - self.gamma * \
            prev_pop[Compartments.Ab] + omega * \
            prev_pop[Compartments.An] - alpha * prev_pop[Compartments.Ab]
        # In
        Y[Compartments.In] = (1.0-self.pA) * self.sigma * prev_pop[Compartments.En] - self.gamma * \
            prev_pop[Compartments.In] - omega * \
            prev_pop[Compartments.In] + alpha * prev_pop[Compartments.Ib]
        # Ib
        Y[Compartments.Ib] = (1.0-self.pA) * (1.0-self.pT) * self.sigma * prev_pop[Compartments.Eb] - self.gamma * \
            prev_pop[Compartments.Ib] + omega * \
            prev_pop[Compartments.In] - alpha * prev_pop[Compartments.Ib]
        # T
        Y[Compartments.T] = (1.0-self.pA) * self.pT * self.sigma * \
            prev_pop[Compartments.Eb] - self.gamma * prev_pop[Compartments.T]
        # Rn
        Y[Compartments.Rn] = self.gamma * (prev_pop[Compartments.An] + prev_pop[Compartments.In]) - omega * \
            prev_pop[Compartments.Rn] + alpha * \
            prev_pop[Compartments.Rb] - self.nu * prev_pop[Compartments.Rn]
        # Rb
        Y[Compartments.Rb] = self.gamma * (prev_pop[Compartments.Ab] + prev_pop[Compartments.Ib] + prev_pop[Compartments.T]) + \
            omega * prev_pop[Compartments.Rn] - alpha * \
            prev_pop[Compartments.Rb] - self.nu * prev_pop[Compartments.Rb]

        if not flag_track_incidence:
            assert np.isclose(Y.sum(), 0.0), "compartment RHSs not adding to 0"

        if flag_track_incidence:
            Y[Compartments.An_inc] = self.pA * \
                self.sigma * prev_pop[Compartments.En]
            Y[Compartments.Ab_inc] = self.pA * \
                self.sigma * prev_pop[Compartments.Eb]

            Y[Compartments.In_inc] = (1.0-self.pA) * \
                self.sigma * prev_pop[Compartments.En]
            Y[Compartments.Ib_inc] = (
                1.0-self.pA) * (1.0-self.pT) * self.sigma * prev_pop[Compartments.Eb]

            Y[Compartments.T_inc] = (1.0-self.pA) * self.pT * self.sigma * \
                prev_pop[Compartments.Eb]

            Y[Compartments.incidence] = lam * prev_pop[Compartments.Sn] + \
                self.qB * lam * prev_pop[Compartments.Sb]

        return Y

    def set_initial_conditions(self, pop_size=1, starting_I=1e-6, starting_B=None):
        """

        :param pop_size: the total population size for behavioural status
        :param num_patches: the total number of patches
        :param starting_I: the total number of starting infectious
        :param starting_E: the total number of starting exposed
        :param starting_R: the total number of starting recovereds
        :return: the initial population by patch and disease status
        """
        # todo: generalise
        # for now just assuming all infectious start in with no behaviour fixme: too many simplifying assumptions currently
        starting_population = np.zeros(11)

        if starting_B is None:
            starting_B = pop_size*(1-self.get_ss_N(Tstar=0))
            # always have some behaviour unless pre specified
            starting_B = max(starting_B, 1e-6)

        # error handling
        assert pop_size > 0, "pop_size must be > 0"
        assert starting_B < pop_size, "starting_B must be smaller than pop_size"
        assert starting_B >= 0, "starting_B must be >= 0"
        assert starting_I >= 0, "starting_I must be >= 0"
        assert starting_I < pop_size - \
            starting_B, "starting_I must be smaller than pop_size - starting_B"

        starting_Sn = pop_size - starting_I - starting_B
        assert starting_Sn >= 0, "starting_Sn must be >= 0"

        # assign values
        starting_population[Compartments.In] = starting_I
        starting_population[Compartments.Sb] = starting_B
        starting_population[Compartments.Sn] = starting_Sn

        return starting_population

    def run(self, t_start=0, t_end=200, t_step=1, t_eval=True, phi=False,
            events=[], flag_incidence_tracking=False):
        """
        Run the model and store data and time

        TO ADD: next gen matrix, equilibrium

        Parameters
        ----------
        IC : TYPE
            Initial condition vector
        t_start : TYPE
            starting time
        t_end : TYPE
            end time
        t_step : TYPE, optional
            time step. The default is 1.
        t_eval : TYPE, optional
            logical: do we evaluate for all time. The default is True.
        events:
            Can pass in a list of events to go to solve_ivp, i.e. stopping conditions
        flag_incidence_tracking:
            flag to indicate whether or not to track cumulative incidence
        phi : a forcing function that multiplies the infection rate

        Returns
        -------
        self with new data added

        """
        IC = self.init_cond

        # Set up positional arguments for odes
        args = []
        if phi:
            args.append(phi)
        else:
            args.append(False)

        if flag_incidence_tracking:
            args.append(flag_incidence_tracking)
            IC = np.concatenate((IC, np.zeros(6)))
        else:
            args.append(False)

        if t_eval:
            t_range = np.arange(
                start=t_start, stop=t_end + t_step, step=t_step)
            self.t_range = t_range

            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            t_eval=t_range,
                            events=events,
                            args=args
                            # rtol=1e-7, atol=1e-14
                            )
            self.results = res.y.T
        else:
            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            events=events,
                            args=args)
            self.results = res.y.T
        if flag_incidence_tracking:
            self.incidence = self.results[:, -1]
            self.results = self.results[:, 0:-1]

    def get_B(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.Sb, Compartments.Eb, Compartments.Ab, Compartments.Ib, Compartments.T, Compartments.Rb]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_S(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.Sn, Compartments.Sb]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_E(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.En, Compartments.Eb]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_A(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.An, Compartments.Ab]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_I(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.In, Compartments.Ib, Compartments.T]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_R(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.Rn, Compartments.Rb]], 1)
            print("Model has not been run")
            return np.nan

    def get_all_I(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, [Compartments.Ab, Compartments.Ib, Compartments.An, Compartments.In, Compartments.T]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_ss_B_a_w(self, T):
        """
        Calculate the steady state of behaviour, and alpha and omega values at steady state when T is known.

        Parameters
        ----------
        T : float
            Chosen proportion who test positive at disease equilibrium

        Returns
        -------
        B : float
            Proportion of the population performing the behaviour at equilibrium
        a : float
            alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
        w : float
            omega transition value at equilibrium (w1 * B + w2 * I + w3)
        """

        # fixme: Is this bound valid?
        # assert not (T > nu/(g + nu)), "invalid choice of I"

        if T < 1e-8:
            T = 0
        if T > 1:  # fixme: shouldn't need this hack -- if remove end up in print error line with no feasible N_opts
            T = 1
        if np.isnan(T):
            T = 0

        # Writing equation in the form C N^2 - D N + E
        C = round(self.w1 - self.a1, 8)
        D = round(self.w2 * T + self.w1 + self.w3 +
                  self.a2 - self.a1 * (1-T), 8)
        E = round(self.a2 * (1-T), 8)

        assert round(E,
                     8) >= 0, f"Condition of G(N=0)= alpha_2 * (1-T)>0 not held in get_ss_B_a_w, T={T}"
        assert round(C - D + E,
                     8) <= 0, f"Condition of G(N=1)=C-D+E<0 not held in get_ss_B_a_w, T={T}"

        if np.isclose(C, 0):
            if np.isclose(D, 0):
                N = 1
            else:
                N = E/D
        else:
            N_opt1 = round((D - np.sqrt(D**2 - 4*C*E))/(2*C), 8)
            N_opt2 = round((D + np.sqrt(D**2 - 4*C*E))/(2*C), 8)
            if N_opt1 >= 0 and N_opt1 <= 1:
                N = N_opt1
            elif N_opt2 >= 0 and N_opt2 <= 1:
                N = N_opt2
            else:
                print(
                    f"Error: no valid steady state solution found for N in get_B_a_w, N_opt1={N_opt1}, N_opt2={N_opt2}, T={T}")
                print(f"C: {C}")
                print(f"D: {D}")
                print(f"E: {E}")
                N = 0.5  # =N_opt1 would be equivalent to code before I added this to explore what was happening
        B = 1 - N
        a = self.rate_to_no_test(N)  # already defined elsewhere
        w = self.rate_to_test(B, T)

        return B, a, w

    def get_ss_SEAR(self, O):
        """
        Calculate the steady state of S, E, A, and R when I is known.

        Parameters
        ----------
        I : float
            Desired prevalence of disease at equilibrium

        Returns
        -------
        R : float
            Proportion of the population recovered at equilibrium
        S : float
            Proportion of the population susceptible at equilibrium
        """
        E = (self.gamma * O)/((1.0-self.pA) * self.sigma)
        A = (self.pA * O)/(1.0-self.pA)
        R = (self.gamma * O)/((1.0-self.pA)*self.nu)
        S = 1.0 - (E + A + O + R)

        return S, E, A, R

    def get_ss_Eb(self, S, E, a, w, lam):
        """
        Calculate the steady state of Eb when steady states for S and E are known.

        Parameters
        ----------
        S : float
            Steady state for total in S
        E : float
            Steady state for total in E
        a : float
            Steady state for alpha: movement out of behaviour
        w : float
            Steady state for omega: movement into behaviour
        lam : float
            Steady state force of infection

        Returns
        -------
        Eb : float
            Proportion of the population in Eb at equilibrium
        """
        numer = self.qB * lam * S - (self.qB * (w + self.sigma) - w) * E
        denom = (1.0-self.qB) * (a + w + self.sigma)

        return numer/denom

    def get_ss_T(self, Eb):
        """
        Calculate the steady state of T, based on parameters + steady state Eb

        Parameters
        ----------
        Eb : float
            Steady state for total in Eb

        Returns
        -------
        T : float
            Proportion of the population in T at equilibrium
        """
        return ((1-self.pA) * self.pT * self.sigma * Eb)/self.gamma

    # def solve_Eb(eb, args):

    #     params = args[0]
    #     I = args[1]

    #     S, E, A, _ = get_SEAR(I, params)

    #     T = get_T(eb, params)

    #     B, w, a = get_B_a_w(T, params)

    #     lam = params["transmission"] * (I + params["qA"]*A + params["qT"] * T)

    #     qB = params["qB"]
    #     s = 1/params["latent_period"]

    #     numer = qB*lam*S - (qB*(w + s) - w) * E
    #     denom = (1-qB) * (a + w + s)

    #     Eb = numer/denom

    #     res = eb - Eb
    #     return res

    # def get_Eb(I, E, params):

    #     eb = E/2
    #     ans = fsolve(solve_Eb, x0=[eb], args=([params, I]))

    #     Eb = ans[0]
    #     return Eb

    def get_ss_Ab(self, Eb, A, a, w):
        """
        Calculate the steady state of Ab when steady states for A and Eb are known.

        Parameters
        ----------
        Eb : float
            Steady state for Eb
        A : float
            Steady state for total in A
        a : float
            Steady state for alpha: movement out of behaviour
        w : float
            Steady state for omega: movement into behaviour

        Returns
        -------
        Ab : float
            Proportion of the population in Ab at equilibrium
        """

        numer = self.pA * self.sigma * Eb + w * A
        denom = self.gamma + a + w

        return numer/denom

    def get_ss_Ib(self, O, T, Eb, a, w):
        """
        Calculate the proportion of people infected and performing the behaviour

        Parameters
        ----------
        O : float
            Desired prevalence of disease at equilibrium
        S : float
            Prevalence of susceptible at equilibrium
        a : float
            alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
        w : float
            omega transition value at equilibrium (w1 * B + w2 * I + w3)

        Returns
        -------
        Ib : float
            Prevalence of infected performing the behaviour.

        """

        numer = (1-self.pA) * (1-self.pT) * self.sigma * Eb + w * \
            (O-T)  # todo: double check as Matt had a bonus - w*T here
        denom = self.gamma + a + w

        return numer/denom

    def get_ss_Rb(self, T, R, Ib, Ab, a, w):
        """
        Calculate steady state proportion of recovereds doing behaviour.

        Parameters
        ----------
        R : float
            Proportion of individuals who are recovered.
        Ib : float
            Proportion of infected individuals doing the behaviour.
        a : float
            alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
        w : float
            omega transition value at equilibrium (w1 * B + w2 * I + w3)

        Returns
        -------
        Rb : float
            Proportion of recovered individuals doing the behaviour.

        """
        numer = self.gamma * (Ab + Ib + T) + w * R
        denom = a + w + self.nu

        return numer/denom

    def get_steady_states(self, O, T):
        """
        Calculate the steady state vector for a given disease prevalence and set of parameters.
        Parameters
        ----------
        O : float
            Desired prevalence of disease at equilibrium; O = In +Ib + T
        T : float
            value of T at equilibrium

        Returns
        -------
        ss : numpy.array
            Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
        """

        B, a, w = self.get_ss_B_a_w(T)
        N = 1.0-B

        if O <= 0.0:
            ans = np.zeros(11)
            ans[Compartments.Sn] = N
            ans[Compartments.Sb] = B
            return ans

        S, E, A, R = self.get_ss_SEAR(O)
        lam = self.ss_rate_to_infect(O=O, A=A, T=T)
        Eb = self.get_ss_Eb(S, E, a, w, lam)
        Ab = self.get_ss_Ab(Eb, A, a, w)
        Ib = self.get_ss_Ib(O, T, Eb, a, w)
        Rb = self.get_ss_Rb(T, R, Ib, Ab, a, w)

        Sb = B - (Eb + Ab + Ib + T + Rb)

        Sn = S-Sb
        En = E-Eb
        An = A-Ab
        Rn = R-Rb

        In = N - (Sn+En+An+Rn)

        ans = np.zeros(11)
        ans[Compartments.Sn] = Sn
        ans[Compartments.En] = En
        ans[Compartments.An] = An
        ans[Compartments.In] = In
        ans[Compartments.Rn] = Rn
        ans[Compartments.Sb] = Sb
        ans[Compartments.Eb] = Eb
        ans[Compartments.Ab] = Ab
        ans[Compartments.Ib] = Ib
        ans[Compartments.T] = T
        ans[Compartments.Rb] = Rb

        return ans

    def solve_ss_T_O(self, x):
        """
        Function to numerically find the disease prevalence at equilibrium for a given set
        of model parameters.  Designed to be used in conjunction with fsolve.

        Parameters
        ----------
        x : [float, float]
            Estimated disease prevalence at equilibrium for [T, O]. T=Total observed testing, O=In+Ib+T total symptomatic prevalence.

        Returns
        -------
        res : float
            The difference between the (lambda + w)*S_n and a*S_b + nu*R_n.
        """
        T = x[0]
        O = x[1]

        # # a hack to enforce biological feasible ranges -- noting `fsolve` does *not* like this hack
        # if T>1 or T<0 or I>1 or I<0 or (T+I)>1:
        #     return [0.9, 0.9]

        _, a, w = self.get_ss_B_a_w(T=T)

        ss_n = self.get_steady_states(O=O, T=T)

        A = ss_n[Compartments.An] + ss_n[Compartments.Ab]

        # T_est = ((1-pA) * pT * s * ss_n[Compartments.Eb])/g

        lam = self.ss_rate_to_infect(O=O, A=A, T=T)

        res = [T-self.get_ss_T(Eb=ss_n[Compartments.Eb]), (lam + w) * ss_n[Compartments.Sn] -
               (a * ss_n[Compartments.Sb] + self.nu * ss_n[Compartments.Rn])]

        return np.array(res)

    def find_ss(self, init_cond=np.nan):
        """
        Calculate the steady states of the system for a given set of model parameters.  We first numerically
        solve for the disease prevalence I, then use I to find all other steady states.

        Parameters
        ----------

        Returns
        -------
        ss : numpy.array
            Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
        Istar : float
            Estimated disease prevalence at equilibrium
        """

        if np.isnan(init_cond).all():
            # self.nu/(self.gamma + self.nu) - 1e-3
            # init_o = np.random.uniform(high=0.2, size=1)
            try:
                E = (self.transmission*(1-(1-self.qA)*self.pA) - self.gamma) / (self.transmission *
                                                                                (1-(1-self.qA)*self.pA) * (self.sigma/self.nu + self.sigma/self.gamma + 1))
                assert E >= 0
                init_o = (1-self.pA)*self.sigma/self.gamma * E - 1e-3
            except:
                init_o = np.random.uniform(high=0.2, size=1)
            init_t = init_o/10
        else:
            init_t = init_cond[0]
            init_o = init_cond[1]

        stop_while_flag = True
        R0 = self.get_reproduction_number()
        c = 0
        while stop_while_flag:
            res = fsolve(self.solve_ss_T_O,
                         x0=[init_t, init_o],
                         xtol=1e-8)  # fixme: Roslyn -- plan to re calculate fsolve if getting outside feasible space

            # Is solutions are too small then set to 0
            if (res[0] < self.init_cond[Compartments.In]):
                res[0] = 0
            if (res[1] < self.init_cond[Compartments.In]):
                res[1] = 0

            if res[0] > res[1]:  # T < O
                init_o = np.random.uniform(high=0.2, size=1)
                init_t = init_o/10
            elif res.sum() < 1e-8 and R0 > 1:  # DFE unstale
                init_o = np.random.uniform(high=0.2, size=1)
                init_t = init_o/10
            elif res.sum() > 1 or res[0] > 1 or res[1] > 1:  # T, O < 1
                init_o = np.random.uniform(high=0.2, size=1)
                init_t = init_o/10
            else:
                ss = self.get_steady_states(O=res[1], T=res[0])

                ss = ss.round(10)

                J = self.get_J(ss=ss)

                eigenValues, _ = np.linalg.eig(J)
                # Ensure found ss is a solution
                if not (np.isclose(self.odes(t=0, prev_pop=ss), 0)).all():
                    init_o = np.random.uniform(high=0.2, size=1)
                    init_t = init_o/10
                elif (ss < 0).any():  # Feasibility must be positive
                    init_o = np.random.uniform(high=0.2, size=1)
                    init_t = init_o/10
                elif not np.isclose(ss.sum(), 1):  # Feasibility, must sum to 1
                    init_o = np.random.uniform(high=0.2, size=1)
                    init_t = init_o/10
                elif (np.real(eigenValues) > 0).any():  # ss must be stable
                    init_o = np.random.uniform(high=0.2, size=1)
                    init_t = init_o/10
                else:
                    stop_while_flag = False
            c += 1
            if c > 100:
                # When all else fails, calculate numerically.
                self.run(t_end=1000, t_step=0.5)
                ss = self.results[-1, :]
                res = "fsolve did not converge"
                stop_while_flag = False
        return ss, res

    def get_ss_N(self, Tstar=np.nan):

        if np.isnan(Tstar):
            Tstar = self.results[-1, Compartments.T]

        A = self.w1 - self.a1
        B = -(self.w2*Tstar + self.w1 + self.w3 + self.a2 - self.a1 * (1-Tstar))
        C = self.a2 * (1-Tstar)

        if np.isclose(A, 0):
            ans = -C/B
        else:
            ans = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)

        ans = min(1, ans)
        ans = max(0, ans)

        return ans

    def get_reproduction_number(self):

        T = 0
        B0, alpha, omega = self.get_ss_B_a_w(T)
        N0 = 1-B0

        if self.infectious_period == 0:
            gamma = 0
        else:
            gamma = 1/self.infectious_period
        if self.latent_period == 0:
            sigma = 0
        else:
            sigma = 1/self.latent_period

        saw = sigma + alpha + omega
        gaw = gamma + alpha + omega

        denom = gamma*saw*gaw

        tnAn = self.pA * (alpha * saw + gamma * (sigma + alpha)) / denom
        tnIn = (1-self.pA) * (alpha * saw + gamma *
                              (sigma + alpha) - self.pT * alpha * omega) / denom
        tnAb = self.pA * omega * (gamma + sigma + alpha + omega) / denom
        tnIb = (1-self.pA) * omega * (sigma + alpha +
                                      (1-self.pT)*(gamma + omega)) / denom
        tnT = (1-self.pA) * self.pT * omega / (gamma * saw)

        tbAn = self.pA * alpha * (gamma + sigma + alpha + omega) / denom
        tbIn = (1-self.pA) * alpha * (gamma + alpha +
                                      (1-self.pT)*(sigma + omega)) / denom
        tbAb = self.pA * (omega * saw + gamma * (sigma + omega))/denom
        tbIb = (1-self.pA)*(omega * saw + gamma * (sigma + omega) -
                            self.pT * (sigma + omega) * (gamma + omega)) / denom

        # tbAb = (self.pA * omega * (gamma + sigma + alpha + omega) +
        #         self.pA * sigma * gamma) / denom
        # tbIb = ((1-self.pA) * alpha * omega + (1-self.pA) *
        #         (1-self.pT) * (sigma*gamma + omega * (sigma + gamma + omega))) / denom
        tbT = (1-self.pA) * self.pT * (sigma + omega) / (gamma * saw)

        Lambda_N = self.transmission * \
            (tnIn + tnIb + self.qA * (tnAn + tnAb) + self.qT * tnT)
        Lambda_B = self.transmission * \
            (tbIn + tbIb + self.qA * (tbAn + tbAb) + self.qT * tbT)

        R0 = Lambda_N * N0 + self.qB * Lambda_B * B0

        return R0

    def get_J0(self, N, T, a, w, gamma):

        dwdB = self.w1
        dadN = self.a1
        dwdT = self.w2

        J0 = np.array(
            [[dwdB * N + dadN * (1-N - T) - w - a, -dwdT * N - a],
             [0, -gamma]]
        )

        return J0

    def get_Jn0(self, ss, nu):

        dwdB = self.w1
        dadN = self.a1
        dwdT = self.w2

        Sn = ss[Compartments.Sn]
        En = ss[Compartments.En]
        An = ss[Compartments.An]
        In = ss[Compartments.In]

        Sb = ss[Compartments.Sb]
        Eb = ss[Compartments.Eb]
        Ab = ss[Compartments.Ab]
        Ib = ss[Compartments.Ib]

        Jn0 = np.array(
            [
                [dwdB * Sn + dadN * Sb + nu, -self.qT *
                    self.transmission * Sn - dwdT * Sn],
                [dwdB * En + dadN * Eb, self.qT *
                    self.transmission * Sn - dwdT * En],
                [dwdB * An + dadN * Ab, -dwdT * An],
                [dwdB * In + dadN * Ib, -dwdT * In]
            ]
        )

        return Jn0

    def get_Jb0(self, ss, nu):

        dwdB = self.w1
        dadN = self.a1
        dwdT = self.w2

        Sn = ss[Compartments.Sn]
        En = ss[Compartments.En]
        An = ss[Compartments.An]
        In = ss[Compartments.In]

        Sb = ss[Compartments.Sb]
        Eb = ss[Compartments.Eb]
        Ab = ss[Compartments.Ab]
        Ib = ss[Compartments.Ib]

        Jb0 = np.array(
            [
                [-dwdB * Sn - dadN * Sb - nu, -self.qB * self.qT *
                    self.transmission * Sb + dwdT * Sn - nu],
                [-dwdB * En - dadN * Eb, self.qB * self.qT *
                    self.transmission * Sb + dwdT * En],
                [-dwdB * An - dadN * Ab, dwdT * An],
                [-dwdB * In - dadN * Ib, dwdT * In]
            ]
        )

        return Jb0

    def get_Jnn(self, lam, w, nu, sigma, gamma, Sn):

        Jnn = np.array(
            [
                [-lam - w - nu, -nu, -self.qA*self.transmission *
                    Sn - nu, -self.transmission*Sn - nu],
                [lam, -sigma - w, self.qA*self.transmission*Sn, self.transmission*Sn],
                [0, self.pA*sigma, -gamma - w, 0],
                [0, (1-self.pA)*sigma, 0, -gamma - w]
            ]
        )

        return Jnn

    def get_Jnb(self, a, Sn):

        Jnb = np.identity(4) * a

        Jnb[0, 2] = -self.qA * self.transmission * Sn
        Jnb[1, 2] = self.qA * self.transmission * Sn
        Jnb[0, 3] = -self.transmission * Sn
        Jnb[1, 3] = self.transmission * Sn

        return Jnb

    def get_Jbn(self, w, Sb):

        Jbn = np.identity(4) * w

        Jbn[0, 2] = -self.qB * self.qA * self.transmission * Sb
        Jbn[1, 2] = self.qB * self.qA * self.transmission * Sb
        Jbn[0, 3] = -self.qB * self.transmission * Sb
        Jbn[1, 3] = self.qB * self.transmission * Sb

        return Jbn

    def get_Jbb(self, lam, a, nu, sigma, gamma, Sb):

        Jbb = np.array(
            [
                [-self.qB*lam - a - nu, -nu, -self.qB*self.qA*self.transmission *
                    Sb - nu, -self.qB*self.transmission*Sb - nu],
                [self.qB*lam, -sigma - a, self.qA*self.qB *
                    self.transmission*Sb, self.qB*self.transmission*Sb],
                [0, self.pA*sigma, -gamma - a, 0],
                [0, (1-self.pA)*(1-self.pT)*sigma, 0, -gamma - a]
            ]
        )

        return Jbb

    def get_J(self, ss=np.nan, numeric=False):

        if np.isnan(ss).any():
            if numeric:
                ss = self.results[-1, :]
            else:
                ss, _ = self.find_ss()

        if self.infectious_period == 0:
            gamma = 0
        else:
            gamma = 1/self.infectious_period
        if self.latent_period == 0:
            sigma = 0
        else:
            sigma = 1/self.latent_period
        if self.immune_period == 0:
            nu = 0
        else:
            nu = 1/self.immune_period

        N = ss[[Compartments.Sn, Compartments.En,
                Compartments.An, Compartments.In,
                Compartments.Rn]].sum()

        Sn = ss[Compartments.Sn]
        En = ss[Compartments.En]
        An = ss[Compartments.An]
        In = ss[Compartments.In]

        Sb = ss[Compartments.Sb]
        Eb = ss[Compartments.Eb]
        Ab = ss[Compartments.Ab]
        Ib = ss[Compartments.Ib]
        T = ss[Compartments.T]

        _, a, w = self.get_ss_B_a_w(T)
        lam = self.rate_to_infect(Ib, In, An, Ab, T)

        J0 = self.get_J0(N, T, a, w, gamma)

        J0n = np.zeros((2, 4))

        J0b = np.zeros((2, 4))
        J0b[1, 1] = (1-self.pA) * self.pT * sigma

        Jn0 = self.get_Jn0(ss, nu)
        Jb0 = self.get_Jb0(ss, nu)

        Jnn = self.get_Jnn(lam, w, nu, sigma, gamma, Sn)
        Jbb = self.get_Jbb(lam, a, nu, sigma, gamma, Sb)

        Jbn = self.get_Jbn(w, Sb)
        Jnb = self.get_Jnb(a, Sn)

        J = np.block(
            [
                [J0, J0n, J0b],
                [Jn0, Jnn, Jnb],
                [Jb0, Jbn, Jbb]
            ]
        )

        return J

    def get_effective_reproduction_number(self):

        if hasattr(self, 'results'):
            T = self.results[:, Compartments.T]

            B = self.get_B()
            omega = self.rate_to_test(B=B, T=T)
            alpha = self.rate_to_no_test(N=1-B)
            N0 = self.results[:, Compartments.Sn]
            B0 = self.results[:, Compartments.Sb]

            if self.infectious_period == 0:
                gamma = 0
            else:
                gamma = 1/self.infectious_period
            if self.latent_period == 0:
                sigma = 0
            else:
                sigma = 1/self.latent_period

            saw = sigma + alpha + omega
            gaw = gamma + alpha + omega

            denom = gamma*saw*gaw

            tnAn = self.pA * (alpha * saw + gamma * (sigma + alpha)) / denom
            tnIn = (1-self.pA) * (alpha * saw + gamma *
                                  (sigma + alpha) - self.pT * alpha * omega) / denom
            tnAb = self.pA * omega * (gamma + sigma + alpha + omega) / denom
            tnIb = (1-self.pA) * omega * (sigma + alpha +
                                          (1-self.pT)*(gamma + omega)) / denom
            tnT = (1-self.pA) * self.pT * omega / (gamma * saw)

            tbAn = self.pA * alpha * (gamma + sigma + alpha + omega) / denom
            tbIn = (1-self.pA) * alpha * (gamma + alpha +
                                          (1-self.pT)*(sigma + omega)) / denom
            tbAb = self.pA * (omega * saw + gamma * (sigma + omega))/denom
            tbIb = (1-self.pA)*(omega * saw + gamma * (sigma + omega) -
                                self.pT * (sigma + omega) * (gamma + omega)) / denom

            # tbAb = (self.pA * omega * (gamma + sigma + alpha + omega) +
            #         self.pA * sigma * gamma) / denom
            # tbIb = ((1-self.pA) * alpha * omega + (1-self.pA) *
            #         (1-self.pT) * (sigma*gamma + omega * (sigma + gamma + omega))) / denom
            tbT = (1-self.pA) * self.pT * (sigma + omega) / (gamma * saw)

            Lambda_N = self.transmission * \
                (tnIn + tnIb + self.qA * (tnAn + tnAb) + self.qT * tnT)
            Lambda_B = self.transmission * \
                (tbIn + tbIb + self.qA * (tbAn + tbAb) + self.qT * tbT)

            R0 = Lambda_N * N0 + self.qB * Lambda_B * B0

            return R0

        else:
            print("Model has not been run")
            return np.nan


# %% functions external to class


def load_param_defaults(filename="model_parameters.json"):
    """
    Written by: Rosyln Hickson
    Pull out default values from a file in json format.
    :param filename: json file containing default parameter values, which can be overridden by user specified values
    :return: loaded expected parameter values
    """
    with open(filename) as json_file:
        json_data = json.load(json_file)
    for key, value in json_data.items():
        json_data[key] = value["exp"]
    return json_data


# %%


if __name__ == "__main__":

    # set up parameter values for the simulations
    flag_use_defaults = True
    flag_save_figs = False
    num_days_to_run = 100

    cust_params = load_param_defaults()
    if not flag_use_defaults:  # version manually overriding values in json file
        w1 = 8
        R0 = 5
        gamma = 1/7
        sigma = 1/3

        cust_params = load_param_defaults()
        cust_params["transmission"] = R0*gamma
        cust_params["infectious_period"] = 1/gamma
        cust_params["immune_period"] = 240
        cust_params["latent_period"] = 1/sigma  # Turning off demography

        cust_params["a1"] = cust_params["a1"]*gamma
        cust_params["w1"] = cust_params["w1"]*gamma
        cust_params["a2"] = cust_params["a1"]*gamma
        cust_params["w2"] = cust_params["w2"]*gamma
        cust_params["w3"] = cust_params["w3"]*gamma

    M1 = bad(**cust_params)

    M1.run(t_end=num_days_to_run)

# %% Steady state plots

    # height=7
    dpi = 300

    ss, res = M1.find_ss()

    plt.figure()
    plt.title("Susceptibles")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Sn],
             label="$S_N$",
             color="blue",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Sn], ss[Compartments.Sn]],
             color="blue",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Sb],
             label="$S_B$",
             color="cornflowerblue",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Sb], ss[Compartments.Sb]],
             color="cornflowerblue",
             linestyle=":")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time (days)")
    plt.legend()
    if flag_save_figs:
        plt.savefig("../img/ss_susceptibles.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Exposed")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.En],
             label="$E_N$",
             color="orange",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.En], ss[Compartments.En]],
             color="orange",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Eb],
             label="$E_B$",
             color="darkorange",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Eb], ss[Compartments.Eb]],
             color="darkorange",
             linestyle=":")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time (days)")
    plt.legend()
    if flag_save_figs:
        plt.savefig("../img/ss_exposed.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Infectious")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.An],
             label="$A_N$",
             color="peru",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.An], ss[Compartments.An]],
             color="peru",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Ab],
             label="$A_B$",
             color="peachpuff",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Ab], ss[Compartments.Ab]],
             color="peachpuff",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.In],
             label="$I_N$",
             color="red",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.In], ss[Compartments.In]],
             color="red",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Ib],
             label="$I_B$",
             color="darkred",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Ib], ss[Compartments.Ib]],
             color="darkred",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.T],
             label="$T$",
             color="lightcoral",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.T], ss[Compartments.T]],
             color="lightcoral",
             linestyle=":")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time (days)")
    plt.legend()
    if flag_save_figs:
        plt.savefig("../img/ss_infectious.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title("Recovereds")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Rn],
             label="$R_N$",
             color="green",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Rn], ss[Compartments.Rn]],
             color="green",
             linestyle=":")
    plt.plot(M1.t_range,
             M1.results[:, Compartments.Rb],
             label="$R_B$",
             color="darkgreen",
             linestyle="-")
    plt.plot([M1.t_range[0], M1.t_range[-1]],
             [ss[Compartments.Rb], ss[Compartments.Rb]],
             color="darkgreen",
             linestyle=":")
    plt.ylabel("Proportion of population")
    plt.xlabel("Time (days)")
    plt.legend()
    if flag_save_figs:
        plt.savefig("../img/ss_recovereds.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# %% Phase space: SI

    plt.figure()
    plt.title("S-I phase plot")
    plt.plot(M1.results[:, Compartments.Sn],
             M1.results[:, Compartments.In],
             label="$N$",
             color="blue",
             linestyle="-")
    plt.plot(M1.results[:, Compartments.Sb],
             M1.results[:, Compartments.Ib],
             label="$B$",
             color="red",
             linestyle="-")
    plt.plot(M1.results[:, Compartments.Sb] + M1.results[:, Compartments.Sn],
             M1.results[:, Compartments.Ib] + M1.results[:,
                                                         Compartments.In] + M1.results[:, Compartments.T],
             label="$Total$",
             color="gray",
             linestyle="-")

    plt.ylabel("I")
    plt.xlabel("S")
    plt.legend()
    if flag_save_figs:
        plt.savefig("../img/ss_susceptibles.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# %% Phase space: BI

    plt.figure()
    plt.title("B-I phase plot")
    # plt.plot(M1.get_B(),
    #          M1.get_I(),
    #          label="All I",
    #          color="gray",
    #          linestyle="-")

    plt.ylabel("Infection Prevalence")
    plt.xlabel("Behaviour prevalence")
    plt.legend()

    plt.plot(M1.get_B(),
             M1.results[:, Compartments.Ib],
             label="$I_B$",
             color="red",
             linestyle="-")

    plt.plot(M1.get_B(),
             M1.results[:, Compartments.T],
             label="$T$",
             color="blue",
             linestyle="-")

    plt.ylabel("Infection Prevalence")
    plt.xlabel("Behaviour prevalence")
    plt.legend()

    if flag_save_figs:
        plt.savefig("../img/ss_susceptibles.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# %% Bifurcation diagram
 # fixme: Roslyn
    R0_min = 0
    R0_max = 10
    R0_step = 0.1
    R0_range = np.arange(R0_min, R0_max + R0_step, step=R0_step)

    pA = cust_params["pA"]
    qA = cust_params["qA"]
    gamma = 1/cust_params["infectious_period"]

    res = []
    # save = False
    M2 = bad(**cust_params)

    for r0 in R0_range:
        beta = r0 * gamma / ((pA*qA + 1 - pA))
        tmp_params = dict(cust_params)
        tmp_params["transmission"] = beta
        M2.update_params(**tmp_params)
        ss, tmp = M2.find_ss()
        res.append(ss)

    res = np.array(res)

    plt.figure()
    plt.title("Susceptible and recovered")
    plt.plot(R0_range,
             res[:, Compartments.Sn],
             color="blue",
             label="$S_N$")
    plt.plot(R0_range,
             res[:, Compartments.Sb],
             color="cornflowerblue",
             label="$S_B$")
    plt.plot(R0_range,
             res[:, Compartments.Rn],
             color="green",
             label="$R_N$")
    plt.plot(R0_range,
             res[:, Compartments.Rb],
             color="darkgreen",
             label="$R_B$")
    plt.ylabel("Steady state value")
    plt.xlabel("'Behaviour free' reproduction number")
    plt.legend(loc=[1.05, 0.])
    if flag_save_figs:
        plt.savefig("../img/ss_by_r0_susceptible.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    # plt.figure()
    # plt.title("Exposed and infectious")
    # plt.plot(R0_range,
    #          res[:, Compartments.En],
    #          color="orange",
    #          label="$E_N$")
    # plt.plot(R0_range,
    #          res[:, Compartments.Eb],
    #          color="darkorange",
    #          label="$E_B$")
    # plt.plot(R0_range,
    #          res[:, Compartments.An],
    #          color="peru",
    #          label="$A_N$")
    # plt.plot(R0_range,
    #          res[:, Compartments.Ab],
    #          color="peachpuff",
    #          label="$A_B$")
    # plt.plot(R0_range,
    #          res[:, Compartments.In],
    #          color="red",
    #          label="$I_N$")
    # plt.plot(R0_range,
    #          res[:, Compartments.Ib],
    #          color="darkred",
    #          label="$I_B$")
    # plt.plot(R0_range,
    #          res[:, Compartments.T],
    #          color="lightcoral",
    #          label="$T$")

    # plt.ylabel("Steady state value")
    # plt.xlabel("'Behaviour free' reproduction number")
    # plt.legend(loc=[1.05, 0.])
    # if flag_save_figs:
    #     plt.savefig("../img/ss_by_r0_infectious.png",
    #                 dpi=dpi,
    #                 bbox_inches="tight")
    #     plt.close()
    # else:
    #     plt.show()

# %% Testing accuracy
    pT_min = 0
    pT_max = 1
    pT_step = 0.01
    pT_range = np.arange(pT_min, pT_max + pT_step, step=pT_step)

    res = []
    # save = False
    tmp_params = dict(cust_params)
    M3 = bad(**tmp_params)

    for pt in pT_range:
        tmp_params["pT"] = pt
        M3.update_params(**tmp_params)
        ss, _ = M3.find_ss()

        res.append(ss)

    res = np.array(res)

    plt.figure()
    plt.plot(pT_range,
             res[:, [Compartments.Sb, Compartments.Eb, Compartments.Ab,
                     Compartments.Ib, Compartments.Rb, Compartments.T]].sum(1),
             label="behaviour")
    plt.plot(pT_range,
             res[:, [Compartments.An, Compartments.In, Compartments.Ab,
                     Compartments.Ib, Compartments.T]].sum(1),
             label="infection")

    plt.ylabel("Proportion population")
    plt.xlabel("p_T")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(res[:, [Compartments.Sb, Compartments.Eb, Compartments.Ab,
                     Compartments.Ib, Compartments.Rb, Compartments.T]].sum(1),
             res[:, [Compartments.An, Compartments.In, Compartments.Ab,
                     Compartments.Ib, Compartments.T]].sum(1))
    plt.xlabel("Behaviour")
    plt.ylabel("Infection")
    plt.title("I vs B for differnt p_T")
    plt.show()


# # %% Check  Bstar

#     print("Checking B star")
#     T = M1.results[-1, Compartments.T]

#     B, a, w = get_B_a_w(T, cust_params)

#     print(f"SS Estimate: {B}")
#     print(f"numeric estimate: {M1.get_B()[-1]}")

#     plt.figure()
#     plt.title("Bstar estimate")
#     plt.plot(M1.t_range, M1.get_B(), label="Behaviour")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [B, B], ":k")
#     plt.legend()
#     plt.show()

# # %% Check S, E, A, R

#     I = M1.get_I()[-1]

#     S, E, A, R = get_SEAR(I, cust_params)

#     print("Checking S star")

#     print(f"SS Estimate: {S}")
#     print(f"numeric estimate: {M1.get_S()[-1]}")

#     plt.figure()
#     plt.title("Sstar estimate")
#     plt.plot(M1.t_range, M1.get_S(), label="Susceptibles")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [S, S], ":k")
#     plt.legend()
#     plt.show()

#     print("Checking E star")

#     print(f"SS Estimate: {E}")
#     print(f"numeric estimate: {M1.get_E()[-1]}")

#     plt.figure()
#     plt.title("Estar estimate")
#     plt.plot(M1.t_range, M1.get_E(), label="Exposed")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [E, E], ":k")
#     plt.legend()
#     plt.show()

#     print("Checking A star")

#     print(f"SS Estimate: {A}")
#     print(f"numeric estimate: {M1.get_A()[-1]}")

#     plt.figure()
#     plt.title("Astar estimate")
#     plt.plot(M1.t_range, M1.get_A(), label="Asymptomatic")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [A, A], ":k")
#     plt.legend()
#     plt.show()

#     print("Checking R star")

#     print(f"SS Estimate: {R}")
#     print(f"numeric estimate: {M1.get_R()[-1]}")

#     plt.figure()
#     plt.title("Rstar estimate")
#     plt.plot(M1.t_range, M1.get_R(), label="Recovered")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [R, R], ":k")
#     plt.legend()
#     plt.show()

# # %% Check Eb
#     lam = cust_params["transmission"] * \
#         ((I-T) + cust_params["qA"]*A + cust_params["qT"] * T)

#     Eb = get_Eb(S, E, a, w, lam, cust_params)

#     print("Checking Eb star")

#     print(f"SS Estimate: {Eb}")
#     print(f"numeric estimate: {M1.results[-1, Compartments.Eb]}")

#     plt.figure()
#     plt.title("EBstar estimate")
#     plt.plot(M1.t_range, M1.results[:, Compartments.Eb], label="Eb")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [Eb, Eb], ":k")
#     plt.legend()
#     plt.show()

# # %% Check Ab

#     # Eb = M1.results[-1, Compartments.Eb]

#     Ab = get_Ab(Eb, A, a, w, cust_params)

#     print("Checking Ab star")

#     print(f"SS Estimate: {Ab}")
#     print(f"numeric estimate: {M1.results[-1, Compartments.Ab]}")

#     plt.figure()
#     plt.title("ABstar estimate")
#     plt.plot(M1.t_range, M1.results[:, Compartments.Ab], label="Ab")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [Ab, Ab], ":k")
#     plt.legend()
#     plt.show()

# # %% Check Ib

#     Ib = get_Ib(I, T, Eb, a, w, cust_params)

#     print("Checking Ib star")

#     print(f"SS Estimate: {Ib}")
#     print(f"numeric estimate: {M1.results[-1, Compartments.Ib]}")

#     plt.figure()
#     plt.title("IBstar estimate")
#     plt.plot(M1.t_range, M1.results[:, Compartments.Ib], label="Ib")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [Ib, Ib], ":k")
#     plt.legend()
#     plt.show()

# # %% Check Rb

#     Rb = get_Rb(T, R, Ib, Ab, a, w, cust_params)

#     print("Checking Rb star")

#     print(f"SS Estimate: {Rb}")
#     print(f"numeric estimate: {M1.results[-1, Compartments.Rb]}")

#     plt.figure()
#     plt.title("Rbstar estimate")
#     plt.plot(M1.t_range, M1.results[:, Compartments.Rb], label="Rb")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [Rb, Rb], ":k")
#     plt.legend()
#     plt.show()


# # %%
#     ss, res = find_ss(cust_params)


# # %% Check I and T

#     plt.figure()
#     plt.title("I")
#     plt.plot(M1.t_range, M1.get_I(), label="infectious")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [res[1], res[1]], ":k")
#     plt.legend()
#     plt.show()

#     plt.figure()
#     plt.title("T")
#     plt.plot(M1.t_range, M1.results[:, Compartments.T], label="T")
#     plt.plot([M1.t_range[0], M1.t_range[-1]], [res[0], res[0]], ":k")
#     plt.legend()
#     plt.show()
