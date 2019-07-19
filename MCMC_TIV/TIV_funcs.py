import pandas as pd
import numpy as np
import scipy as sci
import scipy.stats as st
from scipy.integrate import solve_ivp
from scipy import stats as st
import matplotlib.pyplot as plt

'''
SIR_model takes given parameters [beta, gamma]
and returns solution to SIR differential equations as dictionary where t:[time], y:[S], [I], [R] at each time point (each day in 52 weeks)
'''

def TIV_model(param):  # Where param = [pV, beta]
    # Parameters
    pV = param[0]
    beta = param[1]
    betap = 3e-8 # Leaving this set for now to see if mcmc can find just two parameters first

    # Initial conditions
    T0 = 7e+7 #4e+8
    I0 = 0
    V0 = 1 # 1e+4

    gT = 0.8

    deltaV = 5
    deltaI = 2

    time = np.linspace(0, 11, 11*12)
    y_init = [T0, I0, V0]

    # TIV differential equations
    def TIV_rhs(t, y):
        T, I, V = y
        return [gT * T * (1 - (T + 1) / T0) - (beta * V * T),
                beta * T * V - (deltaI * I),
                pV * I - (deltaV * V) - (beta * V * T)]

    # Solve TIV
    sol = solve_ivp(TIV_rhs, [time[0], time[-1]], y_init, method='BDF', t_eval=time)

    return sol

def TLIV_model(param):
    # Parameters
    pV = param[0]
    beta = param[1]
    gamma = 2
    betap = 3e-8 # Leaving this set for now to see if mcmc can find just two parameters first

    # Initial conditions
    T0 = 7e+7 #4e+8
    L0 = 0
    I0 = 0
    V0 = 1 # 1e+4

    gT = 0.8

    deltaV = 5
    deltaI = 2

    time = np.linspace(0, 9, 9*12)
    y_init = [T0, L0, I0, V0]

    # TIV differential equations
    def TLIV_rhs(t, y):
        T, L, I, V = y
        return [gT * T * (1 - (T+1)/T0) - (beta * V * T),
                (beta * T * V) - (gamma * L),
                (gamma * L) - (deltaI * I),
                (pV * I) - (deltaV * V) - (beta * V * T)]

    # Solve TIV
    sol = solve_ivp(TLIV_rhs, [time[0], time[-1]], y_init, method='BDF', t_eval=time)

    return sol

def TIV_ll(V_data, param):  # Where I_data = I (infected individuals) as retrieved from data
    V_model = TIV_model(param).y[2]  # Obtain model values for I, given new parameters
    exp_sd = 1.5  # sd of experimental measurements of viral titre
    ll = 0

    for k in range(len(V_data)):
        new_ll = st.norm.logpdf(V_data[k], loc=V_model[k], scale=exp_sd)  # norm.logpdf(i, loc=mu, scale=sd)
        print('new_ll = ' + str(new_ll))
        ll = ll + new_ll

    return ll

'''
    plt.figure()
    plt.plot(model_data.t, I_model, label='model')
    plt.plot(model_data.t, I_data, '--', label='data')
    plt.legend()
    plt.ylabel('I')
    plt.show()
'''
    #      if np.isnan(ll):  # can't move from -inf
#            print(str(k) + ' was a nan ')


def TLIV_ll(V_data, param):  # Where I_data = I (infected individuals) as retrieved from data
    V_model = TLIV_model(param).y[2]  # Obtain model values for I, given new parameters
    exp_sd = 1.5  # sd of experimental measurements of viral titre
    ll = 0

    for k in range(len(V_data)):
        new_ll = st.norm.logpdf(V_data[k], loc=V_model[k], scale=exp_sd)  # norm.logpdf(i, loc=mu, scale=sd)
        print('new_ll = ' + str(new_ll))
        ll = ll + new_ll

    return ll

'''
    plt.figure()
    plt.plot(model_data.t, I_model, label='model')
    plt.plot(model_data.t, I_data, '--', label='data')
    plt.legend()
    plt.ylabel('I')
    plt.show()
'''
    #      if np.isnan(ll):  # can't move from -inf
#            print(str(k) + ' was a nan ')
