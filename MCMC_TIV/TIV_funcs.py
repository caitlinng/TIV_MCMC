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

def TIV_model(param, max_time):  # Where param = [pV, beta]
    # Parameters
    pV = param[0]
    beta = param[1]
    deltaV = param[2]
    deltaI = param[3]
    gT = param[4]

    # Initial conditions
    T0 = 7e+7  # Fixing this parameter
    I0 = 0
    V0 = param[5]

    time_points = np.arange(0, max_time, 1)  # Creates array of time points (e.g. Day 1, 2, 3, etc.)
    y_init = [T0, I0, V0]

    # TIV differential equations
    def TIV_rhs(t, y):
        T, I, V = y
        return [gT * T * (1 - (T + I) / T0) - (beta * V * T),
                beta * T * V - (deltaI * I),
                pV * I - (deltaV * V) - (beta * V * T)]

    # Solve TIV
    sol = solve_ivp(TIV_rhs, t_span=(0, max_time), y0=y_init, method='BDF', t_eval=time_points)

    return sol

def TLIV_model(param, max_time):
    # Parameters
    pV = param[0]
    beta = param[1]
    deltaV = param[2]
    deltaI = param[3]  # 2
    gT = param[4]  # 0.8

    gamma = param[6]

    # Initial conditions
    T0 = 7e+7  # Fixing this parameter
    L0 = 0
    I0 = 0
    V0 = param[5]

    time_points = np.arange(0, max_time, 1)  # Creates array of time points (e.g. Day 1, 2, 3, etc.)
    y_init = [T0, L0, I0, V0]

    # TIV differential equations
    def TLIV_rhs(t, y):
        T, L, I, V = y
        return [gT * T * (1 - (T+L+I)/T0) - (beta * V * T),
                (beta * T * V) - (gamma * L),
                (gamma * L) - (deltaI * I),
                (pV * I) - (deltaV * V) - (beta * V * T)]

    # Solve TIV
    sol = solve_ivp(TLIV_rhs, t_span=(0, max_time), y0=y_init, method='BDF', t_eval=time_points)

    return sol


def TIV_ll(V_data, param, max_time):  # Where I_data = I (infected individuals) as retrieved from data

    ll = 0

    # TODO: Try to reject unsolved parameter solutions. Should theoretically stop throwing up IndexError
    if TIV_model(param, max_time).success == False:  #Reject parameter if solve_ivp failed
        ll = -1 * np.inf
        print('solve_ivp failed for param ' + str(param) + ' so rejecting value')
        return ll

    V_model = TIV_model(param, max_time).y[2]  # Obtain model values for I, given new parameters
    exp_sd = 1.5  # sd of experimental measurements of viral titre

    for k in range(len(V_data)):
        new_ll = st.norm.logpdf(V_data[k], loc=V_model[k], scale=exp_sd)  # norm.logpdf(i, loc=mu, scale=sd)
        ll = ll + new_ll

    return ll

'''
        try:
            new_ll = st.norm.logpdf(V_data[k], loc=V_model[k], scale=exp_sd)  # norm.logpdf(i, loc=mu, scale=sd)
            ll = ll + new_ll

        except IndexError:
            print('IndexError occurred with ll ' + str(ll))
            print('k = ' + str(k))
            print('V_data = ' + str(V_data))
            print('V_model = ' + str(V_model))
            print('param = ' + str(param))
            ll = -1 * np.inf  # Rejecting parameter if solve_ivp could not solve
'''



def TLIV_ll(V_data, param):  # Where I_data = I (infected individuals) as retrieved from data
    V_model = TLIV_model(param).y[2]  # Obtain model values for I, given new parameters
    exp_sd = 1.5  # sd of experimental measurements of viral titre
    ll = 0

    for k in range(len(V_data)):
        try:
            new_ll = st.norm.logpdf(V_data[k], loc=V_model[k], scale=exp_sd)  # norm.logpdf(i, loc=mu, scale=sd)
            ll = ll + new_ll

        except IndexError:
            print('IndexError occured')
            print('k = ' + str(k))
            print('V_data = ' + str(V_data))
            print('V_model = ' + str(V_model))
            print('param = ' + str(param))
            ll = -1 * np.inf

    return ll


'''
tune_w()
Checks acceptance rate for this parameter (need to input param-specific values and modifies w accordingly
    If too low (<0.3) multiply parameter's w by 0.75
    If too high (>0.5), multiply w by 1.5
Returns w
'''


def tune_w(accepts, current_iter, w):
    print('w was ' + str(w))

    acceptance_rate = accepts / current_iter

    if acceptance_rate < 0.3:
        w *= 0.75

    if acceptance_rate > 0.5:
        w *= 1.5

    print('Tuned w is now ' + str(w))
    print('Acceptance rate = ' + str(acceptance_rate))

    return w