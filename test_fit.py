import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import TIV_funcs

'''
Plots actual data against model obtained with MCMC-derived parameters
'''

def test_fit(model, real_data, MCMC_param, max_time):  # real_data = [time, viral load], MCMC_param = [pV, data]
    # Generate model data with MCMC derived parameters
    print('max_time = ' + str(max_time))
    sol = model(MCMC_param, max_time)
    #sol = TIV_funcs.TLIV_model(MCMC_param)

    # TODO: Remove when not testing Baccam
    # Parameters
    pV = 0.042
    beta = 6.3e-6
    deltaV = 3.1
    deltaI = 2.8

    # Initial conditions
    T0 = 4e+8  # Fixing this parameter
    I0 = 0
    V0 = 0.91  # param[5]

    MCMC_param = [pV, beta, deltaV, deltaI, V0]
    Baccam_sol = model(MCMC_param, max_time)


    # Scatter plot with actual data
    x = real_data[0]
    y = real_data[1]

    '''
    #plt.figure()
    plt.plot(x, y, 'o', color='black')

    # Plot model data as a line in the same plot as the actual data
    plt.plot(sol.t, sol.y[2], '-', color='blue')
    plt.plot(Baccam_sol.t, Baccam_sol.y[2], '-', color='red')
 #  plt.plot(sol.t, sol.y[1], '-', color='red')

    plt.ylabel('Viral Titre (log TCID50/ml)')
    plt.xlabel('Days')

    '''
    fig1, ax1 = plt.subplots()
    ax1.plot(sol.t, sol.y[2], '-', color='blue')
    ax1.plot(x, y, 'o', color='black')
    ax1.plot(Baccam_sol.t, Baccam_sol.y[2], '-', color='red')

    ax1.set_title('Viral Load')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Viral load (TCID_{50})")
    ax1.set_yscale('log')


    plt.savefig(str(MCMC_param) + '_Vcurve' + '.png')  # Save as .png

    return plt.show()  # TODO: Try

# Commenting out, because it keeps producing the graph when importing the file
'''
# Keep data as is
#V_data = [0.5, 0.5, 0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]

# Adjust data
V_data = [0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]
#V_data = [3.16227766, 31622.7766, 681291.546, 31622.7766,3162.27766, 3162.27766, 681.291546, 3.16227766, 3.16227766]
time = range(0, len(V_data))

plc_n1 = np.array([time, V_data])

# Testing using param from (Yan et al, 2006) -- uncomment init_param
pV = 12.6
beta = 5e-7
deltaV = 8
deltaI = 2
#gT = 0.8
V0 = 1

param_names = ['pV', 'beta', 'deltaV', 'deltaI', 'V0']
MCMC_param = [pV, beta, deltaV, deltaI, V0]
#MCMC_param = [0.16745988115263752, 6.3e-06, 3.1, 2.8, 0.91]
MCMC_param = [2.62513049e+00, 6.30000000e-06, 8.44067659e-03, 1.56867778e+00, 3.94820664e+01]

#MCMC_param = [49.7572602, 9.66321761, 173.35151571, 28.58805245, 64.94998718]

MCMC_param = [18, 5,3e-7, 11, 10, 1000]
MCMC_param = [4.18684776e+00, 6.30000392e-06, 3.10000000e+00, 2.78643243e+00, 9.10000198e-01]
#test_fit(TIV_funcs.TIV_model, plc_n1, MCMC_param, len(V_data))
'''
# Test known data
#V_data = [3.5, 5.5, 6.5, 5.5, 3.5, 4.0, 0.5, 0.5]
V_data = [3162.3, 316227.766, 3162277.66, 316227.766, 3162.3, 10000, 3.16227766, 3.16227766]
time = range(0, len(V_data))

baccam_n4 = np.array([time, V_data])
#log(4.)

# Testing using param from (Baccam et al, 2006) -- uncomment init_param

# Parameters
pV = 0.042
beta = 6.3e-6
deltaV = 3.1
deltaI = 2.8

# Initial conditions
T0 = 4e+8  # Fixing this parameter
I0 = 0
V0 = 0.91  # param[5]

MCMC_param = [pV, beta, deltaV, deltaI, V0]
MCMC_param = [0.099233, 0.000003, 4.999441, 4.998799, 1.202998]
test_fit(TIV_funcs.TIV_model_Baccam, baccam_n4, MCMC_param, len(V_data))
