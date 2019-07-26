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

    # Scatter plot with actual data
    x = real_data[0]
    y = real_data[1]

    #plt.figure()
    plt.plot(x, y, 'o', color='black')

    # Plot model data as a line in the same plot as the actual data
    plt.plot(sol.t, sol.y[2], '-', color='blue')

    plt.ylabel('Viral Titre (log TCID50/ml)')
    plt.xlabel('Days')

    '''
    fig1, ax1 = plt.subplots()
    ax1.plot(sol.t, sol.y[2], x, y, 'o', color='black')
    #ax1.plot(x, y, 'o', color='black')
    ax1.set_title('Viral Load')
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Viral load (TCID_{50})")
    ax1.set_yscale('log')
    '''

    plt.savefig(str(MCMC_param) + '_Vcurve' + '.png')  # Save as .png

    return plt.show()  # TODO: Try

# Commenting out, because it keeps producing the graph when importing the file

# Keep data as is
#V_data = [0.5, 0.5, 0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]

# Adjust data
V_data = [0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]

time = range(0, len(V_data))

plc_n1 = np.array([time, V_data])
print (V_data)
plt.plot(time, V_data, 'o', color='black')


pV = 12.6
beta = 5e-7
deltaV = 8
deltaI = 2
gT = 0.8
V0 = 1

param_names = ['pV', 'beta', 'deltaV', 'deltaI', 'gT', 'V0']

MCMC_param = [31.65752046, 0.21270496, 29.54909252, 18.4281431, 1.68226671, 8.23216014]
MCMC_param = [26, 0.01, 20, 13, 0.8, 11]

MCMC_param = [pV, beta, deltaV, deltaI, gT, V0]
MCMC_param = [49.497919837661165, 9.760356836920902, 129.40185329215433, 22.814432227722477, 19.45175546613011, 79.31321451172576]
test_fit(TIV_funcs.TIV_model, plc_n1, MCMC_param, len(V_data))


