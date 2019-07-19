import matplotlib.pyplot as plt
import numpy as np
import TIV_funcs

'''
Plots actual data against model obtained with MCMC-derived parameters
'''

def test_fit(real_data, MCMC_param):  # real_data = [time, viral load], MCMC_param = [pV, data]
    # Generate model data with MCMC derived parameters
    sol = TIV_funcs.TLIV_model(MCMC_param)

    # Scatter plot with actual data
    x = real_data[0]
    y = real_data[1]

    plt.figure()
    plt.plot(x, y, 'o', color='black')

    # Plot model data as a line in the same plot as the actual data
    plt.plot(sol.t, sol.y[2], '-', color='blue')

    return plt.show()

# Commenting out, because it keeps producing the graph when importing the file

time = range(0, 11)
V_data = [0.5, 0.5, 0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]


'''
# Trying to un-log the data points in case that was why the actual data is so low compared to the model...
# But nope, doesn't work
for i in range(len(V_data)):
    print('i = ' + str(i))
    print('V_data[i] = ' + str(V_data[i]))
    V_data[i] = 10**V_data[i]
    print('V_data[i] (supposedly exp) = ' + str(V_data[i]))
'''

plc_n1 = np.array([time, V_data])
print (V_data)
plt.plot(time, V_data, 'o', color='black')
plc_n1 = np.array([time, V_data])

MCMC_param = [11.2, 0.00007]

test_fit(plc_n1, MCMC_param)
