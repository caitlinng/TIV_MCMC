'''
Fits TIV model to given data
'''

# Module import
import TIV_funcs
import test_fit
import pandas as pd
import numpy as np
import scipy as sci
from scipy import stats as st
import matplotlib.pyplot as plt

# Import viral load data (will try fitting first with plc_n1)
days = range(1, 12)

OST_n1 = np.array([days, [0.5,0.5,0.5,2.5,3.833333,3.5,3.166667,0.833333,0.5,0.5,0.5]])
OST_n2 = np.array([days, [0.5,0.5,0.5,0.5,3.5,2.5,3.5,3.166667,2.5,0.5,0.5]])
OST_n3 = np.array([days, [0.5,0.5,0.5,0.833333,1.166667,2.833333,4.166667,0.833333,1.166667,0.5,0.5]])
OST_n4 = np.array([days, [0.5,0.5,0.5,0.5,0.833333,2.833333,2.833333,3.166667,1.166667,0.833333,0.5]])

plc_n1 = np.array([days, [0.5,0.5,0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]])
plc_n2 = np.array([days, [0.5,0.5,0.5,0.5,4.166667,3.5,3.5,2.833333,0.5,0.5,0.5]])
plc_n3 = np.array([days, [0.5,0.5,0.5,3.833333,4.833333,3.5,3.5,0.5,0.5,0.5,0.5]])
plc_n4 = np.array([days, [0.5,0.5,0.5,1.166667,3.833333,3.5,2.833333,3.166667,0.5,0.5,0.5]])

OST_list = np.array([OST_n1, OST_n2, OST_n3, OST_n4])
plc_list = np.array([plc_n1, plc_n2, plc_n3, plc_n4])

OST_all = np.array([[OST_n1, OST_n2], [OST_n3, OST_n4]])
plc_all = np.array([[plc_n1, plc_n2], [plc_n3, plc_n4]])

# Edit data so that day 1 is when there is detectable increase in viral load:
time = range(0, 9)
V_data = [0.5,4.5,5.833333,4.5,3.5,3.5,2.833333,0.5,0.5]
plc_n1 = np.array([time, V_data])

'''
# Produce synthetic data

# Parameters
data_pV = 210
data_beta = 5e-7
data_param = [data_pV, data_beta]

# Solve TIV model and get viral load measurements as V_data
data = TIV_funcs.TIV_model(data_param)
V_data = data.y[2]

# Plot TIV model
fig1, ax1 = plt.subplots()
fig1.suptitle("Basic TIV Model")

ax1.plot(data.t, data.y[0], data.t, data.y[1])
ax1.set_title("target and infected cells")
ax1.set_xlabel("Days")
ax1.set_ylabel("Cells")
ax1.legend(('T', 'I'))

fig1, ax2 = plt.subplots()
ax2.plot(data.t, data.y[2])
ax2.set_title('Viral Load')
ax2.set_xlabel("Days")
ax2.set_ylabel("Viral load (TCID_{50})")
ax2.set_yscale('log')

plt.show()
'''


'''
MCMC Set-up
'''
# Proposal widths
w = [0.05, 0.05]

# Number of iterates
n_iterates = 1000

# Prior functions (what is our prior belief about beta, gamma)

def prior_param_belief(min, max):
    return st.uniform(loc=min, scale=max-min)

def prior_param_belief_normal(mean, var):
    return st.norm(loc=mean, scale=var)

# Prior belief is that beta and gamma are within a reasonable given range
pV_prior_fun = prior_param_belief(1e-6, 1e+6) #(0, 24) #(-6, 6) #(150, 250)
beta_prior_fun = prior_param_belief(1e-12, 1e-4) #(0, 0.5) # (e-12, e-4)  #(0, 0.5)

prior_funcs = [pV_prior_fun, beta_prior_fun]

# Starting guess [pV, beta]
#init_param = data_param
#init_param = [np.random.uniform(150, 250), np.random.uniform(0, 0.5)]
init_param = [12.25, 5e-7]  #[225, 0]

# Calculate the log likelihood of the initial guess
#init_ll = TIV_funcs.TIV_ll(V_data, init_param)
init_ll = TIV_funcs.TLIV_ll(V_data, init_param) # TODO: Testing with TLIV fit

# And log likelihood of all subsequent guesses
def run_chain(V_data, n_iterates, w, init_param, init_ll, prior_funcs):
    param = init_param.copy()
    ll = init_ll.copy()

    # Establish data storage space for chain
    # Where first column is ll and 1: are the n-parameters [ll, beta, gamma]
    chain = np.zeros((n_iterates, len(param) + 1))
    chain[0, 0] = ll
    chain[0, 1:] = param

    # Run MCMC
    for i in range(n_iterates):

        # Print status every 10 iterations
        if i % 10 == 0:
            print('Iteration ' + str(i) + ' of ' + str(n_iterates))

        # Gibbs loop over number of parameters (j = 0 is beta, j = 1 is gamma)
        for j in range(len(param)):

            # Propose a parameter value within prev. set widths
            prop_param = param.copy()

            # Take a random step size from a uniform distribution (that has width of w)
            prop_param[j] = prop_param[j] - (sci.stats.uniform.rvs(loc=-w[j] / 2, scale=w[j], size=1))
            prop_param[j] = np.ndarray.item(prop_param[j]) # Converts paramater value from single element array into a scalar

            # Deal with invalid proposals by leaving ll, param unchanged
            if prop_param[j] <= 0: # Invalid, so try next parameter proposal
                prop_ll = -1 * np.inf

            else:
                # Calculate LL of proposal
                #prop_ll = TIV_funcs.TIV_ll(V_data, prop_param)
                prop_ll = TIV_funcs.TLIV_ll(V_data, prop_param) # TODO: Testing with TLIV fitting

            # Decide on accept/reject
            prior_fun = prior_funcs[j]  # Grab correct prior function

            # Likelihood ratio st.norm.rvs(1, 1)
            r = np.exp(prop_ll - ll) * prior_fun.pdf(prop_param[j]) / prior_fun.pdf(param[j])
            print('prop_ll = ' + str(prop_ll))
            print ('ll = ' + str(ll))
            print ('prop_ll - ll = ' + str(prop_ll - ll))
            print('np.exp(prop_ll - ll) = ' + str(np.exp(prop_ll - ll)))
            print('r = ' + str(r))
            # Is likelihood ratio less than or equal to one
            alpha = min(1, r)

            # Random number between 0 to 1
            # So will have weighted chance of possibly accepting depending on how likely the new parameter is
            test = np.random.uniform(0, 1)
            # Maybe accept
            if (test < alpha):
                ll = prop_ll.copy()
                param = prop_param.copy()
            # "Else" reject, though nothing to write

            # Store iterate
            chain[i, 0] = ll
            chain[i, 1:] = param
    return chain

chain = run_chain(V_data, n_iterates, w, init_param, init_ll, prior_funcs)

print(chain)

# Show viral data curves (actual data vs model)
V_data = [0.5, 4.5, 5.833333, 4.5, 3.5, 3.5, 2.833333, 0.5, 0.5]
MCMC_param = [chain[1, -1], chain[2, -1]]

test_fit.test_fit(plc_n1, MCMC_param)

# Show MCMC graphs
chain = pd.DataFrame(chain, columns=['ll', 'pV', 'beta']) # Change np to panda array

n = np.arange(int(n_iterates)) # Creates array of iterates

chain.plot(kind= 'line', y = 'll')
plt.ylabel('Log Likelihood')
plt.xlabel('Iterate')

chain.plot(kind = 'line', y = 'beta')
#plt.plot(y = pbeta, color = 'r') # Plot true value as a single line
plt.ylabel('Estimate for beta')
plt.xlabel('Iterate')

chain.plot(kind = 'line', y = 'pV', color = 'b')
#plt.plot(y = pgamma, color = 'r') # Plot true value as a single line
plt.ylabel('Estimate for pV')
plt.xlabel('Iterate')

chain[['beta']].plot(kind = 'hist')

chain[['pV']].plot(kind = 'hist')


plt.show()

