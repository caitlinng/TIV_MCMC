# Module import
import TIV_funcs
import test_fit
import pandas as pd
import numpy as np
import scipy as sci
from scipy import stats as st
import matplotlib.pyplot as plt
import datetime
import math


'''
MCMC Set-up
'''
# Set random seed for reproducibility
#np.random.seed(42)

# Save chain ('y') or not (anything else)
save = 'y'

# Number of iterates
n_iterates = 1000

# Prior functions (what is our prior belief about beta, gamma)

def prior_param_belief(min, max):
    return st.uniform(loc=min, scale=max-min)

def prior_param_belief_normal(mean, var):
    return st.norm(loc=mean, scale=var)

# TODO: Make into a dictionary {name, w, prior, guess}
# Proposal widths (1 for each parameter)
w = [0.00075, 7.5e-6, 0.00075, 0.00075, 0.00075]  #FIXME: Need to find a good one for beta particularly
#w = [4.7514528036117556e-05, 2.6726922020316124e-07, 4.7514528036117556e-05, 4.7514528036117556e-05, 4.7514528036117556e-05]

param_names = ['pV', 'beta', 'deltaV', 'deltaI', 'V0']

# Parameters
pV = 4.2e-2
beta = 6.3e-6
deltaV = 3.1
deltaI = 2.8

# Initial conditions
T0 = 4e+8  # Fixing this parameter
I0 = 0
V0 = 0.91  # param[5]

init_param = [pV, beta, deltaV, deltaI, V0]

# Input own parameters (e.g. from previously run chain)
init_param = [0.1, 1e-6, 5, 5, 1.2]

# Prior belief is that beta and gamma are within a reasonable given range
pV_prior_fun = prior_param_belief(1e-4, 1) #(0, 24) #(-6, 6) #(150, 250)
beta_prior_fun = prior_param_belief(1e-7, 1e-5) #(0, 0.5) # (e-12, e-4)  #(0, 0.5)
deltaV_prior_fun = prior_param_belief(0, 6)
deltaI_prior_fun = prior_param_belief(0, 6)
V0_prior_fun = prior_param_belief(0, 3)

prior_funcs = [pV_prior_fun, beta_prior_fun, deltaV_prior_fun, deltaI_prior_fun, V0_prior_fun]


print('prior_funcs = ' + str(prior_funcs))

# Choose to fit TIV or TLIV model
model_ll = TIV_funcs.TIV_ll_Baccam
model = TIV_funcs.TIV_model_Baccam

# Baccam data
V_data = [3.5, 5.5, 6.5, 5.5, 3.5, 4.0, 0.5, 0.5]  # Baccam patient 4 (log10)

V_data = [3162.3, 316227.766, 3162277.66, 316227.766, 3162.3, 10000, 3.16227766, 3.16227766]
max_time = len(V_data)
time = range(0, max_time)
Baccam_n4 = np.array([time, V_data])

def run_chain(model_ll, init_param, V_data, max_time, n_iterates, w, prior_funcs, calibrate_truth):  # NOTE: Need to input TIV_funcs.model_ll
    calibrate = calibrate_truth

    # Calculate the log likelihood of the initial guess
    init_ll = model_ll(V_data, init_param, max_time)
    # init_ll = TIV_funcs.TLIV_ll(V_data, init_param) # TODO: Testing with TLIV fit

    # And log likelihood of all subsequent guesses
    param = init_param.copy()
    ll = init_ll.copy()

    # Establish data storage space for chain
    # Where first column is ll and 1: are the n-parameters [ll, beta, gamma]
    chain = np.zeros((n_iterates, len(param) + 1))
    chain[0, 0] = ll
    chain[0, 1:] = param

    accepts = np.zeros(len(init_param))

    # Run MCMC
    for i in range(n_iterates):

        # Print status every 100 iterations
        if i % 100 == 0:
            print('Iteration ' + str(i) + ' of ' + str(n_iterates))
        #print('Iteration ' + str(i) + ' of ' + str(n_iterates))

        # Gibbs loop over number of parameters (j = 0 is beta, j = 1 is gamma)
        for j in range(len(param)):
            #print('param j = ' + str(j))

            # For every 100 iterations, calibrate w
            if calibrate is True:
                if i % 100 == 0 and i > 0:
                    w[j] = TIV_funcs.tune_w(accepts=accepts[j], current_iter=i, w=w[j], param_name=param_names[j])
                    print('w = ' + str(w))
                    accepts[j] = 0  # Reset acceptance rate

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
                prop_ll = model_ll(V_data, prop_param, max_time) # TODO: Testing with TLIV fitting

            # Decide on accept/reject
            prior_fun = prior_funcs[j]  # Grab correct prior function

            # Likelihood ratio st.norm.rvs(1, 1)
            r = np.exp(prop_ll - ll) * prior_fun.pdf(prop_param[j]) / prior_fun.pdf(param[j])

            if math.isnan(r):  # FIXME: Is this correct? Parameter not falling into prior function at all
                alpha = 0
                print('r was a NaN')
                print('So param ' + str(prop_param) + ' were rejected')

            else:
                print('r = ' + str(r))
                # Is likelihood ratio less than or equal to one
                alpha = min(1, r)

            # Random number between 0 to 1
            # So will have weighted chance of possibly accepting depending on how likely the new parameter is
            test = np.random.uniform(0, 1)
            # Maybe accept
            if (test < alpha):
                try:
                    ll = prop_ll.copy()
                    param = prop_param.copy()
                    accepts[j] += 1

                    print('Parameters proposed = ' + str(prop_param))
                    print('Accept new parameters ' + str(param))

                except AttributeError:
                    print('In AttributeError, ll was = ' + str(prop_ll))
                    print('So param ' + str(prop_param) + ' were rejected')


            # "Else" reject, though nothing to write
            else:
                if i % 10 == 0:
                    print('Reject parameters ' + str(prop_param))

            # Store iterate
            chain[i, 0] = ll
            chain[i, 1:] = param

            # Update chain.txt file with new iterate as chain is generated (discards first half though)
            '''
            if save == 'y' and i > int(n_iterates/2):
                f = open('chain.txt', 'a')
                f.write('\n' + str(chain[i]))
                f.close()
            '''

    if calibrate is True:
        print('w = ' + str(w))
        for i in range(len(accepts)):
            acceptance_rate = accepts[i]/ n_iterates
            print('Acceptance rate for ' + param_names[i] is str(acceptance_rate))

    return chain

# Start simulation and record time simulation was started
now = datetime.datetime.now()
chain = run_chain(model_ll, init_param, V_data, max_time, n_iterates, w, prior_funcs, calibrate_truth=True)

print(chain)

# Store only last half of chain (discard burn-in, i.e. until n_iterations/2)
chain_half = chain[int(n_iterates/2):, :]

MCMC_param = chain[-1, 1:]  # TODO: Look at last estimated ll

# Save chain to file chains.txt (appends the chain as opposed to creating a new file)
if save == 'y':
    f = open('chain_info.txt')
    f.write('\n')
    f.write('\n' + now.strftime("%Y-%m-%d %H:%M"))
    f.write('\n' + 'init_param = ' + str(init_param))
    f.write('\n' + 'w = ' + str(w))
    f.write('\n' + 'n_iterates = ' + str(n_iterates))
    f.write('\n' + str(chain_half))
    f.write('\n' + 'Last tested parameters were: ' + str(MCMC_param))
    f.write('\n' + 'best_ll row = ' + str(chain.iloc[best_ll_index, :]))
    f.close()


# Modifying data to make easier to plot
chain = pd.DataFrame(chain, columns=['ll', 'pV', 'beta', 'deltaV', 'deltaI', 'V0'])  # Change np to panda array

#chain_half = pd.DataFrame(chain_half, columns=['ll', 'pV', 'beta', 'deltaV', 'deltaI', 'V0'])  # Change np to panda array

best_ll_index = chain[['ll']].idxmax()
print('init_param = ' + str(init_param))
print('best_ll row = ' + str(chain.iloc[best_ll_index, :]))

best_MCMC_param = chain.iloc[best_ll_index, 1:]
print('MCMC_param = ' + str(MCMC_param))


# Show viral data curves (actual data vs model)
test_fit.test_fit(TIV_funcs.TIV_model_Baccam, Baccam_n4, MCMC_param, max_time)


# Show MCMC graphs

# Creating a list of dataframe columns
columns = list(chain)

# Plotting ll
chain.plot(kind= 'line', y = 'll')
plt.ylabel('Log Likelihood')
plt.xlabel('Iterate')
plt.savefig('ll_chain.png')  # Save as .png

# Plotting each parameter
for i in columns[1:]:
    param = i

    # Plotting chain
    chain.plot(kind= 'line', y = param)
    plt.ylabel('Estimate for ' + str(param))
    plt.xlabel('Iterate')
    plt.savefig(str(param) + '_chain' + '.png')  # Save as .png

    # Plotting histogram
    chain[[param]].plot(kind = 'hist')
    plt.savefig(str(param) + '_freq' + '.png')  # Save as .png

plt.show()
