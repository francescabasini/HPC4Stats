## Code for parallelisation of Example 1 - Estimating Pi

from parallel.backends import BackendMPI as Backend
import numpy as np
import datetime

backend = Backend()

# Parameters
mu = 9.2
sigma = 3.9

N = 100000
rng = np.random.RandomState(seed  = 19)

data_Xi = rng.normal(mu, sigma, N) 

### Shared-data
X_bds = backend.broadcast(data_Xi)

# Hyper-parameters
mu_0 = 9
sigma_0 = 0.7

# Posterior Predictive for one sample
def posterior_pred_h(location, data_X, sigma, mu_0, sigma_0, N, rng = np.random.RandomState()):  
    n = len(data_X)
    sigma_n_2 = 1/(n/sigma**2 + 1/sigma_0**2)
    mu_n = sigma_n_2*(mu_0/sigma_0**2 + np.sum(data_X)/sigma**2)
    # sample mu_n from
    mu_n_sampled = rng.normal(mu_n, np.sqrt(sigma_n_2), 1)[0]
    x_tilde = rng.normal(mu_n_sampled, np.sqrt(sigma_n_2+ sigma**2), N)
    # compute function on x_tildes
    h_x = np.mean(x_tilde)
    return mu_n_sampled, h_x


# function to parallelise
def multi_sample_posterior_pred_h(ind, data_X = data_Xi):
    return posterior_pred_h(location=ind, data_X = data_X ,
                            sigma = sigma, mu_0 = mu_0,  sigma_0 = sigma_0, N = 100)

# how many sample?
S = 50000
time_start = datetime.datetime.now()

seed_arr = [ind for ind in range(S)]
seed_pds = backend.parallelize(seed_arr)
accepted_parameters_pds = backend.map(multi_sample_posterior_pred_h, seed_pds)
accepted_parameters = backend.collect(accepted_parameters_pds)

time_end = datetime.datetime.now()
time_delta = time_end - time_start


## first unpack the list
all_parameters = np.array(accepted_parameters).squeeze()
mu_n_estimates = all_parameters[:,0]
h_x_estimates = all_parameters[:,1]
# saving data
np.savetxt('estimates_mu_ex2.csv', mu_n_estimates, delimiter=",")
np.savetxt('estimates_hx_ex2.csv', h_x_estimates, delimiter=",")
print("Done in {} seconds and {} microseconds".format(time_delta.seconds, time_delta.microseconds/1e6))


#mpirun -np 4 python3 pmcabc_gaussian.py