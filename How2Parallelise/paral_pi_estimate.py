## Code for parallelisation of Example 1 - Estimating Pi

from parallel.backends import BackendMPI as Backend
import numpy as np
import datetime

backend = Backend()

S = 500000

# one sample estimate
def sample_est_pi(location, N, rng = np.random.RandomState()):
    x = np.random.uniform(0,1, N)
    f_x = np.sqrt(1-x**2)
    integral = np.mean(f_x)
    return integral*4

# function to parallelise
def multi_sample_pi(ind):
    return sample_est_pi(location=ind, N = 10000)

time_start = datetime.datetime.now()

seed_arr = [ind for ind in range(S)]
seed_pds = backend.parallelize(seed_arr)
accepted_parameters_pds = backend.map(multi_sample_pi, seed_pds)
accepted_parameters = backend.collect(accepted_parameters_pds)

time_end = datetime.datetime.now()
time_delta = time_end - time_start

# saving data
np.savetxt('estimates_pi.csv', np.array(accepted_parameters), delimiter=",")
print("Done in {} seconds and {} microseconds".format(time_delta.seconds, time_delta.microseconds/1e6))

#mpirun -np 4 python3 paral_pi_estimate.py
