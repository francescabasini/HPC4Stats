from parallel.backends import BackendMPI as Backend
backend = Backend()

import numpy as np

X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 0.2, 100)

X_bds = backend.broadcast(X)
Y_bds = backend.broadcast(Y)


def function2parallelise(location, X, Y):
    Z = X+Y
    return Z


def myfunc(ind):
    return function2parallelise(location=ind, X=X, Y=Y)

seed_arr = [ind for ind in range(X.shape[0])]
seed_pds = backend.parallelize(seed_arr)
accepted_parameters_pds = backend.map(myfunc, seed_pds)
accepted_parameters = backend.collect(accepted_parameters_pds)

#mpirun -np 4 python3 pmcabc_gaussian.py