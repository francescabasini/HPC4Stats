from parallel.backends.base import *


def BackendMPI(*args,**kwargs):
    from parallel.backends.mpi import BackendMPI
    return BackendMPI(*args,**kwargs)

def BackendMPITestHelper(*args,**kwargs):
    from parallel.backends.mpi import BackendMPITestHelper
    return BackendMPITestHelper(*args,**kwargs)

def BackendSpark(*args,**kwargs):
    from  parallel.backends.spark import BackendSpark
    return BackendSpark(*args,**kwargs)
