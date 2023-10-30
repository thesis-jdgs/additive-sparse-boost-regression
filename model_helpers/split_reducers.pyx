import numpy as np
cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[::1] mean_reducer(double[::1] y, list[double[::1]] splits):
    cdef Py_ssize_t n = len(splits)
    cdef double[::1] y_means = np.empty(n, dtype=np.float64)

    cdef Py_ssize_t i
    cdef double[::1] split
    for i in range(n):
        split = splits[i]
        y_means[i] = np.mean(y[split])
    return y_means


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[::1] sum_reducer(double[::1] y, list[double[::1]] splits):
    cdef Py_ssize_t n = len(splits)
    cdef double[::1] y_sum = np.empty(n, dtype=np.float64)

    cdef Py_ssize_t i
    cdef double[::1] split
    for i in range(n):
        split = splits[i]
        y_sum[i] = np.sum(y[split])
    return y_sum
