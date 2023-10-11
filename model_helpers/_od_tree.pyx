import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np


cdef struct SplitInfo:
    int index
    bint should_split


@cython.boundscheck(False)
@cython.wraparound(False)
cdef SplitInfo best_split(
        double[::1] y,
        float l2_regularization,
):
    cdef double[::1] left_gradient = np.cumsum(y)
    cdef Py_ssize_t n = y.shape[0]
    cdef double total_gradient = left_gradient[-1]

    cdef double left_grad, right_grad,
    cdef Py_ssize_t i, left_hess, right_hess
    cdef double[::1] gain_arr = np.empty_like(y, order="C")
    for i in prange(n, nogil=True):
        left_grad = left_gradient[i]
        left_hess = i + 1

        right_grad = total_gradient - left_grad
        right_hess = n - i

        gain_arr[i] = (left_grad * left_grad) / (left_hess + l2_regularization) + (right_grad * right_grad) / (right_hess + l2_regularization)

    cdef int best_index = np.argmax(gain_arr)
    cdef double best_gain = gain_arr[best_index]
    cdef double loss = total_gradient * total_gradient / (n + 1 + l2_regularization)
    return SplitInfo(best_index, best_gain <= loss)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef build_list_tree(
    float[::1] X,
    double[::1] y,
    double[::1] sample_weight,
    float l2_regularization,
    int min_samples_leaf,
    int min_samples_split,
    int max_depth
):
    cdef int stack_size = y.size // min_samples_split + 1
    cdef int[:, ::1] stack = np.empty((stack_size, 3), dtype=int)
    cdef int stack_pos = 1
    cdef float[::1] split_values = np.empty_like(
        y, dtype=np.float32, order="C", subok=False
    )

    cdef double[::1] leaf_values = np.empty_like(
        y, dtype=np.float64, order="C", subok=False
    )
    cdef float[::1] split_stack = np.empty_like(
        y, dtype=np.float32, order="C", subok=False
    )
    cdef int sv_pos = 0
    cdef int lv_pos = 0
    cdef int ss_pos = 0
    stack[0, 0] = 0
    stack[0, 1] = y.size
    stack[0, 2] = 0

    cdef int start, end, depth, n_samples, index, ip
    cdef double numerator_lv, denominator_lv
    cdef bint admissible
    cdef float split

    cdef Py_ssize_t i
    cdef double[::1] y_weighted = np.empty(y.size, dtype=np.float64)
    cdef double[::1] cumulative_gradient = np.empty(y.size, dtype=np.float64)
    cdef double[::1] cumulative_hessian = np.empty(y.size, dtype=np.float64)

    cdef double cum_sum_gradient = 0.0
    cdef double cum_sum_hessian = 0.0

    for i in range(stack[0, 1]):
        y_weighted[i] = y[i] * sample_weight[i]
        cum_sum_gradient += y_weighted[i]
        cum_sum_hessian += sample_weight[i]
        cumulative_gradient[i] = cum_sum_gradient
        cumulative_hessian[i] = cum_sum_hessian
    cdef double total_gradient, total_hessian, left_grad, right_grad, left_hess, right_hess
    cdef double gradient_shift, hessian_shift
    cdef double best_gain, gain

    while stack_pos > 0:
        start = stack[stack_pos - 1, 0]
        end = stack[stack_pos - 1, 1]
        depth = stack[stack_pos - 1, 2]
        stack_pos -= 1
        n_samples = end - start
        if depth >= max_depth or n_samples < min_samples_split or n_samples <= min_samples_leaf:
            if start > 0:
                numerator = cumulative_gradient[end - 1] - cumulative_gradient[start - 1]
                denominator = cumulative_hessian[end - 1] - cumulative_hessian[start - 1]
            else:
                numerator = cumulative_gradient[end - 1]
                denominator = cumulative_hessian[end - 1]
            leaf_values[lv_pos] = numerator / (denominator + l2_regularization)
            lv_pos += 1
            if ss_pos > 0:
                split_values[sv_pos] = split_stack[ss_pos - 1]
                sv_pos += 1
                ss_pos -= 1
        else:
            gradient_shift = y_weighted[start] - cumulative_gradient[start]
            hessian_shift = sample_weight[start] - cumulative_hessian[start]
            total_gradient = cumulative_gradient[end - 1] + gradient_shift
            total_hessian = cumulative_hessian[end - 1] + hessian_shift
            best_gain = -np.inf
            for i in range(n_samples):
                left_grad = cumulative_gradient[i + start] + gradient_shift
                left_hess = cumulative_hessian[i + start] + hessian_shift

                right_grad = total_gradient - left_grad
                right_hess = total_hessian - left_hess
                gain = (left_grad * left_grad) / (left_hess + l2_regularization) + (right_grad * right_grad) / (
                            right_hess + l2_regularization)
                if gain > best_gain:
                    best_gain = gain
                    index = i
            if best_gain <= total_gradient * total_gradient / (total_hessian + l2_regularization):
                if start > 0:
                    numerator = cumulative_gradient[end - 1] - cumulative_gradient[start - 1]
                    denominator = cumulative_hessian[end - 1] - cumulative_hessian[start - 1]
                else:
                    numerator = cumulative_gradient[end - 1]
                    denominator = cumulative_hessian[end - 1]
                leaf_values[lv_pos] = numerator / (denominator + l2_regularization)
                lv_pos += 1
                if ss_pos > 0:
                    split_values[sv_pos] = split_stack[ss_pos - 1]
                    sv_pos += 1
                    ss_pos -= 1
            else:
                split = X[start + index]
                split_stack[ss_pos] = split
                ss_pos += 1
                ip = start + index + 1
                stack[stack_pos, 0] = ip
                stack[stack_pos, 1] = end
                stack[stack_pos, 2] = depth + 1
                stack_pos += 1
                stack[stack_pos, 0] = start
                stack[stack_pos, 1] = ip
                stack[stack_pos, 2] = depth + 1
                stack_pos += 1

    return leaf_values[:lv_pos], split_values[:sv_pos]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] eval_piecewise(
    np.ndarray[float, ndim=1] x,
    np.ndarray[float, ndim=1] split_values,
    np.ndarray[double, ndim=1] leaf_values,
):
    return leaf_values[np.searchsorted(split_values, x)]
