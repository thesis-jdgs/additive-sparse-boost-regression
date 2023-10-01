"""Python wrapper for C potts module"""
import ctypes
from os.path import abspath
from os.path import dirname
from platform import system as operating_system
from typing import Optional
from typing import Tuple

import numpy as np


os_name = operating_system()
path = dirname(abspath(__file__))
clib_path = f"{path}\\l2_potts.dll" if os_name == "Windows" else f"{path}/l2_potts.so"
loader = ctypes.CDLL(clib_path)
loader.l2_potts.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # input
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # weights
    ctypes.c_int,  # data size
    ctypes.c_double,  # l0 fused regularization
    ctypes.c_double,  # l2 regularization
    ctypes.c_int,  # excluded interval size
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),  # split indexes
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # leaves
)
loader.l2_potts.restype = ctypes.c_uint


def l2_potts(
    x: np.ndarray[float],
    y: np.ndarray[float],
    weights: Optional[np.ndarray] = None,
    l0_fused_regularization: float = 1.0,
    l2_regularization: float = 0.0,
    excluded_interval_size: int = 0,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """Denoise a 1D signal using the Potts model.

    Parameters
    ----------
    x : np.ndarray[float]
        Input signal times.
    y : np.ndarray[float]
        Input signal values.
    weights : np.ndarray, optional
        Weights of the input signal. If None, all weights are set to 1.
    l0_fused_regularization : float, default=1.0
        Regularization parameter for the l0 fused norm.
    l2_regularization : float, default=0.0
        Regularization parameter for the l2 norm.
    excluded_interval_size : int, default=0
        Minimum size of intervals that should not be denoised.

    Returns
    -------
    leaves : np.ndarray[float]
        Leaves of the tree.
    split_values : np.ndarray[float]
        Splits of the tree.

    """
    np_input = np.ascontiguousarray(y, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(np_input)
    split_indexes = np.empty_like(np_input, dtype=np.int32)
    leaves = np.empty_like(np_input, dtype=np.float64)
    counter = loader.l2_potts(
        np_input,
        weights,
        y.size,
        l0_fused_regularization,
        l2_regularization,
        excluded_interval_size,
        split_indexes,
        leaves,
    )
    leaves = leaves[counter::-1]
    split_indexes = (
        split_indexes[counter - 1 :: -1] if counter > 0 else np.empty(0, dtype=np.int32)
    )
    split_values = x[split_indexes]
    return leaves, split_values
