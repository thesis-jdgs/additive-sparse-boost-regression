"""Python wrapper for C tv1d module"""
import ctypes
from os.path import abspath
from os.path import dirname
from platform import system as operating_system
from typing import Optional

import numpy as np


os_name = operating_system()
path = dirname(abspath(__file__))
clib_path = f"{path}\\condat_tv.dll" if os_name == "Windows" else f"{path}/condat_tv.so"
loader = ctypes.CDLL(clib_path)
loader.tvod_denoise.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # input
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # output
    ctypes.c_uint,  # width
    ctypes.c_double,  # lambda
)
loader.tvod_denoise.restype = None


def tv1d_denoise(
    arr: np.ndarray, lambda_: float = 1.0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Denoise a 1D signal using total variation regularization.

    Parameters
    ----------
    arr : np.ndarray
        Input signal.
    lambda_ : float
        Regularization parameter.
    out : Optional[np.ndarray], optional
        Output array, by default None

    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    if out is None:
        out = np.empty_like(arr)
    loader.tvod_denoise(arr, out, arr.size, lambda_)
    return out
