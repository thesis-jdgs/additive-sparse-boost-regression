"""Implement classes for typing estimators."""
from typing import Callable

import numpy as np
import pandas as pd


Data = np.ndarray | pd.DataFrame
Target = np.ndarray | pd.Series
OneVectorFunction = Callable[[np.ndarray], np.ndarray]
TwoVectorFunction = Callable[[np.ndarray, np.ndarray | float], np.ndarray]

try:
    from typing import Self as _Self

    Self = _Self
except ImportError:
    Self = "Self"
