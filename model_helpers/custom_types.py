"""Implement classes for typing estimators."""

# Standard library imports
from typing import Callable

# Third party imports
import numpy as np
import pandas as pd


Data = np.ndarray | pd.DataFrame
Target = np.ndarray | pd.Series
OneVectorFunction = Callable[[np.ndarray], np.ndarray]
TwoVectorFunction = Callable[[np.ndarray, np.ndarray | float], np.ndarray]
