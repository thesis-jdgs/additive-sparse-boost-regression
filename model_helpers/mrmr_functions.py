"""Implement functions for the MRMR algorithm."""
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression


# Relevancy
def f_regression_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the f_regression score for each feature.
    It assumes that the data has been centered.

    Parameters
    ----------
    X : np.ndarray
        The data matrix. It must have 0 mean on each column.
    y : np.ndarray
        The target vector.

    Returns
    -------
    np.ndarray
        The f_regression score for each feature.
    """
    return f_regression(X, y, center=False)[0]


# Redundancy
def absolute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Return the absolute correlation matrix.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.

    Returns
    -------
    np.ndarray
        The absolute value of each entry on the correlation matrix,
        except the diagonal, which is zero.
    """
    C = np.abs(np.corrcoef(X, rowvar=False))
    C[np.diag_indices_from(C)] = 0.0
    return C


# mRMR scheme
def safe_divide(divisor: np.ndarray, dividend: np.ndarray | float) -> np.ndarray:
    """Divide two arrays, returning zero when the divisor is zero.

    Parameters
    ----------
    divisor : np.ndarray
        The divisor array.
    dividend : np.ndarray
        The dividend array. It must have the same shape as the divisor.
        It can have zero values.

    Returns
    -------
    np.ndarray
        The result of the division. Entries where the dividend is zero are zero.
    """
    return np.divide(divisor, dividend, out=np.zeros_like(divisor), where=dividend != 0)


def mutual_information_matrix(
    X: np.ndarray,
    discrete_features: list[int] = None,
) -> np.ndarray:
    """Return a matrix with the mutual information between each pair of features.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.

    discrete_features: list[int], optional
        The indices of the discrete features.
        If None, all features are assumed to be continuous.

    Returns
    -------
    np.ndarray
        The mutual information matrix.
    """
    if discrete_features is None:
        discrete_features = np.zeros(X.shape[1], dtype=bool)
    C = np.empty((X.shape[1], X.shape[1]))
    for i, is_discrete in enumerate(discrete_features):
        if is_discrete:
            C[i, :] = mutual_info_classif(
                X, X[:, i], discrete_features=discrete_features
            )
        else:
            C[i, :] = mutual_info_regression(
                X, X[:, i], discrete_features=discrete_features
            )
    C = np.exp(C)
    C[np.diag_indices_from(C)] = 0.0
    return C
