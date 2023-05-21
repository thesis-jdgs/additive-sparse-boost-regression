"""Implement a 1D regression tree."""
from __future__ import annotations

import contextlib

import attrs
import numba as nb
import numpy as np

# Standard library imports
# Third party imports

# Python 3.11 feature
with contextlib.suppress(ImportError):
    from typing import Self


@nb.njit(
    nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:]),
    fastmath=True,
    cache=True,
)
def eval_piecewise(
    x: np.ndarray, split_values: np.ndarray, leaf_values: np.ndarray
) -> np.ndarray:
    """Evaluate a piecewise constant function at a given point."""
    return leaf_values[np.searchsorted(split_values, x)]


@attrs.define(slots=True)
class ListTree:
    """Tree represented as a list, for faster inference and merging.

    Parameters
    ----------
    leaf_values : np.ndarray of shape (n_splits + 1,)
        The values of the tree's leaves.
    split_values : np.ndarray of shape (n_splits,)
        The values at which the tree splits the data.
    """

    leaf_values: np.ndarray = attrs.field(converter=np.array)
    split_values: np.ndarray = attrs.field(converter=np.array)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the piecewise constant function at a given point.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            The point at which to evaluate the tree. It can be a single value,
            or an array of values.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The value of the tree at the given points.
        """
        return eval_piecewise(x, self.split_values, self.leaf_values)


@attrs.define(slots=True)
class ListTreeRegressor:
    """A 1D Decision Tree Regressor, represented as a list,
    for faster inference and merging.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the list.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    l2_regularization : float
        The L2 regularization parameter.
    bias : float
        A correction term to add to the tree's predictions.
    learning_rate : float
        A correction term to multiply the tree's predictions by.
    is_selected : bool
        Whether the tree is the null tree or not.
    feature_name : str
        The name of the feature the tree splits on.
    output_name : str
        The name of the output the tree predicts.

    Attributes
    ----------
    list_tree_ : ListTree
        The tree represented as a list. It is learnt after fitting.
    """

    # Hyper-parameters
    max_depth: int = attrs.field(default=3)
    min_samples_split: int = attrs.field(default=2)
    min_samples_leaf: int = attrs.field(default=1)
    l2_regularization: float = attrs.field(default=0.01)
    # Correction parameters
    bias: float = attrs.field(default=0.0)
    learning_rate: float = attrs.field(default=1.0)
    is_selected: bool = attrs.field(default=True)
    # Model information
    feature_name: str | None = attrs.field(default=None)
    output_name: str | None = attrs.field(default=None)
    # Learnt after fitting
    list_tree_: ListTree = attrs.field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model to the given data.

        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        y : np.ndarray
            The target values.

        Returns
        -------
        Self
            The fitted model.
        """
        self.list_tree_ = ListTree(
            *build_list_tree(
                X,
                y,
                self.l2_regularization,
                self.min_samples_leaf,
                self.min_samples_split,
                self.max_depth,
            )
        )
        return self  # type: ignore

    def fix_bias(self, X: np.ndarray) -> float:
        """Fix the bias of the tree, and return it.

        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        """
        self.bias = np.mean(self.predict(X))  # type: ignore
        return self.bias  # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the values of the given data.

        Parameters
        ----------
        X : np.ndarray
            The data to predict.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        if self.is_selected:
            return self.learning_rate * self.list_tree_(X) - self.bias
        return np.zeros(len(X))

    def get_mean_absolute_score(self, x: np.ndarray) -> float:
        """Compute the mean absolute score of the estimator.

        Parameters
        ----------
        x : np.ndarray of length n_samples
            The data to score.

        Returns
        -------
        float
            The mean absolute score.
        """
        return np.mean(np.abs(self.predict(x)))  # type: ignore

    def get_split_count(self) -> int:
        """Get the number of splits in the tree."""
        if self.is_selected:
            return len(self.list_tree_.split_values)
        return 0


@attrs.define(slots=True)
class MCListTreeRegressor:
    """A 1D Monte Carlo Decision Tree Regressor.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    random_generator : np.random.Generator
        The random generator to use.
    max_depth : int
        The maximum depth of the list.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    l2_regularization : float
        The L2 regularization parameter.
    bias : float
        A correction term to add to the tree's predictions.
    learning_rate : float
        A correction term to multiply the tree's predictions by.
    is_selected : bool
        Whether the tree is the null tree or not.
    feature_name : str
        The name of the feature the tree splits on.
    output_name : str
        The name of the output the tree predicts.

    Attributes
    ----------
    list_tree_ : ListTree
        The tree represented as a list. It is learnt after fitting.
    """

    # Hyper-parameters
    max_depth: int = attrs.field(default=3)
    min_samples_split: int = attrs.field(default=2)
    min_samples_leaf: int = attrs.field(default=1)
    l2_regularization: float = attrs.field(default=0.01)
    # Ensemble parameters
    n_estimators: int = attrs.field(default=10)
    random_generator: np.random.Generator = attrs.field(default=np.random.default_rng())
    # Correction parameters
    bias: float = attrs.field(default=0.0)
    learning_rate: float = attrs.field(default=1.0)
    is_selected: bool = attrs.field(default=True)
    # Model information
    feature_name: str | None = attrs.field(default=None)
    output_name: str | None = attrs.field(default=None)
    # Learnt after fitting
    list_tree_: ListTree = attrs.field(init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        return self  # TODO: implement


def sum_trees(trees: list[ListTree]) -> ListTree:
    """Sum a list of trees together, as piecewise constant functions.

    Parameters
    ----------
    trees : list[ListTree]
        The trees to sum.

    Returns
    -------
    ListTree
        The sum of the trees.
    """
    merged_splits = np.unique(np.concatenate([tree.split_values for tree in trees]))
    merged_leaves = np.empty(len(merged_splits) + 1, dtype=np.float32)
    merged_leaves[0] = np.sum([tree.leaf_values[0] for tree in trees])
    merged_leaves[1:] = np.sum([tree(merged_splits) for tree in trees], axis=0)
    return ListTree(
        split_values=merged_splits,
        leaf_values=merged_leaves,
    )


def sum_tree_regressors(
    regressors: list[ListTreeRegressor],
    feature_name: str,
    output_name: str,
) -> ListTreeRegressor:
    """Sum a list of tree regressors together, as piecewise constant functions.
    It is assumed that all regressors have the same hyper-parameters.
    The bias is added to the sum of the trees.

    Parameters
    ----------
    regressors : list[ListTreeRegressor]
        The tree regressors to sum. If empty, the null tree is returned.
    feature_name : str
        The name of the feature.
    output_name : str
        The name of the output.

    Returns
    -------
    ListTreeRegressor
        The sum of the tree regressors, with the given bias.
    """
    if len(regressors) == 0:
        return ListTreeRegressor(
            is_selected=False, feature_name=feature_name, output_name=output_name
        )
    tree = sum_trees([regressor.list_tree_ for regressor in regressors])
    first_regressor = regressors[0]
    regressor = ListTreeRegressor(
        max_depth=first_regressor.max_depth,
        min_samples_split=first_regressor.min_samples_split,
        min_samples_leaf=first_regressor.min_samples_leaf,
        l2_regularization=first_regressor.l2_regularization,
        learning_rate=first_regressor.learning_rate,
        feature_name=feature_name,
        output_name=output_name,
    )
    regressor.list_tree_ = tree
    return regressor


@nb.njit(
    nb.types.Tuple((nb.int32, nb.boolean))(nb.float32[:], nb.float32),
    fastmath=True,
    cache=True,
)
def best_split(y: np.ndarray, l2_regularization: float) -> tuple[int, bool]:
    """Finds the index of the best split, and whether the best split is
    worth it."""
    left_gradient = np.cumsum(y)
    left_hessian = np.arange(1, len(y) + 1)
    total_gradient = left_gradient[-1]
    total_hessian = left_hessian[-1]
    right_gradient = total_gradient - left_gradient
    right_hessian = total_hessian - left_hessian

    gain = (left_gradient * left_gradient) / (left_hessian + l2_regularization) + (
        right_gradient * right_gradient
    ) / (right_hessian + l2_regularization)
    loss = (total_gradient * total_gradient) / (total_hessian + l2_regularization)

    best_index: int = np.argmax(gain)  # type: ignore
    best_gain = gain[best_index]

    return best_index, best_gain <= loss


@nb.njit(
    nb.types.Tuple((nb.float32[:], nb.float32[:]))(
        nb.float32[:], nb.float32[:], nb.float32, nb.int64, nb.int64, nb.int64
    ),
    fastmath=True,
    cache=True,
)
def build_list_tree(
    X: np.ndarray,
    y: np.ndarray,
    l2_regularization: float,
    min_samples_leaf: int,
    min_samples_split: int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    stack_size = y.size // min_samples_split + 1
    stack = np.empty((stack_size, 3), dtype=np.int64)
    stack_pos = 0
    split_values = np.empty_like(y, dtype=np.float32)
    leaf_values = np.empty_like(y, dtype=np.float32)
    split_stack = np.empty_like(y, dtype=np.float32)
    sv_pos = 0
    lv_pos = 0
    ss_pos = 0
    stack[stack_pos] = (0, y.size, 0)
    stack_pos += 1
    while stack_pos > 0:
        start, end, depth = stack[stack_pos - 1]
        stack_pos -= 1
        n_samples = end - start
        y_node = y[start:end]
        if depth >= max_depth or n_samples < min_samples_split:
            lv = np.sum(y_node) / (n_samples + l2_regularization)
            leaf_values[lv_pos] = lv
            lv_pos += 1
            if ss_pos > 0:
                split_values[sv_pos] = split_stack[ss_pos - 1]
                sv_pos += 1
                ss_pos -= 1
        else:
            index, admissible = best_split(y_node, l2_regularization)
            if admissible or n_samples <= min_samples_leaf:
                lv = np.sum(y_node) / (n_samples + l2_regularization)
                leaf_values[lv_pos] = lv
                lv_pos += 1
                if ss_pos > 0:
                    split_values[sv_pos] = split_stack[ss_pos - 1]
                    sv_pos += 1
                    ss_pos -= 1
            else:
                split = X[start + index]
                split_stack[ss_pos] = split
                ss_pos += 1
                ip = start + np.searchsorted(X[start:end], split, side="right")
                stack[stack_pos] = (ip, end, depth + 1)
                stack_pos += 1
                stack[stack_pos] = (start, ip, depth + 1)
                stack_pos += 1
    return leaf_values[:lv_pos], split_values[:sv_pos]


"""
@nb.njit(
    nb.types.List(nb.types.Tuple((nb.float32[:], nb.float32[:])))(
        nb.float32[:], nb.float32[:], nb.float32,
        nb.int64, nb.int64, nb.int64, nb.int64,
        nb.types.NumPyRandomGeneratorType(),
        nb.float64,
    ),
    fastmath=True,
    cache=True,
    parallel=True,
    nogil=True,
)
"""


def build_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    l2_regularization: float,
    min_samples_leaf: int,
    min_samples_split: int,
    max_depth: int,
    n_estimators: int,
    rng: np.random.Generator,
    subsample: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return_list = []
    row_size = X.shape[0]
    for _ in nb.prange(n_estimators):
        selected_rows = rng.random(row_size) <= subsample
        X_sub = X[selected_rows]
        y_sub = y[selected_rows]
        return_list.append(
            build_list_tree(
                X_sub,
                y_sub,
                l2_regularization,
                min_samples_leaf,
                min_samples_split,
                max_depth,
            )
        )
    return return_list
