"""Implement a 1D regression tree."""
from __future__ import annotations

import contextlib

import attrs
import numpy as np

from model_helpers._od_tree import build_list_tree
from model_helpers._od_tree import eval_piecewise


# Python 3.11 feature
with contextlib.suppress(ImportError):
    from typing import Self


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
    l1_regularization: float = attrs.field(default=0.01)
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
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
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
    merged_leaves = np.empty(len(merged_splits) + 1, dtype=np.float64)
    merged_leaves[0] = np.sum([tree.leaf_values[0] for tree in trees])
    merged_leaves[1:] = np.sum([tree(merged_splits) for tree in trees], axis=0)
    return ListTree(
        split_values=merged_splits,
        leaf_values=merged_leaves,
    )


def sum_tree_regressors(
    regressors: list[ListTreeRegressor],
    feature_name: str | None = None,
    output_name: str | None = None,
) -> ListTreeRegressor:
    """Sum a list of tree regressors together, as piecewise constant functions.
    It is assumed that all regressors have the same hyper-parameters.
    The bias is added to the sum of the trees.

    Parameters
    ----------
    regressors : list[ListTreeRegressor]
        The tree regressors to sum. If empty, the null tree is returned.
    feature_name : str, optional
        The name of the feature.
    output_name : str, optional
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
        l1_regularization=first_regressor.l1_regularization,
        learning_rate=first_regressor.learning_rate,
        feature_name=feature_name,
        output_name=output_name,
    )
    regressor.list_tree_ = tree
    return regressor
