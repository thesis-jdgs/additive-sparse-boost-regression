"""Implement a 1D regression tree."""
from __future__ import annotations

import contextlib
import math
from collections import deque

import attrs
import numba as nb
import numpy as np
from sklearn.metrics import mean_squared_error

from potts.potts_wrapper import l2_potts

# from model_helpers._od_tree import build_list_tree
# Python 3.11 feature
with contextlib.suppress(ImportError):
    from typing import Self


@nb.njit(
    nb.float64[:](nb.float32[:], nb.float32[:], nb.float64[:]),
    fastmath=True,
    cache=True,
)
def eval_piecewise(
    x: np.ndarray, split_values: np.ndarray, leaf_values: np.ndarray
) -> np.ndarray:
    """Evaluate a piecewise constant function at a given point."""
    return leaf_values[np.searchsorted(split_values, x, side="right")]


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
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    l2_regularization : float
        The L2 regularization parameter.
    l0_fused_regularization : float
        The L0 fused regularization parameter.
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
    min_samples_leaf: int = attrs.field(default=1)
    l2_regularization: float = attrs.field(default=0.01)
    l0_fused_regularization: float = attrs.field(default=1.0)
    # Correction parameters
    bias: float = attrs.field(default=0.0)
    learning_rate: float = attrs.field(default=1.0)
    is_selected: bool = attrs.field(default=True)
    # Model information
    feature_name: str | None = attrs.field(default=None)
    output_name: str | None = attrs.field(default=None)
    # Learnt after fitting
    list_tree_: ListTree = attrs.field(init=False, repr=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ) -> Self:
        """Fit the model to the given data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples,)
            The training data.
        y_train : np.ndarray of shape (n_samples,)
            The training labels.
        sample_weight : np.ndarray of shape (n_samples,)
            The sample weights.
        X_valid : np.ndarray of shape (n_samples,), optional
            Unused parameter, used by the `ListTreeRegressorCV` subclass.
        y_valid : np.ndarray of shape (n_samples,), optional
            Unused parameter, used by the `ListTreeRegressorCV` subclass.

        Returns
        -------
        Self
            The fitted model.

        """

        self.list_tree_ = ListTree(
            *l2_potts(
                X_train,
                y_train,
                sample_weight,
                l2_regularization=self.l2_regularization,
                l0_fused_regularization=self.l0_fused_regularization,
                excluded_interval_size=self.min_samples_leaf,
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

    def _get_line(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> tuple[int, float]:
        slope = self.get_split_count()
        indices = np.searchsorted(self.list_tree_.split_values, X, side="right")
        prediction = self.list_tree_.leaf_values[indices]
        counts = np.bincount(indices)
        penalization = (
            self.l2_regularization * (self.list_tree_.leaf_values**2) @ counts
        )
        fidelity = np.square(prediction - y) @ sample_weight
        intercept = fidelity + penalization
        return slope, intercept  # type: ignore


@attrs.define(slots=True)
class ListTreeRegressorCV(ListTreeRegressor):
    """A 1D Decision Tree Regressor, represented as a list,
    for faster inference and merging. It is cross-validated, given a range of
    l0 fused regularization parameters.

    Parameters
    ----------
    min_l0_fused_regularization : float
        The minimum L0 fused regularization parameter.
    max_l0_fused_regularization : float
        The maximum L0 fused regularization parameter.

    For the other parameters, see `ListTreeRegressor`.

    """

    # Hyper-parameters
    min_samples_leaf: int = attrs.field(default=1)
    l2_regularization: float = attrs.field(default=0.01)
    min_l0_fused_regularization: float = attrs.field(default=0.0)
    max_l0_fused_regularization: float = attrs.field(default=1.0)
    # Correction parameters
    bias: float = attrs.field(default=0.0)
    learning_rate: float = attrs.field(default=1.0)
    is_selected: bool = attrs.field(default=True)
    # Model information
    feature_name: str | None = attrs.field(default=None)
    output_name: str | None = attrs.field(default=None)
    # Learnt after fitting
    list_tree_: ListTree = attrs.field(init=False, repr=False)

    def _build_regressors(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> list[ListTreeRegressor]:
        left_regressor = ListTreeRegressor(
            l0_fused_regularization=self.min_l0_fused_regularization,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=self.min_samples_leaf,
        )
        left_regressor.fit(X, y, sample_weight)
        right_regressor = ListTreeRegressor(
            l0_fused_regularization=self.max_l0_fused_regularization,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=self.min_samples_leaf,
        )
        right_regressor.fit(X, y, sample_weight)
        a_l, b_l = left_regressor._get_line(X, y, sample_weight)
        a_r, b_r = right_regressor._get_line(X, y, sample_weight)

        regressors = []
        if a_l != a_r:
            queue = deque([(a_l, b_l, a_r, b_r)])
            while queue:
                a_l, b_l, a_r, b_r = queue.pop()
                q = (b_r - b_l) / (a_l - a_r)
                regressor = ListTreeRegressor(
                    l0_fused_regularization=q,
                    l2_regularization=self.l2_regularization,
                    min_samples_leaf=self.min_samples_leaf,
                )
                regressor.fit(X, y, sample_weight)
                a_q, b_q = regressor._get_line(X, y, sample_weight)

                if math.isclose(q * a_q + b_q, q * a_l + b_l, rel_tol=0.05):
                    if a_q != a_l:
                        regressors.append(regressor)
                else:
                    queue.append((a_q, b_q, a_r, b_r))
                    queue.append((a_l, b_l, a_q, b_q))
        cardinality_list = {regressor.get_split_count() for regressor in regressors}
        if left_regressor.get_split_count() not in cardinality_list:
            regressors.append(left_regressor)
        if right_regressor.get_split_count() not in cardinality_list:
            regressors.append(right_regressor)
        return regressors

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ):
        """Fit the model.

        Parameters
        ----------
        X_train : np.ndarray
            The training features.
        y_train : np.ndarray
            The training labels.
        sample_weight : np.ndarray
            The training sample weights.
        X_valid : np.ndarray
            The validation features.
        y_valid : np.ndarray
            The validation labels.
        """
        regressor_list = self._build_regressors(X_train, y_train, sample_weight)
        score_list = [
            mean_squared_error(y_valid, regressor.predict(X_valid))
            for regressor in regressor_list
        ]
        best_index = np.argmin(score_list)
        self.list_tree_ = regressor_list[best_index].list_tree_
        self.l0_fused_regularization = regressor_list[
            best_index
        ].l0_fused_regularization
        return self


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
        min_samples_leaf=first_regressor.min_samples_leaf,
        l2_regularization=first_regressor.l2_regularization,
        l0_fused_regularization=first_regressor.l0_fused_regularization,
        learning_rate=first_regressor.learning_rate,
        feature_name=feature_name,
        output_name=output_name,
    )
    regressor.list_tree_ = tree
    return regressor


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float32[:]))(
        nb.float32[:],
        nb.float64[:],
        nb.float64[:],
        nb.float32,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
    cache=True,
)
def build_list_tree(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    l2_regularization: float,
    min_samples_leaf: int,
    min_samples_split: int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    stack_size = y.size // min_samples_split + 1
    stack = np.empty((stack_size, 3), dtype=np.int64)
    stack_pos = 0
    split_values = np.empty_like(y, dtype=np.float32)
    leaf_values = np.empty_like(y, dtype=np.float64)
    split_stack = np.empty_like(y, dtype=np.float32)
    sv_pos = 0
    lv_pos = 0
    ss_pos = 0
    stack[stack_pos] = (0, y.size, 0)
    stack_pos += 1

    y_weighted = y * sample_weight
    cumulative_gradient = np.cumsum(y_weighted)
    cumulative_hessian = np.cumsum(sample_weight)
    while stack_pos > 0:
        start, end, depth = stack[stack_pos - 1]
        stack_pos -= 1
        n_samples = end - start
        if (
            depth >= max_depth
            or n_samples < min_samples_split
            or n_samples <= min_samples_leaf
        ):
            if start > 0:
                numerator = (
                    cumulative_gradient[end - 1] - cumulative_gradient[start - 1]
                )
                denominator = (
                    cumulative_hessian[end - 1] - cumulative_hessian[start - 1]
                )
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
            left_gradient = (
                cumulative_gradient[start:end]
                - cumulative_gradient[start]
                + y_weighted[start]
            )
            left_hessian = (
                cumulative_hessian[start:end]
                - cumulative_hessian[start]
                + sample_weight[start]
            )
            total_gradient = left_gradient[-1]
            total_hessian = left_hessian[-1]
            right_gradient = total_gradient - left_gradient
            right_hessian = total_hessian - left_hessian

            gain = np.square(left_gradient) / (left_hessian + l2_regularization)
            gain += np.square(right_gradient) / (right_hessian + l2_regularization)
            loss = (total_gradient * total_gradient) / (
                total_hessian + l2_regularization
            )

            index: int = np.argmax(gain)  # type: ignore
            if gain[index] <= loss:
                if start > 0:
                    numerator = (
                        cumulative_gradient[end - 1] - cumulative_gradient[start - 1]
                    )
                    denominator = (
                        cumulative_hessian[end - 1] - cumulative_hessian[start - 1]
                    )
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
                stack[stack_pos] = (ip, end, depth + 1)
                stack_pos += 1
                stack[stack_pos] = (start, ip, depth + 1)
                stack_pos += 1
    return leaf_values[:lv_pos], split_values[:sv_pos]
