"""Implement the additive sparse boosting regressor."""
from __future__ import annotations

import contextlib
import warnings

import attrs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from model_helpers.custom_types import Data
from model_helpers.custom_types import OneVectorFunction
from model_helpers.custom_types import Target
from model_helpers.custom_types import TwoVectorFunction
from model_helpers.mrmr_functions import absolute_correlation_matrix
from model_helpers.mrmr_functions import f_regression_score
from model_helpers.od_tree import ListTreeRegressor
from model_helpers.od_tree import sum_tree_regressors
from model_helpers.plotting import plot_categorical
from model_helpers.plotting import plot_continuous
from model_helpers.preprocessor import TreePreprocessor

# Python 3.11 compatibility
with contextlib.suppress(ImportError):
    from typing import Self


__version__ = "0.0.1"


@attrs.define(slots=False, kw_only=True)
class SparseAdditiveBoostingRegressor(BaseEstimator, RegressorMixin):
    """A sparse generalized additive model with decision trees.

    Parameters
    ----------
    n_estimators : int, default=100, range=[0, inf)
        The number of estimators to use in the ensemble.
    learning_rate : float, default=0.7, range=(0.0, 1.0]
        The learning rate, that is to say, the shrinkage factor of the estimators.
    random_state : int, optional
        The random state to use. It is used for subsampling and split finding.
    row_subsample : float, default=0.632, range=(0.0, 1.0]
        The fraction of rows to subsample on each iteration.
    validation_fraction : float, default=0.2, range=(0.0, 1.0]
        The fraction of rows to use for validation in case
        validation_set is not provided to the fit method.
    n_iter_no_change : int, default=5, range=[0, inf)
        The number of iterations without improvement to wait before stopping.
    fill_value : float, default=10_000
        The value to use to fill the missing values.
    max_bins: int, default=256
        The maximum number of bins to use for discretization.
    max_depth : int, default=2, range=[1, inf)
        The maximum depth of each tree.
    min_samples_split : int, default=2, range=[1, inf)
        The minimum number of samples to split.
    min_samples_leaf : int, default=1, range=[1, inf)
        The minimum number of samples to be a leaf.
    l2_regularization : float, default=0.1, range=[0.0, inf)
        The L2 regularization to use for the gain.
    l1_regularization : float, default=1.0, range=[0.0, inf)
        The L1 regularization to use for the gain.
    relevancy_scorer : TwoVectorFunction, default=f_regression_score
        The function to use to score the relevancy of the features.
    redundancy_matrix : OneVectorFunction, default=absolute_correlation_matrix
        The function to use to compute the redundancy matrix.
    mrmr_scheme : TwoVectorFunction, default=np.subtract
        The minimum redundancy maximum relevancy scheme to use.
    categorical_features : list of int, optional
        The indices of the categorical features. The estimator will never try to
        deduce which features are categorical without this information.
    feature_names_in_ : np.ndarray of str, optional
        The names of the features.
    output_name : str, optional
        The name of the output.

    Attributes
    ----------
    preprocessor_ : TreePreprocessor
        The estimator used to preprocess the data.
    estimators_ : list of ListTreeRegressor
        The estimators used to fit the data.
    intercept_ : float
        The intercept of the model, which corrects for the mean of the target.
    selection_history_ : np.ndarray
        The history of the selected features at each iteration.
    score_history_ : np.ndarray
        The history of the scores at each iteration. Contains the training and
        validation RMSE.
    selection_count_ :  np.ndarray
        The number of times each feature was selected.
    selected_features_ : np.ndarray
        The indices of the selected features.
    """

    # Boosting Hyper-parameters
    n_estimators: int = attrs.field(
        default=100, validator=attrs.validators.ge(1), converter=int
    )
    learning_rate: float = attrs.field(
        default=0.1,
        validator=[attrs.validators.gt(0.0), attrs.validators.le(1.0)],
        converter=float,
    )
    random_state: int = attrs.field(default=None)
    row_subsample: float = attrs.field(
        default=0.632,
        validator=[attrs.validators.gt(0.0), attrs.validators.le(1.0)],
        converter=float,
    )
    # Validation Hyper-parameters
    validation_fraction: float = attrs.field(
        default=0.1,
        validator=[attrs.validators.gt(0.0), attrs.validators.le(1.0)],
        converter=float,
    )
    n_iter_no_change: int = attrs.field(
        default=10, validator=attrs.validators.ge(1), converter=int
    )
    # Preprocessor Hyper-parameters
    fill_value: float = attrs.field(default=10_000, converter=float)
    max_bins: int = attrs.field(
        default=256, validator=attrs.validators.gt(0), converter=int
    )
    # Tree Hyper-parameters
    max_depth: int = attrs.field(
        default=2, validator=attrs.validators.ge(1), converter=int
    )
    l2_regularization: float = attrs.field(
        default=0.1, validator=attrs.validators.gt(0.0), converter=float
    )
    l1_regularization: float = attrs.field(
        default=1.0, validator=attrs.validators.ge(0.0), converter=float
    )
    min_samples_split: int = attrs.field(
        default=2, validator=attrs.validators.ge(2), converter=int
    )
    min_samples_leaf: int = attrs.field(
        default=1, validator=attrs.validators.ge(1), converter=int
    )
    # Functions for scoring and selecting features
    relevancy_scorer: TwoVectorFunction = attrs.field(default=f_regression_score)
    redundancy_matrix: OneVectorFunction = attrs.field(
        default=absolute_correlation_matrix
    )
    mrmr_scheme: TwoVectorFunction = attrs.field(default=np.subtract)
    # Optional information arguments
    categorical_features: list[int] = attrs.field(factory=list)
    feature_names_in_: np.ndarray = attrs.field(default=None)
    output_name: str = attrs.field(default=None)
    # Parameters learnt after fitting
    preprocessor_: TreePreprocessor = attrs.field(init=False, repr=False)
    regressors_: list[ListTreeRegressor] = attrs.field(init=False, repr=False)
    intercept_: float = attrs.field(init=False, repr=False)
    selection_history_: np.ndarray = attrs.field(init=False, repr=False)
    score_history_: np.ndarray = attrs.field(init=False, repr=False)
    selection_count_: np.ndarray = attrs.field(init=False, repr=False)
    # Private attributes
    _regressors: list[list[ListTreeRegressor]] = attrs.field(init=False, repr=False)
    _random_generator: np.random.Generator = attrs.field(init=False, repr=False)
    _n: int = attrs.field(init=False, repr=False)
    _m: int = attrs.field(init=False, repr=False)
    _is_fitted: bool = attrs.field(init=False, repr=False, default=False)
    _indexing_cache: list[tuple[np.ndarray, np.ndarray] | None] = attrs.field(
        init=False, repr=False
    )

    @selection_history_.default  # type: ignore
    def _set_selection_history(self) -> np.ndarray:
        return np.empty(self.n_estimators, dtype=int)

    @score_history_.default  # type: ignore
    def _set_score_history(self) -> np.ndarray:
        return np.empty((self.n_estimators, 2), dtype=float)

    @_random_generator.default  # type: ignore
    def _set_random_generator(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    @preprocessor_.default  # type: ignore
    def _set_preprocessor(self) -> TreePreprocessor:
        return TreePreprocessor(
            categorical_features=self.categorical_features,
            fill_value=self.fill_value,
            max_bins=self.max_bins,
            random_state=self.random_state,
        )

    def fit(
        self, X: Data, y: Target, validation_set: tuple[Data, Target] | None = None
    ) -> Self:
        """Fit a generalized additive model with sparse functions, that is to say,
            with features selected by mRMR.

        Parameters
        ----------
        X : Data of shape (n_samples, n_features)
            Training data.
        y : Target of shape (n_samples,)
            Target values.
        validation_set : tuple of Data and Target, optional
            The validation set to use to evaluate the model at each iteration.
            If not provided, then the training set is split into training
            and validation sets.

        Returns
        -------
        Self
            The fitted model.
        """
        if validation_set is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state
            )
        else:
            X_val, y_val = validation_set
        self._initialize_fit_params(X, y)
        # Convert the data to numpy arrays
        y_train = np.array(y, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train = np.array(
                self.preprocessor_.fit_transform(X, y_train), dtype=np.float32
            )
        X_val = np.array(self.preprocessor_.transform(X_val), dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float64)
        # Fit the model and correct the bias
        self._fit(X_train, y_train, X_val, y_val)
        self.regressors_ = [
            sum_tree_regressors(estimators, name, self.output_name)
            for estimators, name in zip(self._regressors, self.feature_names_in_)
        ]
        self.intercept_ += np.sum(
            [
                regressor.fix_bias(X_train[:, i])
                for i, regressor in enumerate(self.regressors_)
            ]
        )
        # Get the history
        self.selection_history_ = self.selection_history_[: self.n_estimators]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.sqrt(self.score_history_, out=self.score_history_)
        self._is_fitted = True
        return self  # type: ignore

    def _initialize_fit_params(self, X: Data, y: Target) -> None:
        """Initialize the parameters for fitting."""
        self._m, self._n = np.array(X).shape
        if self.feature_names_in_ is None:
            self.feature_names_in_ = (
                np.array(X.columns)
                if isinstance(X, pd.DataFrame)
                else np.array([f"feature_{i}" for i in range(self._n)])
            )
        if self.output_name is None:
            self.output_name = y.name if isinstance(y, pd.Series) else "output"
        self._n_trials = int(self.row_subsample * self._m)
        self._regressors: list[list[ListTreeRegressor]] = [[] for _ in range(self._n)]
        self._indexing_cache = [None for _ in range(self._n)]
        self.selection_count_ = np.zeros(self._n, dtype=np.float32)

    def _fit(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """Run the fitting algorithm."""
        self.intercept_ = np.mean(y)  # type: ignore
        residual = y - self.intercept_
        residual_valid = y_val - self.intercept_
        # First round
        self._boost(X, residual, 0.0, 0, X_val, residual_valid)
        # Initialize redundancy for the next rounds
        redundancy_matrix = self.redundancy_matrix(X).astype(np.float32)
        redundancy = np.zeros_like(X[0])
        for model_count in range(1, self.n_estimators):
            redundancy += redundancy_matrix[self.selection_history_[model_count - 1]]
            self._boost(
                X,
                residual,
                redundancy / model_count,
                model_count,
                X_val,
                residual_valid,
            )
            if model_count >= self.n_iter_no_change:
                last_iterations = self.score_history_[
                    model_count - self.n_iter_no_change : model_count, 1
                ]
                if np.all(np.diff(last_iterations) >= np.finfo(np.float32).eps):
                    break
        # Early stopping
        self.score_history_ = self.score_history_[
            : model_count + 1  # noqa pyUnresolvedReferences
        ]
        self.n_estimators = np.argmin(self.score_history_[:, 1])  # type: ignore
        self._regressors = self._regressors[: self.n_estimators]
        self.selection_history_ = self.selection_history_[: self.n_estimators]
        self.selection_count_ = np.unique(self.selection_history_, return_counts=True)[
            1
        ]

    def _boost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        redundancy: np.ndarray | float,
        model_count: int,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Boost the model by fitting a new function and updating the residual."""
        # Select the best feature
        relevancy = self.relevancy_scorer(X, y)
        score = self.mrmr_scheme(relevancy, redundancy)
        selected_feature: int = np.argmax(score)  # type: ignore
        self.selection_history_[model_count] = selected_feature
        self.selection_count_[selected_feature] += 1
        # Split the data
        x_passed, splitted = self._get_index(X, selected_feature)
        lengths = np.array([len(indices) for indices in splitted])
        min_size = np.min(lengths)
        size = int(self.row_subsample * min_size)
        random_indexes = self._random_generator.choice(np.arange(min_size), size=size)
        y_passed = np.array([y[indices][random_indexes].mean() for indices in splitted])
        # Fit the model on the selected feature and sampled rows
        new_model = ListTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            l2_regularization=self.l2_regularization,
            l1_regularization=self.l1_regularization,
            learning_rate=self.learning_rate,
        )
        new_model.fit(x_passed, y_passed)
        # Update the residual
        y_pred = new_model.predict(X[:, selected_feature])
        y -= y_pred
        # Score the model on the validation set
        y_pred_val = new_model.predict(X_val[:, selected_feature])
        y_val -= y_pred_val
        self.score_history_[model_count] = [
            np.square(y).mean(),
            np.square(y_val).mean(),
        ]
        # Update the ensemble
        self._regressors[selected_feature].append(new_model)

    def _get_index(self, X: np.ndarray, feature: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the ordered indexes of the feature."""
        if self._indexing_cache[feature] is None:
            x = X[:, feature]
            x_passed, inverse_indices = np.unique(x, return_inverse=True)
            sorted_indices = np.argsort(inverse_indices)
            sorted_x = x[sorted_indices]
            split_indices = np.where(np.diff(sorted_x) != 0)[0] + 1
            splitted = np.split(sorted_indices, split_indices)
            self._indexing_cache[feature] = (x_passed, splitted)
        return self._indexing_cache[feature]  # type: ignore

    def contribution_frame(self, X: Data) -> pd.DataFrame:
        """DataFrame of the contribution of each feature for each sample.
        Each row is a sample and each column is a feature.
        The first column is the intercept.

        Parameters
        ----------
        X : Data of shape (n_samples, n_features)
            The data to compute the contribution for.

        Returns
        -------
        contribution : DataFrame of shape (n_samples, n_features + 1)
            The contribution matrix.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self} cannot predict before calling fit.")
        X_arr = np.array(self.preprocessor_.transform(X))
        contribution = np.empty((len(X), self._n + 1), dtype=float)
        contribution[:, 0] = self.intercept_
        contribution[:, 1:] = np.array(
            [
                regressor.predict(X_arr[:, i])
                for i, regressor in enumerate(self.regressors_)
            ]
        ).T
        contribution_df = pd.DataFrame(
            contribution, columns=["intercept"] + self.feature_names_in_
        )
        return contribution_df

    def predict(self, X: Data) -> Target:
        """Predict the response for the data.

        Parameters
        ----------
        X : Data of shape (n_samples, n_features)
            The data to predict the response for.

        Returns
        -------
        Target of shape (n_samples,)
            The predicted response.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self} cannot predict before calling fit.")
        X_arr = np.array(self.preprocessor_.transform(X), dtype=np.float32)
        return self.intercept_ + np.sum(
            [
                regressor.predict(X_arr[:, i])
                for i, regressor in enumerate(self.regressors_)
                if regressor.is_selected
            ],
            axis=0,
        )

    def plot_model_information(self) -> None:
        """Plot the model information that was collected during training.
        In particular, the plot includes training and validation error,
         as well as the selected features.
        Another plot shows the complexity of the model at each feature.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self} cannot plot before calling fit.")
        # Printing non-selected features
        non_selected = [
            model.feature_name for model in self.regressors_ if model.is_selected
        ]
        print(f"The following features were not selected: {non_selected}")
        # Plot score history
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        training, validation = self.score_history_.T
        iteration_count = np.arange(self.n_estimators)
        fig.add_trace(
            go.Scatter(
                x=iteration_count,
                y=validation,
                name="validation rmse",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=iteration_count,
                y=training,
                name="training rmse",
                line=dict(color="blue"),
            ),
            secondary_y=True,
        )
        # Add selected features to the plot
        fig.add_trace(
            go.Scatter(
                x=iteration_count,
                y=self.feature_names_in_[self.selection_history_],
                name="feature",
                mode="markers",
                marker=dict(color="green", symbol="square", size=1.3),
            ),
            secondary_y=False,
        )
        fig.update_layout(
            showlegend=True,
            template="plotly",
            title="Iteration history",
            xaxis_title="Number of iterations",
            yaxis_title="Selected feature",
            yaxis2_title=f"RMSE {self.output_name}",
        )
        fig.show()

        # Plot complexities
        pd.options.plotting.backend = "plotly"
        complexities = pd.Series(
            {
                model.feature_name: model.get_split_count()
                for model in self.regressors_
                if model.is_selected
            }
        )
        fig = complexities.sort_values().plot.barh()
        fig.update_layout(
            yaxis_title="Features",
            xaxis_title="Split count",
            title="Complexity of each estimator",
        )
        fig.show()

    def explain(self, X: Data) -> None:
        """Explain the model decision at the data X.
        X can be the training data to explain how the model was built,
        or can be new data points to explain the model decision at these points.

        Parameters
        ----------
        X : Data of shape (n_samples, n_features)
            The data to explain the model decision at.
        """
        if not self._is_fitted:
            raise NotFittedError(f"{self} cannot plot before calling fit.")
        selected = [
            (feature_index, model)
            for feature_index, model in enumerate(self.regressors_)
            if model.is_selected
        ]

        # Plot mean importances
        X_train = np.array(X, dtype=np.float32)
        X_arr = np.array(self.preprocessor_.transform(X), dtype=np.float32)
        pd.options.plotting.backend = "plotly"
        scores = pd.Series(
            {
                model.feature_name: model.get_mean_absolute_score(
                    X_arr[:, feature_index]
                )
                for feature_index, model in selected
            }
        )
        scores["intercept"] = self.intercept_
        pd.options.plotting.backend = "plotly"
        fig = scores.sort_values().plot.barh()
        fig.update_layout(
            yaxis_title="Features",
            xaxis_title="Mean absolute score",
            title="Mean importance of each feature",
        )
        fig.show()

        # Plot importances
        for feature_index, model in selected:
            if feature_index not in self.categorical_features:
                x_vals, index = np.unique(X_arr[:, feature_index], return_index=True)
                x_to_plot = self.preprocessor_.inverse_transform(X_arr[index])[
                    :, feature_index
                ]
                fig = plot_continuous(
                    model,
                    x_to_plot,
                    x_vals,
                )
            else:
                x_to_plot = X_train[:, feature_index]
                x_vals = X_arr[:, feature_index]
                fig = plot_categorical(
                    model,
                    x_to_plot,
                    x_vals,
                )
            fig.show()


def __main__(learning_rate: float = 0.5, n_estimators: int = 50, **kwargs) -> None:
    """Run the California housing example."""
    import time

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    def evaluate(model):
        start = time.perf_counter_ns()
        model.fit(X_train, y_train)
        end = time.perf_counter_ns()
        print(f"Training time: {(end - start) / 1e9:.2f} seconds")
        print(f"Training score: {model.score(X_train, y_train):.4f}")
        print(f"Testing score: {model.score(X_test, y_test):.4f}")
        print(f"Number of features: {model.selection_count_}")

    # Evaluate the model
    model = SparseAdditiveBoostingRegressor(
        learning_rate=learning_rate, n_estimators=n_estimators, **kwargs
    )
    evaluate(model)
    print(model.n_estimators)


if __name__ == "__main__":
    __main__(
        n_estimators=1_000,
        learning_rate=0.3,
        l2_regularization=3.0,
        l1_regularization=1.0,
        max_depth=6,
        random_state=0,
        row_subsample=0.85,
    )
