"""Implement a preprocessor for tree-based estimators."""
import contextlib

import attrs
import numpy as np
from category_encoders.glmm import GLMMEncoder
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

from model_helpers.custom_types import Data
from model_helpers.custom_types import Target
from model_helpers.od_tree import Self


@attrs.define(slots=False, kw_only=True)
class EmptyTransformer(BaseEstimator, TransformerMixin):
    """Implement a transformer that does nothing."""

    def fit(self, X: Data, y: Target = None) -> Self:  # noqa
        return self

    def transform(self, X: Data) -> np.ndarray:  # noqa
        return X

    def inverse_transform(self, X: Data) -> np.ndarray:
        return X


@attrs.define(slots=False, kw_only=True)
class TreePreprocessor(BaseEstimator, TransformerMixin):
    """Implement a preprocessor for tree-based estimators.

    Parameters
    ----------
    fill_value : float, default=-1
        The value to use to fill the missing values.
    categorical_features : list of int, optional
        The indices of the categorical features.
    max_bins : int, default=256
        The maximum number of bins to use for discretization.
    random_state : int, optional
        The random state to use for discretization.
    """

    fill_value: float = attrs.field(default=-1)
    categorical_features: list[int] = attrs.field(repr=False, factory=list)
    max_bins: int = attrs.field(default=256)
    random_state: int = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        """Initialize all the preprocessor steps."""
        self._imputer = SimpleImputer(strategy="constant", fill_value=self.fill_value)
        self._categorical_encoder = GLMMEncoder()
        self._discretizer = KBinsDiscretizer(
            n_bins=self.max_bins,
            encode="ordinal",
            random_state=self.random_state,
            dtype=np.float32,
        )
        self._scaler = EmptyTransformer()  # StandardScaler(with_std=False)
        self._has_categorical_features = len(self.categorical_features) > 0

    def fit(self, X: Data, y: Target) -> Self:
        self._numerical_features = np.setdiff1d(
            np.arange(X.shape[1]), self.categorical_features
        )
        self._has_numerical_features = len(self._numerical_features) > 0
        X_ = self._imputer.fit_transform(X)
        if self._has_categorical_features:
            X_[:, self.categorical_features] = self._categorical_encoder.fit_transform(
                X_[:, self.categorical_features], y
            )
        if self._has_numerical_features:
            X_[:, self._numerical_features] = self._discretizer.fit_transform(
                X_[:, self._numerical_features]
            )
        self._scaler.fit(X_)
        return self

    def transform(self, X: Data) -> np.ndarray:
        X_ = self._imputer.transform(X)
        if self._has_categorical_features:
            X_[:, self.categorical_features] = self._categorical_encoder.transform(
                X_[:, self.categorical_features]
            )
        if self._has_numerical_features:
            X_[:, self._numerical_features] = self._discretizer.transform(
                X_[:, self._numerical_features]
            )
        return self._scaler.transform(X_)

    def inverse_transform(self, X: Data) -> np.ndarray:
        X_ = self._scaler.inverse_transform(X)
        with contextlib.suppress(NotFittedError):
            X_[:, self._numerical_features] = self._discretizer.inverse_transform(
                X_[:, self._numerical_features]
            )
        return X_
