"""Tests for the models to be proper sklearn estimators."""
import unittest

import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


R2_THRESHOLD = 0.5


def generate_data():
    """Make a simple regression dataset."""
    return make_regression(
        n_samples=20_000,
        n_features=10,
        n_informative=1,
        bias=5.0,
        noise=20,
        random_state=42,
    )


class TestSparseAdditiveBoostingRegressor(unittest.TestCase):
    def setUp(self):
        self.X, self.y = generate_data()
        self.regressor = xgb.XGBRegressor()

    def test_fit(self):
        self.regressor.fit(self.X, self.y)

    def test_predict(self):
        self.regressor.fit(self.X, self.y)
        y_pred = self.regressor.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_score(self):
        self.regressor.fit(self.X, self.y)
        y_pred = self.regressor.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        self.assertGreaterEqual(r2, R2_THRESHOLD)

    def test_clone(self):
        self.regressor.fit(self.X, self.y)
        cloned_regressor = clone(self.regressor)
        self.assertIsNot(self.regressor, cloned_regressor)
        self.assertEqual(self.regressor.get_params(), cloned_regressor.get_params())

    def test_pipeline(self):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", self.regressor),
            ]
        )
        pipeline.fit(self.X, self.y)
        pipeline.predict(self.X)

    def test_grid_search(self):
        param_grid = {
            "n_estimators": [10, 20],
            "learning_rate": [0.1, 0.2],
            "max_depth": [2, 3],
        }
        grid_search = GridSearchCV(
            self.regressor,
            param_grid,
            cv=3,
        )
        grid_search.fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
