# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np

from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_equal

from estimators import Estimator
from estimators import RegressionEstimator
from estimators import MulticlassEstimator


def test_reals():
    X, y = make_regression(n_samples=100, n_features=10, random_state=0)

    est = RegressionEstimator()
    est.fit(X, y)
    y_pred = est.predict(X)

    assert_equal(len(y_pred.shape), 1)
    assert_equal(y_pred.shape[0], X.shape[0])
    assert_equal(y_pred.dtype, np.float64)

    df = est.decision_function(X)
    assert_equal(df.shape[0], X.shape[0])


def test_probability_simplex():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                               n_informative=5, random_state=0)

    est = MulticlassEstimator()
    est.fit(X, y)
    y_pred = est.predict(X)

    assert_equal(len(y_pred.shape), 1)
    assert_equal(y_pred.shape[0], X.shape[0])
    assert_equal(y_pred.dtype, np.int64)

    df = est.decision_function(X)
    assert_equal(df.shape[0], X.shape[0])
