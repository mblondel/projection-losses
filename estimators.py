# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import add_dummy_feature
from sklearn.metrics.pairwise import pairwise_kernels

from polytopes import UnitCube
from polytopes import ProbabilitySimplex
from polytopes import Knapsack
from polytopes import CartesianProduct
from polytopes import Birkhoff
from polytopes import Permutahedron
from polytopes import OrderSimplex


class Reals(object):

    def Euclidean_project(self, theta):
        # Identity function.
        return theta

    def MAP(self, theta):
        # For ordinal regression only.
        return np.round(theta).ravel()


def Shannon_negentropy(u):
    mask = u > 0
    return np.sum(u[mask] * np.log(u[mask]))


class Estimator(BaseEstimator):

    def __init__(self, projection_type="Euclidean", projection_set="unit-cube",
                 map_set=None, min_labels=0, max_labels=None,
                 alpha=1.0, fit_intercept=True,
                 kernel=None, degree=3, coef0=1, gamma=1,
                 max_iter=500, tol=1e-5,
                 random_state=None, verbose=0):
        self.projection_type = projection_type
        self.projection_set = projection_set
        self.map_set = map_set
        self.min_labels = min_labels
        self.max_labels = max_labels
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def _get_set(self, name):
        d = {
            "reals": Reals(),
            "unit-cube": UnitCube(),
            "simplex": ProbabilitySimplex(),
            "Cartesian-cube": CartesianProduct(UnitCube()),
            "Cartesian-simplex": CartesianProduct(ProbabilitySimplex()),
            "knapsack": Knapsack(min_labels=self.min_labels,
                                 max_labels=self.max_labels),
            "Birkhoff": Birkhoff(),
            "permutahedron": Permutahedron(),
            "order-simplex": OrderSimplex(),
        }

        if not name in d:
            raise ValueError("Invalid polytope / set name.")

        return d[name]

    def _get_projection_set(self):
        return self._get_set(self.projection_set)

    def _get_map_set(self):
        if self.map_set is None:
            map_set = self.projection_set
        else:
            map_set = self.map_set

        return self._get_set(map_set)

    def _solve_lbfgs(self, X, Y):
        n_samples, n_features = X.shape
        # If Y.shape = n_samples x n_classes, then d = n_classes
        # If Y.shape = n_samples x n_classes x n_classes, then d = n_classes^2
        d = np.prod(Y.shape[1:])
        polytope = self._get_projection_set()
        Y_flat = Y.reshape(n_samples, -1)

        def _func(coef):
            coef = coef.reshape(d, n_features)

            # n_samples x d
            theta = safe_sparse_dot(X, coef.T)

            if self.projection_type == "Euclidean":
                u = polytope.Euclidean_project(theta)
                loss = np.sum(theta * u)
                loss -= 0.5 * np.sum(u ** 2)
                loss += 0.5 * np.sum(Y ** 2)

            elif self.projection_type == "KL":
                u = polytope.KL_project(theta)
                loss = np.sum(theta * u)
                loss -= Shannon_negentropy(u)
                loss += Shannon_negentropy(Y)

            else:
                raise ValueError("Invalid projection type.")

            loss -= np.sum(Y_flat * theta)
            loss /= n_samples

            # d x n_features
            grad = safe_sparse_dot(u.T, X)
            grad -= safe_sparse_dot(Y_flat.T, X)
            grad /= n_samples

            # Regularization term
            loss += 0.5 * self.alpha * np.sum(coef ** 2)
            grad += self.alpha * coef

            return loss, grad.ravel()

        coef0 = np.zeros(d * n_features, dtype=np.float64)
        coef, funcval, infodic = fmin_l_bfgs_b(_func, coef0,
                                               maxiter=self.max_iter)

        if infodic["warnflag"] != 0:
            print("NOT CONVERGED: ", infodic["task"])

        return coef.reshape(d, n_features)

    def _kernel(self, X):
        return pairwise_kernels(X, self.X_tr_, metric=self.kernel,
                                degree=self.degree, coef0=self.coef0,
                                gamma=self.gamma, filter_params=True)

    def fit(self, X, Y):
        if self.kernel is not None:
            self.X_tr_ = X.copy()
            X = self._kernel(X)

        if self.fit_intercept:
            X = add_dummy_feature(X)

        if hasattr(Y, "toarray"):
            raise ValueError("scipy sparse matrices not supported for Y")

        Y = np.array(Y)

        self.coef_ = self._solve_lbfgs(X, Y)

        return self

    def decision_function(self, X):
        if self.kernel is not None:
            X = self._kernel(X)

        if self.fit_intercept:
            X = add_dummy_feature(X)
        return safe_sparse_dot(X, self.coef_.T)

    def predict(self, X, V=None, b=None):
        """
        V and/or b can be passed to do calibrated decoding (see paper).
        """
        df = self.decision_function(X)

        polytope = self._get_projection_set()

        if self.projection_type == "Euclidean":
            df = polytope.Euclidean_project(df)
        elif self.projection_type == "KL":
            df = polytope.KL_project(df)
        else:
            raise ValueError("Projection type not implemented")

        if V is not None:
            if hasattr(V, "mvec"):
                df = -V.mvec(df)
            else:
                df = df.dot(-V)

        if b is not None:
            df -= b

        return self._get_map_set().MAP(df)


class RegressionEstimator(Estimator):

    def __init__(self, *args, **kw):
        super(RegressionEstimator, self).__init__(*args, **kw)
        self.projection_set = "reals"

    def fit(self, X, y):
        Y = y.reshape(-1, 1)
        return super(RegressionEstimator, self).fit(X, Y)

    def predict(self, X):
        ret = super(RegressionEstimator, self).predict(X)
        return ret.ravel()


class MulticlassEstimator(Estimator):

    def __init__(self, *args, **kw):
        super(MulticlassEstimator, self).__init__(*args, **kw)
        self.projection_set = "simplex"

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        y = self.label_encoder_.transform(y)
        lb = LabelBinarizer(neg_label=0)
        Y = lb.fit_transform(y)
        return super(MulticlassEstimator, self).fit(X, Y)

    def predict(self, X):
        ret = super(MulticlassEstimator, self).predict(X)
        return self.label_encoder_.inverse_transform(ret)
