# Author: Mathieu Blondel, 2019
# License: BSD

from itertools import product

import numpy as np

import ot
from simplex import project_simplex, constrained_softmax


def _vectorize(func, theta):
    theta = np.array(theta)
    flat = len(theta.shape) == 1

    if flat:
        return func(theta)
    else:
        n_samples = theta.shape[0]
        ret = [func(theta[i]) for i in range(n_samples)]
        return np.array(ret)


class Polytope(object):

    def Euclidean_project(self, theta):
        """
        Compute Euclidean projection.

        Parameters
        ----------
        theta: array, shape = n_samples x n_features
            Input array.

        Returns
        -------
        out: array, shape = n_samples x n_features
            Output array
        """
        return _vectorize(self._Euclidean_project, theta)

    def KL_project(self, theta):
        """
        Compute KL projection.

        Parameters
        ----------
        theta: array, shape = n_samples x n_features
            Input array.

        Returns
        -------
        out: array, shape = n_samples x n_features
            Output array
        """
        return _vectorize(self._KL_project, theta)

    def project(self, theta, projection_type="Euclidean"):
        if projection_type == "Euclidean":
            return self.Euclidean_project(theta)
        elif projection_type == "KL":
            return self.KL_project(theta)
        else:
            raise ValueError("Unknown projection_type.")

    def MAP(self, theta):
        """
        Compute MAP projection.

        Parameters
        ----------
        theta: array, shape = n_samples x n_features
            Input array.

        Returns
        -------
        out: array, shape = n_samples x n_outputs
            Output array
        """
        return self.inv_phi(self.argmax(theta))

    def argmax(self, theta):
        """
        Compute argmax.

        Parameters
        ----------
        theta: array, shape = n_samples x n_features
            Input array.

        Returns
        -------
        out: array, shape = n_samples x n_features
            Output array
        """
        return _vectorize(self._argmax, theta)

    def _MAP(self, theta):
        return self._inv_phi(self._argmax(theta))

    #def max(self, theta):
        #return np.sum(theta * self.argmax(theta), axis=1)

    def phi(self, Y):
        return _vectorize(self._phi, Y)

    def inv_phi(self, Y):
        return _vectorize(self._inv_phi, Y)


class UnitCube(Polytope):

    def Euclidean_project(self, theta):
        return np.minimum(np.maximum(theta, 0), 1)

    def KL_project(self, theta):
        theta = np.array(theta)
        return np.minimum(np.exp(theta - 1), 1)

    def argmax(self, theta):
        theta = np.array(theta)
        return (theta > 0).astype(int)

    def phi(self, y):
        return y

    def inv_phi(self, y):
        return y

    def vertices(self, size):
        for tup in product([0,1], repeat=size):
            yield np.array(tup)


class ProbabilitySimplex(Polytope):

    def Euclidean_project(self, theta):
        theta = np.array(theta)

        if len(theta.shape) == 1:
            return project_simplex(theta)
        elif len(theta.shape) == 2:
            return project_simplex(theta, axis=1)
        else:
            raise ValueError("Invalid shape for theta.")

    def KL_project(self, theta):
        theta = np.array(theta)

        flat = len(theta.shape) == 1
        if flat:
            theta = theta.reshape(1, -1)

        # Just the usual softmax with the usual stability trick.
        max_theta = np.max(theta, axis=1)
        exp_theta = np.exp(theta - max_theta[:, np.newaxis])
        ret = exp_theta / np.sum(exp_theta, axis=1)[:, np.newaxis]

        if flat:
            ret = np.ravel(ret)

        return ret

    # FIXME: vectorize
    def _argmax(self, theta):
        # Return one-hot vectors.
        n_classes = len(theta)
        ret = np.zeros(n_classes)
        ret[np.argmax(theta)] = 1
        return ret

    def MAP(self, theta):
        # Return integers.
        if len(theta.shape) == 1:
            return np.argmax(theta)
        elif len(theta.shape) == 2:
            return np.argmax(theta, axis=1)
        else:
            raise ValueError("Invalid shape for theta.")

    def vertices(self, size):
        I = np.eye(size)
        for row in I:
            yield row


class CartesianProduct(Polytope):

    def __init__(self, polytope):
        self.polytope = polytope

    def _apply_func(self, theta, func):
        # theta should be of shape (n_classes x n_classes,)
        n_classes = int(np.sqrt(theta.shape[0]))
        theta = theta.reshape(n_classes, n_classes)

        u = np.zeros_like(theta)
        for j in range(n_classes):
            u[j] = func(theta[j])

        # Need to return the same shape as theta.
        return u.ravel()

    def _Euclidean_project(self, theta):
        return self._apply_func(theta, self.polytope.Euclidean_project)

    def _KL_project(self, theta):
        return self._apply_func(theta, self.polytope.KL_project)

    def _argmax(self, theta):
        n_classes = int(np.sqrt(theta.shape[0]))
        theta = theta.reshape(n_classes, n_classes)
        ret = np.zeros_like(theta)
        for j in range(n_classes):
            ret[j] = self.polytope.argmax(theta[j])
        return ret.ravel()

    def vertices(self, size):  # size = len(theta)
        n_classes = int(np.sqrt(size))
        for prod in product(np.eye(n_classes), repeat=n_classes):
            yield np.array(prod).ravel()


class Knapsack(Polytope):

    def __init__(self, max_labels, min_labels=0, algo="isotonic"):
        self.max_labels = max_labels
        self.min_labels = min_labels
        self.algo = algo

    def _project_equality(self, theta, n_labels):
        # Project onto {y in [0,1]^k : sum(y) = n_labels}.
        if self.algo == "isotonic":
            w = np.zeros(len(theta))
            w[:n_labels] = 1
            return Permutahedron(w, w_sorted=True).project(theta)

        elif self.algo == "bisection":
            eps = 1e-6
            upper = np.max(theta)
            lower = -upper
            current = np.inf

            for it in range(100):
                if np.abs(current) / n_labels < eps and current < 0:
                    break

                tau = (upper + lower) / 2.0
                mu = np.minimum(np.maximum(theta - tau, 0), 1)
                current = np.sum(mu) - n_labels
                if current <= 0:
                    upper = tau
                else:
                    lower = tau
            return mu

        else:
            raise ValueError("Invalid algorithm name")

    def _Euclidean_project(self, theta):
        # First attempt to project on the unit cube.
        u = np.minimum(np.maximum(theta, 0), 1)
        su = np.sum(u)

        if self.min_labels <= su and su <= self.max_labels:
            # If the inequality is satisfied, we're done.
            return u
        else:
            if su >= self.max_labels:
                return self._project_equality(theta, self.max_labels)
            else:
                return self._project_equality(theta, self.min_labels)

    def _KL_project(self, theta):
        theta = np.array(theta)
        # First attempt to project on the unit cube.
        u = np.minimum(np.exp(theta - 1), 1)
        su = np.sum(u)

        if self.min_labels <= su and su <= self.max_labels:
            # If the inequality is satisfied, we're done.
            return u
        else:
            if su >= self.max_labels:
                n_labels = self.max_labels
            else:
                # su <= 0 should never happen so n_labels can't be 0
                n_labels = self.min_labels

            n_labels = self.max_labels
            z = theta - np.log(n_labels)
            u = np.ones(len(theta)) / float(n_labels)
            return constrained_softmax(z, u) * n_labels

    def _argmax(self, theta):
        theta = np.array(theta)
        sol = np.zeros_like(theta)
        top = np.argsort(theta)[::-1]
        # We pick labels between 'min_labels' and 'max_labels' only if the
        # corresponding theta is non-negative.
        sol[top[self.min_labels:self.max_labels]] = 1
        sol = np.logical_and(sol.astype(bool), theta >= 0)
        sol = sol.astype(int)
        # If 'min_labels' is set, the first 'min_labels' labels must be picked.
        sol[top[:self.min_labels]] = 1
        return sol

    def vertices(self, size):
        max_labels = size if self.max_labels is None else self.max_labels
        for tup in product([0,1], repeat=size):
            ret = np.array(tup)
            s = np.sum(ret)
            if self.min_labels <= s and s <= max_labels:
                yield ret


class Birkhoff(Polytope):

    def __init__(self, max_iter=1000, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol

    def _project(self, theta, regul):
        theta = np.array(theta)
        d = theta.shape[0]
        n_classes = int(np.sqrt(d))
        theta = theta.reshape(n_classes, n_classes)

        o = np.ones(n_classes)

        # We want to solve argmin_T ||T - theta ||^2.
        alpha = ot.solve_semi_dual(o, o, -theta, regul,
                                   max_iter=self.max_iter, tol=self.tol)
        ret = ot.get_plan_from_semi_dual(alpha, o, -theta, regul)

        return ret.ravel()

    def _Euclidean_project(self, theta):
        return self._project(theta, ot.SquaredL2(gamma=1.0))

    def _KL_project(self, theta):
        return self._project(theta, ot.NegEntropy(gamma=1.0))

    def _argmax(self, theta):
        from scipy.optimize import linear_sum_assignment

        n_classes = int(np.sqrt(theta.shape[0]))
        theta = theta.reshape(n_classes, n_classes)

        # We want to maximize.
        rows, cols = linear_sum_assignment(-theta)

        # Construct permutation matrix.
        ret = np.zeros((n_classes, n_classes))
        for j in range(len(rows)):
            ret[rows[j], cols[j]] = 1

        return ret.ravel()

    def _phi(self, y):
        """From permutation to flattend permutation matrix.

        The input y should be of the form y[rank] = label.

        The returned permutation matrix has the form Y[rank, label].
        The matrix is flattened.
        """
        n_classes = y.shape[0]
        ret = np.zeros((n_classes, n_classes))
        for j in range(n_classes):
            ret[j, y[j]] = 1
        return ret.ravel()

    def _inv_phi(self, y):
        """From flattened permutation matrix to permutation."""
        n_classes = int(np.sqrt(Y.shape[0]))
        Y = y.reshape(n_classes, n_classes)

        ret = np.zeros(n_classes)
        for j in range(n_classes):
            ret[j] = np.argmax(Y[j])

        return ret

    def _MAP(self, theta):
        n_classes = int(np.sqrt(theta.shape[0]))
        perm_matrix = self._argmax(theta).reshape(n_classes, n_classes)
        return self._inv_phi(perm_matrix)

    def vertices(self, size):  # size = len(theta)
        size = int(np.sqrt(size))
        for y in Permutahedron().vertices(size):
            yield self._phi(y)

def inv_permutation(p):
    ret = np.zeros(len(p), dtype=np.int)
    ret[p] = np.arange(len(p))
    return ret


class Permutahedron(Polytope):

    def __init__(self, w=None, w_sorted=False):
        self.w = w
        self.w_sorted = w_sorted

    def _Euclidean_project(self, theta):
        from sklearn.isotonic import isotonic_regression

        n_classes = len(theta)
        w = self.w
        if w is None:
            w = np.arange(n_classes)[::-1]

        w = np.array(w)
        if not self.w_sorted:
            w = w[np.argsort(w)[::-1]]
        perm = np.argsort(theta)[::-1]
        theta = theta[perm]
        dual_sol = isotonic_regression(w - theta, increasing=True)
        primal_sol = dual_sol + theta
        return primal_sol[inv_permutation(perm)]

    def _KL_project(self, theta):
        raise NotImplementedError

    def _MAP(self, theta):
        n_classes = len(theta)
        w = self.w
        if w is None:
            w = np.arange(n_classes)[::-1]

        w = np.array(w)
        if not self.w_sorted:
            w = w[np.argsort(w)[::-1]]
        perm = np.argsort(theta)[::-1]
        inv = inv_permutation(perm)
        return w[inv]

    def _argmax(self, theta):
        return self._MAP(theta)

    def _phi(self, y):
        # FIXME: implement this for general w.
        return y

    def vertices(self, size):
        from itertools import permutations
        for perm in permutations(np.arange(size)):
            yield np.array(perm)


class OrderSimplex(Polytope):

    def _Euclidean_project(self, theta):
        from sklearn.isotonic import isotonic_regression
        return isotonic_regression(theta, y_min=0, y_max=1, increasing=False)

    def _KL_project(self, theta):
        raise NotImplementedError

    def _MAP(self, theta):
        n_classes = len(theta) + 1
        scores = np.zeros(n_classes)
        scores[0] = 0
        for i in range(1, n_classes):
            scores[i] = scores[i-1] + theta[i-1]
        # Returns number between 1 and n_classes.
        return np.argmax(scores) + 1

    # FIXME: move n_classes and neg_label to __init__?
    def _phi(self, y, n_classes, neg_label=0):
        ret = np.zeros(n_classes - 1)
        for i in range(1, n_classes):  # from 1 to n_classes-1
            if y > i:
                ret[i-1] = 1
            else:
                ret[i-1] = neg_label
        return ret

    def phi(self, Y, k, neg_label=0):
        return np.array([self._phi(y, k, neg_label) for y in Y])

    def _argmax(self, theta):
        n_classes = len(theta) + 1
        return self._phi(self._MAP(theta), n_classes)

    def vertices(self, size):  # size = len(theta) = n_classes - 1
        y = np.zeros(size)
        yield y
        for i in range(size):
            y = y.copy()
            y[i] = 1
            yield y
