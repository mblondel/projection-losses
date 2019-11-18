# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal

from polytopes import UnitCube
from polytopes import ProbabilitySimplex
from polytopes import Knapsack
from polytopes import Permutahedron
from polytopes import Birkhoff
from polytopes import CartesianProduct
from polytopes import OrderSimplex


def bf_search(poly, theta):
    """Brute-force search"""
    best = None
    best_score = -np.inf
    for y in poly.vertices(theta.shape[0]):
        score = np.sum(y * theta)
        if score > best_score:
            best_score = score
            best = y
    return best


def test_linear_oracle():
    w = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    w2 = np.array([0.4, 0.1, 0.2, 0.3, 0.5])

    for poly in (UnitCube(),
                 ProbabilitySimplex(),
                 Knapsack(max_labels=4),
                 Knapsack(min_labels=2, max_labels=4),
                 Permutahedron(),
                 Permutahedron(w),
                 Permutahedron(w, w_sorted=True),
                 Permutahedron(w2),
                 Birkhoff(),
                 CartesianProduct(ProbabilitySimplex()),
                 OrderSimplex(),
                ):

        for seed in range(10):
            rng = np.random.RandomState(seed)

            for func in (rng.randn, rng.rand):
                if isinstance(poly, (Birkhoff, CartesianProduct)):
                    theta = func(25)
                else:
                    theta = func(5)

                # Test correctness.
                sol = np.dot(bf_search(poly, theta), theta)
                sol2 = np.dot(poly.argmax(theta), theta)
                assert_almost_equal(sol, sol2)

                # Test vectorization.
                sol3 = poly.argmax([theta, theta])
                assert_equal(len(sol3), 2)
                assert_almost_equal(sol3[0].dot(theta), sol2)
                assert_almost_equal(sol3[1].dot(theta), sol2)
