# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np
from sklearn.utils.testing import assert_almost_equal

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
    for poly in (UnitCube(),
                 ProbabilitySimplex(),
                 Knapsack(max_labels=4),
                 Knapsack(min_labels=2, max_labels=4),
                 Permutahedron(),
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

                sol = np.dot(bf_search(poly, theta), theta)
                sol2 = np.dot(poly.argmax(theta), theta)
                assert_almost_equal(sol, sol2)
