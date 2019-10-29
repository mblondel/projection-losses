# Author: Mathieu Blondel, 2019
# License: BSD

import numpy as np

from polytopes import UnitCube
from polytopes import ProbabilitySimplex
from polytopes import Knapsack
from polytopes import OrderSimplex
from polytopes import Permutahedron
from polytopes import Birkhoff
from polytopes import CartesianProduct

from fw import project_fw
from fista import KL_project_fista

from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal


def test_Euclidean_projection():
    rng = np.random.RandomState(0)

    for poly in (UnitCube(),
                ProbabilitySimplex(),
                Knapsack(max_labels=4),
                Knapsack(max_labels=4, min_labels=2),
                OrderSimplex(),
                Permutahedron(),
                Birkhoff(max_iter=1000, tol=1e-5),
                CartesianProduct(ProbabilitySimplex())):

        for func in (rng.randn, rng.rand,):
            if isinstance(poly, (Birkhoff, CartesianProduct)):
                theta = func(100) * 5
            else:
                theta = func(10) * 5

            sol = project_fw(theta, poly.argmax,
                             variant="pairwise",
                             max_iter=1000,
                             tol=1e-9)

            # Test correctness.
            sol_proj = poly.Euclidean_project(theta)
            error = np.mean((sol - sol_proj) ** 2)
            assert_less(error, 1e-5)

            # Test vectorization.
            sol_proj2 = poly.Euclidean_project([theta])
            assert_equal(len(sol_proj2), 1)
            assert_array_almost_equal(sol_proj2[0], sol_proj)


def test_KL_projection():
    rng = np.random.RandomState(0)

    for poly in (UnitCube(),
                 ProbabilitySimplex(),
                 Knapsack(max_labels=4),
                 Knapsack(max_labels=4, min_labels=2),
                 Birkhoff(max_iter=3000, tol=1e-5),
                 CartesianProduct(ProbabilitySimplex()),
                 #Permutahedron(),  # not implemented
                 #OrderSimplex(),  # not implemented
                ):

        for func in (rng.randn, rng.rand,):
            if isinstance(poly, (Birkhoff, CartesianProduct)):
                theta = func(100) * 5
            else:
                theta = func(10) * 5

            sol = project_fw(theta, poly.argmax,
                             #variant="pairwise",
                             projection_type="KL",
                             max_iter=1000,
                             tol=1e-9)

            # Test correctness.
            sol_proj = poly.KL_project(theta)
            error = np.mean((sol - sol_proj) ** 2)
            assert_less(error, 1e-3)

            # Test vectorization.
            sol_proj2 = poly.KL_project([theta])
            assert_equal(len(sol_proj2), 1)
            assert_array_almost_equal(sol_proj2[0], sol_proj)
