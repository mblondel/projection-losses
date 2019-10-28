"""
Efficient implementation of FISTA.
"""

# Author: Mathieu Blondel, 2017, 2019
# License: BSD 3 clause

import numpy as np


def fista(sfunc, nsfunc, x0, args=None, max_iter=500, max_linesearch=20,
          eta=2.0, tol=1e-3, verbose=0):

    if args is None:
        args = []

    y = x0.copy()
    x = y
    L = 1.0
    t = 1.0

    for it in range(max_iter):
        f_old, grad = sfunc(y, args=args, ret_grad=True)

        for ls in range(max_linesearch):
            y_proj, g = nsfunc(y - grad / L, L)  # TODO: modify nsfuncs to not compute g always
            diff = (y_proj - y).ravel()
            sqdist = np.dot(diff, diff)
            dist = np.sqrt(sqdist)

            f = sfunc(y_proj, args=args, ret_grad=False)

            F = f
            Q = f_old + np.dot(diff, grad.ravel()) + 0.5 * L * sqdist

            if F <= Q:
                break

            L *= eta

        if ls == max_linesearch - 1 and verbose:
            print("Line search did not converge.")

        if verbose:
            print("%d. %f" % (it + 1, dist))

        if dist <= tol:
            if verbose:
                print("Converged.")
            break

        x_next = y_proj
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        y = x_next + (t-1) / t_next * (x_next - x)
        t = t_next
        x = x_next

    return y_proj


def KL_project_fista(theta, proj_oracle, init=None, max_iter=500, tol=1e-3,
                     verbose=0):

    def sfunc(u, args=None, ret_grad=False):
        # maximimize <theta, u> - <u, log u>

        eps = 1e-9
        mask = u > eps
        obj = np.dot(theta[mask], u[mask]) - np.dot(u[mask], np.log(u[mask]))

        if not ret_grad:
            return -obj

        g = np.zeros_like(theta)
        g[mask] = theta[mask] - np.log(u[mask]) - 1
        u_eps = u[~mask] + eps
        # Near u = 0, we use the exact derivative for <theta, u>
        # and a finite difference for <u, log u>.
        g[~mask] = theta[~mask] - (u_eps * np.log(u_eps)) / eps

        return -obj, -g

    def nsfunc(u, L):
        return proj_oracle(u), 0

    init = np.ones_like(theta) if init is None else init.copy()

    return fista(sfunc, nsfunc, init, verbose=verbose,
                 max_iter=max_iter, tol=tol)
