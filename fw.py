# Author: Mathieu Blondel
# License: BSD

from collections import defaultdict

import numpy as np
import joblib


# This is needed to use NumPy arrays as dictionary keys.
class atom_container(object):

    def __init__(self, atom):
        self.atom = atom

    def __hash__(self):
        return int(joblib.hash(self.atom), base=16)
        #return hash(self.atom.tobytes())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get(self):
        return self.atom


class ProjectionType(object):

    def line_search(self, u, theta, delta):
        gamma = 0

        for it in range(10):
            dp = self.grad(u + gamma * delta, theta).dot(delta)
            dpp = self.Hessian_vec(u, theta, delta).dot(delta)

            if gamma == 0:
                pg = min(dp, 0)
            elif gamma == 1:
                pg = max(dp, 0)
            else:
                pg = dp

            if abs(pg) < 1e-5:
                #print("Converged on iter=", it + 1)
                break

            gamma = gamma - dp / dpp
            gamma = min(1, max(0, gamma))

        return gamma


class Euclidean(ProjectionType):

    def obj(self, u, theta):
        return np.sum((u - theta) ** 2)

    def grad(self, u, theta):
        return u - theta

    def Hessian_vec(self, u, theta, delta):
        return delta

    def line_search(self, u, theta, delta):
        denom = np.sum(delta ** 2)
        if denom == 0:
            return 0

        neg_grad = -self.grad(u, theta)
        dgap = np.sum(delta * neg_grad)
        gamma = dgap / denom
        gamma = min(max(gamma, 0), 1.0)

        return gamma


EPS = 1e-10


class KL(ProjectionType):

    def obj(self, u, theta):
        v = np.exp(theta - 1)
        mask = u > EPS
        # higher is better
        ret = np.dot(u[mask], np.log(u[mask] / v[mask]))
        ret -= np.sum(u[mask])
        ret += np.sum(v)
        return ret

    def grad(self, u, theta):
        # gradient of <u, log u> - <u, theta>
        g = np.zeros(len(u), dtype=np.float64)
        mask = u > EPS
        g[mask] = np.log(u[mask]) + 1 - theta[mask]
        u[u <= 0] = 0
        u_eps = u[~mask] + EPS
        # Near u = 0, we use the exact derivative for <theta, u>
        # and a finite difference for <u, log u>.
        g[~mask] = (u_eps * np.log(u_eps)) / EPS - theta[~mask]
        return g

    def Hessian_vec(self, u, theta, delta):
        hv = np.zeros(len(u), dtype=np.float64)
        mask = u > EPS
        hv[mask] = 1. / u[mask] * delta[mask]
        # Near u = 0, we use a second-order forward difference.
        u_eps = u[~mask] + EPS
        u_2eps = u[~mask] + 2 * EPS
        hv[~mask] = (u_2eps * np.log(u_2eps) - 2 * u_eps * np.log(u_eps))
        hv[~mask] *= delta[~mask]
        hv[~mask] /= (EPS ** 2)
        return hv


def project_fw(theta, argmax_oracle, variant="vanilla",
               projection_type="Euclidean", line_search=True,
               init=None, max_iter=1000, tol=1e-6, ret_obj=False, verbose=0):
    """Compute Euclidean or KL projection using (pairwise) FW."""

    # Initialization.
    if init is None:
        u = argmax_oracle(theta)
    else:
        u = init.copy()

    # Initialize active set.
    active_set = defaultdict(float)
    active_set[atom_container(u)] = 1.0

    if projection_type == "Euclidean":
        proj = Euclidean()
    elif projection_type == "KL":
        proj = KL()
    else:
        raise ValueError("Invalid projection_type.")

    if ret_obj:
        obj_values = [proj.obj(u, theta)]

    for it in range(max_iter):
        neg_grad = -proj.grad(u, theta)

        # Forward direction.
        s = argmax_oracle(neg_grad)  # best atom
        delta = s - u

        # Away direction.
        if variant == "pairwise":
            best_score = np.inf
            v = None
            #for ac, p in active_set.items():
            for ac in active_set:
                score = np.sum(neg_grad * ac.get())
                if score < best_score:
                    best_score = score
                    v = ac.get()  # worst atom

            #delta_away = u - v

        # Duality gap.
        dgap = np.sum(delta * neg_grad)
        if dgap < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            break

        if variant == "pairwise":
            #delta = delta + delta_away = s - u + u - v
            delta = s - v
            gamma_max = active_set[atom_container(v)]

        elif variant == "vanilla":
            # Vanilla FW.
            gamma_max = 1
        else:
            raise ValueError("Unknown variant")

        # Step size
        if line_search:
            gamma = proj.line_search(u, theta, delta)
            if gamma == 0:
                break
            gamma = min(gamma, gamma_max)
        else:
            gamma = 2. / (it + 2)

        # Update convex combination.
        if variant == "pairwise":
            active_set[atom_container(v)] -= gamma
            active_set[atom_container(s)] += gamma

        # Update iterate.
        u = u + gamma * delta

        # Objective value
        if ret_obj:
            obj_values.append(proj.obj(u, theta))

        # Clean up zeros to speed up away-step searches.
        zeros = [ac for ac, p in active_set.items() if p == 0]
        for ac in zeros:
            active_set.pop(ac)

        # Sanity checks.
        assert all(p > 0 for p in active_set.values())
        assert np.abs(1 - sum(active_set.values())) <= 1e-6

    if ret_obj:
        return u, np.array(obj_values)
    else:
        return u


if __name__ == '__main__':
    import matplotlib.pylab as plt
    from polytopes import Knapsack

    rng = np.random.RandomState(0)
    theta = rng.randn(1000)
    projection_type = "KL"
    #projection_type = "Euclidean"

    poly = Knapsack(max_labels=20)

    sol_fw, obj_fw = project_fw(theta, poly._argmax, variant="vanilla",
                                projection_type=projection_type, ret_obj=True)
    sol_pw, obj_pw = project_fw(theta, poly._argmax, variant="pairwise",
                                projection_type=projection_type, ret_obj=True)
    obj_fw -= obj_pw.min()
    obj_pw -= obj_pw.min()

    plt.figure()
    plt.plot(np.arange(len(obj_fw)) + 1, obj_fw, label="FW")
    plt.plot(np.arange(len(obj_pw)) + 1, obj_pw, label="PW", ls="--")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.show()
