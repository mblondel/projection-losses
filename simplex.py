import numpy as np

# Code by Mathieu Blondel, 2018.
def project_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2

    z: float or array
        If array, len(z) must be compatible with V

    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return project_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return project_simplex(V, z, axis=1).ravel()


# Code by Andre Martins.
def constrained_softmax(z, u):
    z -= np.mean(z)
    q = np.exp(z)
    active = np.ones(len(u))
    mass = 0.
    p = np.zeros(len(z))
    while True:
        inds = active.nonzero()[0]
        p[inds] = q[inds] * (1. - mass) / sum(q[inds])
        found = False
        #import pdb; pdb.set_trace()
        for i in inds:
            if p[i] > u[i]:
                p[i] = u[i]
                mass += u[i]
                found = True
                active[i] = 0
        if not found:
            break
    #print mass
    #print active
    return p
