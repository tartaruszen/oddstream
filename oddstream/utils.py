import numpy as np
from scipy.stats.distributions import norm

"""
Simulate from a Multivariate Normal Distribution

Produces one or more samples from the specified multivariate normal distribution.

B. D. Ripley (1987) Stochastic Simulation. Wiley. Page 98.

Borrowed from R: mvrnorm {MASS}

TODO: numba
"""
def mvrnorm(n, mu, Sigma, tol = 1e-6):
    p = len(mu)
    # TODO: add exception: (p, p) == Sigma.shape
    eigValues, eigVectors = np.linalg.eig(Sigma)
    # TODO: check positive definite: eigen values > tol * abs(eigV[0])
    X = np.array(norm.rvs(size = n * p)).reshape(n, p)
    eigValues[eigValues < 0] = 0
    eigValues = np.sqrt(eigValues)
    X = mu + np.dot(np.dot(eigVectors, np.diag(eigValues)), X.T)
    return X.T
