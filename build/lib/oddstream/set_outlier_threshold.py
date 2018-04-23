import numpy as np
import math

from oddstream.kde_estimation import KDEEstimation2D
from oddstream.utils import mvrnorm


def set_outlier_threshold(pc_norm, p_rate, trials):
    fhat = KDEEstimation2D(pc_norm)
    m = pc_norm.shape[0]

    f_extreme = [0 for _ in range(0, m)]
    for t in range(0, trials):
        sample = np.random.choice(range(0, m), m, replace=True)
        # TODO: not nice, but should work
        Sigma = np.cov(pc_norm.T)
        f_extreme[t] = np.min([fhat.evaluate_points(mvrnorm(n=1, mu=pc_norm[s, :], Sigma=Sigma)) for s in sample])

    k = 1 / (2 * math.pi)
    # TODO: why np.sqrt(-2 * np.log(f_val) - 2 * np.log(2 * math.pi))?
    psi_trans = [np.sqrt(-2 * np.log(f_val) - 2 * np.log(2 * math.pi)) if f_val < k else 0.0 for f_val in f_extreme]

    p = np.count_nonzero(psi_trans) / trials
    y = - np.log(-1 * np.log(1 - p_rate * p))
    cm = np.sqrt(2 * np.log(m)) - ((np.log(np.log(m)) + np.log(4 * math.pi)) / (2 * np.sqrt(2 * np.log(m))))
    dm = 1 / (np.sqrt(2 * np.log(m)))
    t = cm + y * dm
    threshold_fnx = np.exp(-1 * ((t**2) + 2 * np.log(2 * math.pi)) / 2)
    return {'threshold_fnx': threshold_fnx, 'fhat': fhat, 'Sigma': sigma}

