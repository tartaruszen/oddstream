import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


# too slow
def kde_bandwidth(x):
    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv = 10)
    grid.fit(x[:, None])
    return grid.best_params_

def kde_estimation(x):
    n = x.shape[0]
    d = x.shape[1]
    bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
    kde = KernelDensity(bandwidth=bw, metric='euclidean',
                        kernel='gaussian', algorithm='ball_tree')
    return kde.fit(x)


def set_outlier_threshold(pc_norm, p_rate, trials):

    return 0.0
