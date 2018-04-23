import oddstream
from sklearn.decomposition import PCA
import numpy as np
from oddstream import set_outlier_threshold
from oddstream.kde_estimation import KDEEstimation2D
from oddstream.utils import mvrnorm
from fastkde import fastKDE

N = int(2e3)
var1 = 50*np.random.normal(size=N) + 0.1
var2 = 0.01*np.random.normal(size=N) - 300
v = np.array([var1, var2]).T

pca = PCA(n_components=2)
pca.fit(v)
pc_norm = pca.transform(v)
fastKDE.pdf_at_points(pc_norm[1:10, 0], pc_norm[1:10, 1])
kde = KDEEstimation2D(pc_norm.T)
kde.evaluate_points(pc_norm.T)

Sigma = np.cov(pc_norm)
mvrnorm(1, pc_norm[0, :],)