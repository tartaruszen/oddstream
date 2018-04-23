import oddstream
from sklearn.decomposition import PCA
import numpy as np
from oddstream.set_outlier_threshold import set_outlier_threshold
from oddstream.kde_estimation import KDEEstimation2D
from oddstream.utils import mvrnorm
from fastkde import fastKDE
import fastkde

N = int(2e3)
var1 = 50*np.random.normal(size=N) + 0.1
var2 = 0.01*np.random.normal(size=N) - 300
v = np.array([var1, var2]).T

pca = PCA(n_components=2)
pca.fit(v)
pc_norm = pca.transform(v)
kde = KDEEstimation2D(pc_norm)
kde.evaluate_points(pc_norm)

import pylab as PP
PP.contor(kde.axes[0], kde.axes[1], kde.pdf)
PP.show()

PP.pcolor(kde.pdf)
PP.colorbar()
PP.show()

PP.pcolor(pc_norm)
PP.colorbar()
PP.show()

set_outlier_threshold(pc_norm, 0.5, 100)




# fastKDE.pdf_at_points(pc_norm[1:10, 0], pc_norm[1:10, 1])
kde = KDEEstimation2D(pc_norm)

s = pc_norm[0, :]
x_low = s[0] <= kde.axes[0]
index_up_x = kde.get_index_upper(x_low, 0)
x_up = s[0] >= kde.axes[0]
index_low_x = kde.get_index_lower(x_up)

x_low = s[1] <= kde.axes[1]
index_up_y = kde.get_index_upper(x_low, 1)
x_up = s[1] >= kde.axes[1]
index_low_y = kde.get_index_lower(x_up)

value = 0.0
for i in range(index_low_x, index_up_x + 1):
    for j in range(index_low_y, index_up_y + 1):
        value += kde.pdf.T[i, j]
value /= 4


kde.evaluate_points(pc_norm)

Sigma = np.cov(pc_norm)
mvrnorm(1, pc_norm[0, :],)


def evaluate_pdf_value(s):
    c1 = s[0] <= axes[0]
    c2 = s[0] >= axes[0]
    up_x = np.min([index for index in range(0, len(c1)) if c1[index]])
    low_x = np.max([index for index in range(0, len(c2)) if c2[index]])

    c1 = s[1] <= axes[1]
    c2 = s[1] >= axes[1]
    up_y = np.min([index for index in range(0, len(c1)) if c1[index]])
    low_y = np.max([index for index in range(0, len(c2)) if c2[index]])

    value = 0.0
    for i in range(low_x, up_x + 1):
        for j in range(low_y, up_y + 1):
            value += pdf[i, j]
    value /= 4
    return value