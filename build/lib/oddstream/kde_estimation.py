from fastkde import fastKDE
import numpy as np

"""
    Fast 2D Kernel Density Estimation with simple point evaluation
"""
class KDEEstimation2D(object):
    def __init__(self, X):
        self.pdf, self.axes = fastKDE.pdf(X[:, 0], X[:, 1])

    def evaluate_points(self, X):
        m = X.shape[0]
        values = np.array(range(0, m), dtype=float)
        for i in range(0, m):
            values[i] = self.evaluate_pdf_value(X[i, :])
        return values

    def evaluate_pdf_value(self, s):
        x_up = s[0] <= self.axes[0]
        index_up_x = self.get_index_upper(x_up, 0)
        x_low = s[0] >= self.axes[0]
        index_low_x = self.get_index_lower(x_low)

        y_up = s[1] <= self.axes[1]
        index_up_y = self.get_index_upper(y_up, 1)
        y_low = s[1] >= self.axes[1]
        index_low_y = self.get_index_lower(y_low)

        # TODO
        value = 0.0
        for i in range(index_low_x, index_up_x + 1):
            for j in range(index_low_y, index_up_y + 1):
                value += self.pdf.T[i, j]
        value /= 4
        return value

    def get_index_upper(self, values, index):
        c = [i for i in range(0, len(values)) if values[i]]
        if len(c) == 0:
            up = self.pdf.shape[index] - 2
        else:
            up = np.min(c)
        return up

    def get_index_lower(self, values):
        c = [i for i in range(0, len(values)) if values[i]]
        if len(c) == 0:
            up = 0
        else:
            up = np.max(c)
        return up