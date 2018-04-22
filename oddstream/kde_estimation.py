from fastkde import fastKDE

"""
    Fast 2D Kernel Density Estimation with point evaluation
"""
class KDEEstimation2D(object):
    def __init__(self, X):
        self.pdfObject = fastKDE.fastKDE(X, doSaveMarginals=False, doFFT = False, positiveShift = False, logAxes = False)
        self.pdfObject.applyBernacchiaFilter()

    def evaluate_points(self, X):
        return self.pdfObject.__transformphiSC_points__(X)
