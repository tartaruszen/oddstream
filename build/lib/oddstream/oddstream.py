from oddstream.set_outlier_threshold import set_outlier_threshold
from oddstream.keras_autoencoder import KerasAutoEncoder

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

"""
    Detecting outliers in a collection of streaming time series.
"""
class OddStreams():
    def __init__(self, max_iterations, trials, config, concept_drift = False, callbacks = [], useCUDNN = False):
        self.max_iterations = max_iterations
        self.trials = trials
        self.concept_drift = concept_drift
        self.pca = PCA(n_components=2)
        self.autoencoder = KerasAutoEncoder(config = config, callbacks = callbacks, useCUDNN = useCUDNN)


    def train(self, X):
        self.autoencoder.train(X)
        train_features = self.autoencoder.extract_features(X)
        train_features = scale(train_features, axis = 1)
        self.pca.fit(train_features)
        pc_norm = self.pca.transform(train_features)
        self.kde_object = set_outlier_threshold(pc_norm, trials=self.trials)


    def predict(self, X):
        window_features = self.autoencoder.extract_features(X)
        # TODO post-process window_features
        window_features = scale(window_features, axis=1)
        pc_window = self.pca.transform(window_features)
        window_fhat = self.kde_object['fhat'].evaluate_points(pc_window)
        outliers = [i for i in range(0, len(window_fhat)) if window_fhat[i] < self.kde_object['threshold_fnx']]
        return outliers
