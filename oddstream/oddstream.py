from oddstream.get_pca_space import get_pca_features
from oddstream.set_outlier_threshold import set_outlier_threshold

class OddStreams():
    def __init__(self, max_iterations, trials):
        self.max_iterations = max_iterations
        self.trials = trials


    def train(self, X):
        # train auto encoder

        # extract features
        train_features = 0

        # scale features

        # get pca
        pc = get_pca_features(train_features)

        # outlier threshold
        self.threshold = set_outlier_threshold(pc, trials=self.trials)


    def predict(self):


        return 0
