from sklearn.decomposition import PCA

def get_pca_features(X):
    pca = PCA(n_components = 2)
    pc_comp = pca.fit_transform(X)
    return pc_comp