from sklearn.decomposition import PCA

"""
    Calculates the first two PCA components
"""
def get_pca_features(X):
    pca = PCA(n_components = 2)
    pc_comp = pca.fit_transform(X)
    return pc_comp