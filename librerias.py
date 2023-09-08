import logging
from networkx import is_distance_regular
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator


import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the explained variance and the principal components
        self.explained_variance_ = eigenvalues
        self.components_ = eigenvectors.T

        if self.n_components is not None:
            self.explained_variance_ = eigenvalues[:self.n_components]
            self.components_ = eigenvectors.T[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X):
        # Compute SVD
        U, S, V = np.linalg.svd(X)
        self.components_ = V[:self.n_components]

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



