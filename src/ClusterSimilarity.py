from numpy.matlib import rand
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

# BaseEstimator: daje metody get_params() i set_params() dzieki czemu moge przeszukiwac hiperparametry
# TransformerMixin: na podstawie fit() i transform() sam tworzy metode fit_transform()

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters = 10, gamma = 1.0, random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    # FIT: obserwuje dane i zapisuje n_clusters skupisk:
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, y, sample_weight=sample_weight)
        return self

    # TRANSFORM: bierze dane i robi już coś z nimi (tutaj używa jądra RBF do obliczania odleglosci od tych punktow skupisk)
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    # Pobieram nazwy skupisk
    def get_feature_names_out(self, names=None):
        return [f"Prawdopodobienstwo {i} skupienia" for i in range (len(self.kmeans_.cluster_centers_))]