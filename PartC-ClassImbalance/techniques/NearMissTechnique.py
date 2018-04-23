from Technique import Technique
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from imblearn.under_sampling import NearMiss

class NearMissTechnique(Technique, BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, version=1, ratio="auto", n_neighbours=3):
        self.version = version
        self.ratio = ratio
        self.n_neighbours = n_neighbours
        self._base_classifier = base_classifier

    def fit(self, train_x, train_y):
        near_miss = NearMiss(version=self.version,
                             ratio=self.ratio,
                             n_neighbors=self.n_neighbours)

        resampled_x, resampled_y = near_miss.fit_sample(train_x, train_y)

        self._base_classifier.fit(resampled_x, resampled_y)

        return self

    def predict(self, x):
        return self._base_classifier.predict(x)