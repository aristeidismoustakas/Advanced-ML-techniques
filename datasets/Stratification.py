from sklearn.base import BaseEstimator, ClassifierMixin,clone
from costcla.sampling import undersampling


class Stratification(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, cost_matrix):
        super(Stratification, self).__init__()
        self._cost_matrix = cost_matrix
        self._base_classifier = base_classifier
        self._estimator = None

    def fit(self, train_x, train_y):
        self._estimator = clone(self._base_classifier)
        x,y = undersampling
        self._estimator = self._estimator.fit(x, y)
        return self

    def predict(self, x):
        prediction = self._estimator.predict(x)
        return prediction

    def undersampling = undersampling(self._x, self._y, cost_matrix, per=0.5)

