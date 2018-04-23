from Technique import Technique
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

class CSRoulette(Technique, BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, cost_matrix, n_estimators=10):
        super(CSRoulette, self).__init__()
        self._cost_matrix = cost_matrix
        self._no_of_estimators = n_estimators
        self._estimators = []
        self._base_classifier = base_classifier

    def fit(self, train_x, train_y):
        self._estimators = []

        for i in range(self._no_of_estimators):
            x, y = zip(*self._sample_roulette(train_x, train_y))

            estimator = clone(self._base_classifier)
            estimator.fit(x, y)

            self._estimators.append(estimator)

        return self

    def predict(self, x):
        predictions = np.zeros(len(x))

        for i in range(self._no_of_estimators):
            predictions += self._estimators[i].predict(x)

        predictions = predictions / self._no_of_estimators

        return [self._prob_to_label(pred) for pred in predictions]

    def _prob_to_label(self, val):
        if val > 0.5:
            return 1
        else:
            return 0

    def _sample_roulette(self, x, y, sample_size=None):
        if sample_size == None:
            sample_size = len(x)

        val_list = []

        for x_val, y_val in zip(x, y):
            misclass_cost = 0
            if y_val == 1:
                misclass_cost = self._cost_matrix[1][0]
            else:
                misclass_cost = self._cost_matrix[0][1]

            for i in range(misclass_cost):
                val_list.append((x_val, y_val))

        indexes = np.random.choice(len(val_list), size=sample_size, replace=True)
        return [val_list[i] for i in indexes]