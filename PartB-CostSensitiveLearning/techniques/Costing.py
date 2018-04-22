from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
import random


class Costing(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, cost_matrix, n_estimators=10):
        super(Costing, self).__init__()
        self._cost_matrix = cost_matrix
        self._no_of_estimators = n_estimators
        self._estimators = []
        self._base_classifier = base_classifier

    def fit(self, train_x, train_y):
        self._estimators = []

        for i in range(self._no_of_estimators):
            X, y = self._costing_sample(train_x,train_y)

            estimator = clone(self._base_classifier)
            estimator.fit(X, y)

            self._estimators.append(estimator)

        return self

    def predict(self, x):
        final_predictions = np.zeros(len(x))
        predictions = np.zeros((len(x), self._no_of_estimators))

        for i in range(self._no_of_estimators):
            predictions[:, i] = self._estimators[i].predict(x)
        count_ones = np.count_nonzero(predictions == 1, axis=1)
        for i in range(len(x)):
            if count_ones[i] > self._no_of_estimators/2:
                final_predictions[i] = 1
            else:
                final_predictions[i] = 0

        return final_predictions

    def _costing_sample(self, X, y, C=None):
        X = np.asarray(X)
        y = np.asarray(y)
        selected = []

        if C == None:
            C = np.max(np.asarray(self._cost_matrix))

        for i in range(len(X)):
            if y[i] == 0:
                current_cost = self._cost_matrix[0][1]
            else:
                current_cost = self._cost_matrix[1][0]

            random_num = random.random()
            if random_num <= float(current_cost)/C:
                selected.append(i)

        return X[selected, :], y[selected]
