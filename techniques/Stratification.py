from sklearn.base import BaseEstimator, ClassifierMixin,clone
import numpy as np

class Stratification(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, cost_matrix):
        super(Stratification, self).__init__()
        self._cost_matrix = cost_matrix
        self._base_classifier = base_classifier
        self._estimator = None

    def fit(self, train_x, train_y):
        self._estimator = clone(self._base_classifier)
        x,y = self._my_undersampling(train_x, train_y)
        self._estimator = self._estimator.fit(x, y)
        return self

    def predict(self, x):
        prediction = self._estimator.predict(x)
        return prediction

    def _my_undersampling(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self._cost_matrix[0][1] < self._cost_matrix[1][0]:
            class_to_undersampling = 0
            stable_class = 1
            ratio = self._cost_matrix[0][1] /self._cost_matrix[1][0]
        else:
            class_to_undersampling = 1
            stable_class = 0
            ratio = self._cost_matrix[1][0] /self._cost_matrix[0][1]

        indices_to_undersampling = np.where(y == class_to_undersampling)[0]
        stable_indices = np.where(y == stable_class)[0]
        sel_under_indices = indices_to_undersampling[np.random.randint(len(indices_to_undersampling), size=int(ratio * len(stable_indices)))]
        final_indices = np.concatenate((stable_indices, sel_under_indices))
        X_new = X[final_indices, :]
        y_new = y[final_indices]
        return X_new,y_new