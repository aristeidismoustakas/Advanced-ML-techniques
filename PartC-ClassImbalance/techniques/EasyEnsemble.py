from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from imblearn.ensemble import EasyEnsemble

class EasyEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, n_estimators=10):
        super(EasyEnsemble, self).__init__()
        self._no_of_estimators = n_estimators
        self._estimators = []
        self._base_classifier = base_classifier

    def fit(self, train_x, train_y):
        self._estimators = []
        ee = EasyEnsemble(random_state=42, replacement=True, n_subsets=self._no_of_estimators)
        X_res, y_res = ee.fit_sample(train_x, train_y)

        for i in range(self._no_of_estimators):
            X, y = X_res[i,:,:], y_res[i,:,:]

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
