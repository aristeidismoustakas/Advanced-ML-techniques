from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE

class SMOTETechnique(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier):
        super(SMOTETechnique, self).__init__()
        self._base_classifier = base_classifier

    def fit(self, train_x, train_y):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_sample(train_x, train_y)
        self._base_classifier = self._base_classifier.fit(X_res, y_res)
        return self

    def predict(self, x):
        final_predictions = self._base_classifier.predict(x)
        return final_predictions

