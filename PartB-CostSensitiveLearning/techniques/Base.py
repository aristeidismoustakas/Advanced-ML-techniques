from .Technique import Technique

class Base(Technique):

    def __init__(self, model, cost_matrix):
        self._base_classifier = model

    def fit(self, train_x, train_y):
        return self._base_classifier.fit(train_x, train_y)

    def predict(self, x):
        return self._base_classifier.predict(x)