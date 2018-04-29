class Base(object):

    def __init__(self, model, *args, **kwargs):
        self._base_classifier = model

    def fit(self, train_x, train_y):
        return self._base_classifier.fit(train_x, train_y)

    def predict(self, x):
        return self._base_classifier.predict(x)