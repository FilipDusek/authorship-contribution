from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV


class BaselineClassifier:
    def __init__(self, pt=0.1):
        self.clf = OneVsRestClassifier(LinearSVC(C=1))
        self.pt = pt

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):
        predictions = self.clf.predict(X)

        return predictions
