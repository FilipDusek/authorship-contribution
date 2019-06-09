from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV


class BaselineClassifier:
    def __init__(self):
        self.clf = OneVsRestClassifier(LinearSVC(C=1))

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):

        return self.clf.predict(X)
