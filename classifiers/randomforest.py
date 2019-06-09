from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_

class RandomForestClassifier:
    def __init__(self, pt=0.2):
        self.clf = RandomForestClassifier_(n_estimators=100)
        self.pt = pt

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):
        return self.clf.predict(X)
