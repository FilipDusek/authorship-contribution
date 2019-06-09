from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_

class RandomForestClassifier:
    def __init__(self):
        self.clf = RandomForestClassifier_(n_estimators=100)

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):
        return self.clf.predict(X)
