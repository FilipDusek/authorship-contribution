from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_

class RandomForestClassifier:
    def __init__(self, pt=0.2):
        self.clf = RandomForestClassifier_(n_estimators=100)
        self.pt = pt

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):
        predictions = self.clf.predict(X)
        proba = self.clf.predict_proba(X)

        # Reject option (used in open-set cases)
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if (sproba[0] - sproba[1]) < self.pt:
                predictions[i] = u'<UNK>'

        return predictions
