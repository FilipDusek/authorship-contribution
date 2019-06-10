from sklearn.naive_bayes import ComplementNB


class NaiveBayesClassifier:
    def __init__(self):
        self.clf = ComplementNB()

    def fit(self, X, y):
        self.clf.fit(X, y)

        return self

    def predict(self, X):

        return self.clf.predict(X)
