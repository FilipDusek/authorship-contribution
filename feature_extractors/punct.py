import collections as ct
import numpy as np

class PunctFeatureExtractor:
    def __init__(self, marks=5, weighted=True):
        self.marks = marks
        self.weighted = weighted

    def fit(self, X):
        targetdict = {mark:count for mark, count in ct.Counter(X).items() if mark.isalnum() == False and mark != " "}
        self.topmarks = sorted(targetdict, key=targetdict.get, reverse=True)[:self.marks]

    def _transform_single(self, document):
        vector_counts = np.zeros(len(self.topmarks))
        for i in range(len(self.topmarks)):
            vector_counts[i]=document.count(self.topmarks[i])
            if self.weighted == True:
                vector_counts[i] = vector_counts[i]/len(document)
        return vector_counts

    def transform(self, X):
        return np.array([self._transform_single(document) for document in X])
