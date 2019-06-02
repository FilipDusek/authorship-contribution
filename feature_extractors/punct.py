import collections as ct

class PunctFeatureExtractor:
    def __init__(self, marks=5, weighted=True):
        self.marks = marks
        self.weighted = weighted
    def _target(self, X):
        targetdict = {mark:count for mark, count in ct.Counter(X).items() if mark.isalnum() == False and mark != " "}
        self.topmarks = sorted(targetdict, key=targetdict.get, reverse=True)[:self.marks]
    def _extract(self, X):
        vector_counts = np.zeros(len(self.topmarks))
        for i in range(len(self.topmarks)):
            vector_counts[i]=X.count(self.topmarks[i])
            if self.weighted == True:
                vector_counts[i] = vector_counts[i]/len(X)
        return vector_counts