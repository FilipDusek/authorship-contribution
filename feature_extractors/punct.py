import collections as ct
import numpy as np
from string import punctuation

class PunctFeatureExtractor:
    def __init__(self, marks=0, weighted=True):
        self.marks = marks
        self.weighted = weighted

    def fit(self, X):
        punctuation_counts = {}
        for i in X:
            counts = ct.Counter(i)
            new_punc = {k:v for k, v in counts.items() if k in punctuation}
            punctuation_counts = {key: punctuation_counts.get(key, 0) + new_punc.get(key, 0) for key in set(punctuation_counts) | set(new_punc)}
        if self.marks > 0:
            self.topmarks = sorted(punctuation_counts, key=punctuation_counts.get, reverse=True)[:self.marks]
        else:
            self.topmarks = sorted(punctuation_counts, key=punctuation_counts.get, reverse=True)
    def _transform_single(self, document):
        vector_counts = np.zeros(len(self.topmarks))
        for i in range(len(self.topmarks)):
            vector_counts[i] = document.count(self.topmarks[i])
        if self.weighted == True:
            n_vector_counts = np.zeros(len(self.topmarks))
            for punc in range(len(vector_counts)):
                n_vector_counts[punc] = vector_counts[punc]/max(vector_counts)
        return n_vector_counts
    
    def transform(self, X):
        return np.array([self._transform_single(document) for document in X])