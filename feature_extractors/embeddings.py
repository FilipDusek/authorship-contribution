import numpy as np
import spacy

class WordEmbeddingsFeatureExtractor:
    """
        To use this feature extractor, run:
            `python -m spacy download en_core_web_lg`
    """
    def __init__ (self):
        self.embedder = spacy.load('en_core_web_lg')
    def fit(self, X):
        pass
    def transform(self, X):
        we = np.array([self.embedder(doc).vector for doc in X])
        return we
