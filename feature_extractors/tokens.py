import numpy as np
from nltk import word_tokenize


class TokensFeatureExtractor:
    """
     Basically the same as BOW, but it uses tokens not lemmas (so includes
     punctuation), and it doesn't scale the counts
    """

    def __init__(self, low_cut=0.25, high_cut=1):
        self.low_cut = low_cut
        self.high_cut = high_cut

    def _clean(self, doc):
        clean_doc = word_tokenize(doc)
        return [token.lower() for token in clean_doc]

    def fit(self, X_train):
        unique_words = {}
        for doc in X_train:
            for word in self._clean(doc):
                if word not in unique_words:
                    unique_words[word] = 1
                else:
                    unique_words[word] += 1
        low_cut = int(self.low_cut * len(unique_words))
        high_cut = int(self.high_cut * len(unique_words))
        self.words = sorted(unique_words, key=unique_words.get, reverse=False)[low_cut:high_cut]

    def transform(self, X_test):
        bows = []
        for doc in X_test:
            bow = np.zeros(len(self.words))
            for word in self._clean(doc):
                if word in self.words:
                    bow[self.words.index(word)] += 1
            bows.append(bow)
        return np.array(bows)
