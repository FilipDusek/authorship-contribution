import numpy as np
import re
from nltk.stem import WordNetLemmatizer


class BOWFeatureExtractor:

    def __init__(self, low_cut=0.95, high_cut=1):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.lemmatizer = WordNetLemmatizer()

    def _clean(self, doc):
        clean_doc = re.split(r'\W+', doc)
        lemmatized_doc = [self.lemmatizer.lemmatize(
            token.lower()) for token in clean_doc if token != ""]
        return lemmatized_doc

    def fit(self, X):
        unique_words = {}
        for doc in X:
            for word in self._clean(doc):
                if word not in unique_words:
                    unique_words[word] = 1
                else:
                    unique_words[word] += 1
        low_cut = int(self.low_cut * len(unique_words))
        high_cut = int(self.high_cut * len(unique_words))
        self.words = sorted(unique_words, key=unique_words.get, reverse=False)[low_cut:high_cut]

    def transform(self, X):
        bows = []
        for doc in X:
            bow = np.zeros(len(self.words))
            for word in self._clean(doc):
                if word in self.words:
                    bow[self.words.index(word)] += 1
            max_count = max(bow)
            if max_count != 0:
                for i in range(len(bow)):
                    bow[i] = bow[i] / max_count
            bows.append(bow)
        return np.array(bows)
