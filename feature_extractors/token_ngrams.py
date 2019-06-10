import numpy as np
from nltk import word_tokenize
from collections import Counter
import re

class TokenNGramsFeatureExtractor:
    
    def __init__ (self, low_cut = 0.8, high_cut = 1, n=1, with_punct = True):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.n = n
        self.with_punct = with_punct
        
    def _clean(self, doc):
        if self.with_punct == True:
            tokens = word_tokenize(doc)
        else:
            tokens = list(filter(None, re.split(r'\W+', doc)))

        ngrams = [" ".join(tokens[i:i+(self.n)]) for i in range(0, len(tokens)-(self.n-1))]
        return ngrams
    
    def fit(self, X):
        counts = sum([Counter(self._clean(doc)) for doc in X], Counter())
        low_cut = int(self.low_cut*len(counts))
        high_cut = int(self.high_cut*len(counts))
        self.words = sorted(counts, key=counts.get, reverse=False)[low_cut:high_cut]
        
    def transform(self, X):
        bows = []
        for doc in X:
            bow = np.zeros(len(self.words))
            for word in self._clean(doc):
                if word in self.words:
                    bow[self.words.index(word)] += 1
            bows.append(bow)
        return np.array(bows)