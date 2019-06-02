from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, defaultdict
import string

import nltk


class POSFeatureExtractor:
    def sliding_window(self, l, take=3):
        slices = (l[i:] for i in range(take))
        for items in zip(*slices):
            yield items

    def __init__(self, n=3, min_count=3):
        self.n = n
        self.min_count = min_count

    def _trans_tag(self, tag):
        if tag[0] not in string.ascii_uppercase:
            return None

        return tag[0]

    def _get_counts(self, X):
        tokens = nltk.word_tokenize(X)
        pos_tags = [tag for word, tag in nltk.pos_tag(tokens)]
        # pos_tags = [tag for tag in pos_tags if tag is not None]
        ngrams = [ngram for ngram in self.sliding_window(pos_tags, self.n)]
        counts = {k: v for k, v in Counter(ngrams).items()
                  if v > self.min_count}

        return counts

    def fit(self, X):
        counts = [self._get_counts(item) for item in X]
        self.vectorizer = DictVectorizer().fit(counts)

        return self

    def transform(self, X):
        counts = [self._get_counts(item) for item in X]

        return self.vectorizer.transform(counts).toarray()
