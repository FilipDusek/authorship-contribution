from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from collections import defaultdict


class BaselineFeatureExtractor():
    def __init__(self, n=3, min_f=5):
        self.n = n
        self.min_f = min_f

    def fit(self, X):
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(self.n, self.n),
            lowercase=False,
            vocabulary=self._extract_vocabulary(X)
        )

        # This could be optimized. Documents are vectorized but the result is
        # used only to initialize MaxAbsScaler.

        vectorized = self.vectorizer.fit_transform(X).astype(float)
        for i, doc in enumerate(X):
            vectorized[i] = vectorized[i] / len(X[i])

        self.max_abs_scaler = preprocessing.MaxAbsScaler()
        self.max_abs_scaler.fit(vectorized)

        return self

    def transform(self, X):
        vectorized = self.vectorizer.transform(X).astype(float)

        for i, doc in enumerate(X):
            vectorized[i] = vectorized[i] / len(X[i])

        return self.max_abs_scaler.transform(vectorized).toarray()

    def _extract_vocabulary(self, texts):
        """
        Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
        """
        occurrences = defaultdict(int)
        for text in texts:
            text_occurrences = self._represent_text(text)
            for ngram in text_occurrences:
                if ngram in occurrences:
                    occurrences[ngram] += text_occurrences[ngram]
                else:
                    occurrences[ngram] = text_occurrences[ngram]
        vocabulary = []
        for i in occurrences.keys():
            if occurrences[i] >= self.min_f:
                vocabulary.append(i)
        return vocabulary

    def _represent_text(self, text):
        """
        Extracts all character 'n'-grams from  a 'text'
        """
        if self.n > 0:
            tokens = [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
        frequency = defaultdict(int)
        for token in tokens:
            frequency[token] += 1
        return frequency
