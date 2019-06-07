from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler


class BaseFeatureExtractor:
    def __init__(self, use_tfidf=True, use_scaler=True, *args, **kwargs):
        vectorizer_cls = TfidfVectorizer if use_tfidf else CountVectorizer
        tokenizer_fn = getattr(self, 'tokenize', None)
        self.vectorizer = vectorizer_cls(tokenizer=tokenizer_fn, *args, **kwargs)

        self.use_scaler = use_scaler
        # inheritence would be cleaner, but how to handle use_tfidf arg?
        self.inverse_transform = self.vectorizer.inverse_transform

    def fit(self, X):
        vectorized = self.vectorizer.fit_transform(X)
        if self.use_scaler:
            self.scaler = MaxAbsScaler()
            self.scaler.fit(vectorized)

        return self

    def transform(self, X):
        vectorized = self.vectorizer.transform(X).toarray()
        if self.use_scaler:
            vectorized = self.scaler.transform(vectorized)

        return vectorized
