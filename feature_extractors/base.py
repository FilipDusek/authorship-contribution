from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import numbers
import nltk


class TfLimitMixin:
    def __init__(self, tf_lowcut=0.0, tf_highcut=1.0, tokenizer=None, *args, **kwargs):
        self.tf_lowcut = tf_lowcut
        self.tf_highcut = tf_highcut
        super().__init__(*args, **kwargs)
        self.tokenizer = nltk.word_tokenize if tokenizer is None else tokenizer

    def _limit_features(self, X, vocabulary, *args, **kwargs):
        X_limited, removed_terms = super()._limit_features(X, vocabulary, *args, **kwargs)

        tfs = np.asarray(X.sum(axis=0)).ravel()
        n_term = len(tfs)
        low_term_count = int(self.tf_lowcut
                             if isinstance(self.tf_lowcut, numbers.Integral)
                             else self.tf_lowcut * n_term)
        high_term_count = int(self.tf_highcut
                              if isinstance(self.tf_highcut, numbers.Integral)
                              else self.tf_highcut * n_term)

        mask_inds = (tfs).argsort()[low_term_count:high_term_count]
        mask = np.zeros(len(tfs), dtype=bool)
        mask[mask_inds] = True

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]

        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")

        return X_limited[:, kept_indices], removed_terms


class CountVectorizerTfLimit(TfLimitMixin, CountVectorizer):
    pass


class TfidfVectorizerTfLimit(TfLimitMixin, TfidfVectorizer):
    pass


class FeatureExtractor:
    def __init__(self, use_tfidf=True, use_scaler=True, tokenizer=None, *args, **kwargs):
        vectorizer_cls = TfidfVectorizerTfLimit if use_tfidf else CountVectorizerTfLimit
        tokenizer_fn = getattr(self, 'tokenize', None)
        if isinstance(tokenizer, str) and tokenizer.lower() == 'pos':
            tokenizer_fn = self._pos_tokenize

        self.vectorizer = vectorizer_cls(tokenizer=tokenizer_fn, *args, **kwargs)
        self.use_scaler = use_scaler
        # inheritence would be cleaner, but how to handle use_tfidf arg?
        self.inverse_transform = self.vectorizer.inverse_transform

    def _pos_tokenize(self, doc):
        tokens = nltk.word_tokenize(doc)
        _, tags = zip(*nltk.pos_tag(tokens))

        return tags

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
