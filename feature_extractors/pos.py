import nltk
from .base import FeatureExtractor


class PosFeatureExtractor(FeatureExtractor):
    def tokenize(self, doc):
        tokens = nltk.word_tokenize(doc)
        _, tags = zip(*nltk.pos_tag(tokens))

        return tags
