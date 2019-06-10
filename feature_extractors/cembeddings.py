import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize

class CustomEmbeddingsFeatureExtractor:
    def __init__(self):
        pass
    
    def fit(self, X):
        words = [[word_tokenize(sent) for sent in sent_tokenize(doc)]for doc in X]
        words = [sentence for doc in words for sentence in doc]
        self.model = Word2Vec(words, min_count=1, size=300, workers = 2, window = 5, iter=1000)
        
    def transform(self, X):
        mean_wvs = []
        for doc in X:
            filtered_doc = filter(lambda word: word in self.model.wv.vocab, word_tokenize(doc))
            wvs = [self.model[word] for word in filtered_doc]
            mean_wvs.append(np.mean(np.array(wvs), axis=0))
        return np.array(mean_wvs)    
            