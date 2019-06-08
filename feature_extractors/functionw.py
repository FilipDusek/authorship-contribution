import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk

class FunctionWFeatureExtractor:
    def __init__(self, percent_cutoff = 0.9):
        self.percent_cutoff = percent_cutoff
        self.lemmatizer = WordNetLemmatizer()
    def _clean(self, X):
        clean_doc = ""
        for doc in X:
            for char in doc:
                if char.isalnum() == True or char == " " or char == "''":
                    clean_doc = clean_doc + char
                if char.isalnum() == False:
                    clean_doc = clean_doc + " "
        clean_doc = clean_doc.split(" ")
        lemmatized_doc = [lemmatizer.lemmatize(token) for token in clean_doc if token != ""]
        return lemmatized_doc
        
    def fit(self, X):
        cleaned_X = self._clean(X)
        wordfreq = {}
        for i in cleaned_X:
            if i in wordfreq:
                wordfreq[i] += 1
            else:
                wordfreq[i] = 1
        sorted_wordfreq = sorted(wordfreq, key=wordfreq.get, reverse=True)
        accumulate = 0
        count_words = 0
        for word in sorted_wordfreq:
            if accumulate <= self.percent_cutoff:
                accumulate += (wordfreq[word]/len(cleaned_X))
                count_words += 1     
        self.topwords = sorted_wordfreq[1:count_words]
        
    def transform(self, X):
        transformed_docs = []
        for doc in range(len(X)):
            clean_doc = self._clean(X[doc])      
            funct_array = np.zeros(len(self.topwords))
            for word in range(len(self.topwords)):
                funct_array[word] = X[doc].count(self.topwords[word])
            normed_funct_array = np.zeros(len(self.topwords)) 
            for normed_word in range(len(normed_funct_array)):
                normed_funct_array[normed_word] = funct_array[normed_word]/max(funct_array)
            transformed_docs.append(normed_funct_array)
        return np.array(transformed_docs)