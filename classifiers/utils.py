from collections import defaultdict
import numpy as np

def to_clf(cls):
    """
        This function modifies sklearn cluster classes to map cluster centers to labels.
        Most frequent label corresponding to each cluster center is used as its label.
    """
    class ClusterClassifier(cls):
        def fit(self, X, y):
            self.n_clusters = len(set(y))
            super().fit(X, y)
            label_idx = zip(y, super().predict(X))
            idx_to_label = defaultdict(list)
            for label, index in label_idx:
                idx_to_label[index].append(label)

            self.idx_to_label = {key: max(set(vals), key=vals.count)
                                 for key, vals in idx_to_label.items()}
            return self

        def predict(self, X):
            result = super().predict(X)
            return np.array([self.idx_to_label[item] for item in result])

    return ClusterClassifier
