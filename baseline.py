from functools import lru_cache
import warnings
import time

from sklearn.metrics import f1_score
import numpy as np

from problems import ProblemLoader
from feature_extractors import FeatureExtractor
from classifiers import SVMClassifier
from utils import hashabledict

warnings.simplefilter("ignore")


@lru_cache(maxsize=128)
def extract(extractor_cls, extractor_args, problem):
    extractor = extractor_cls(**extractor_args)
    extractor.fit(problem.train.X)
    train_data = extractor.transform(problem.train.X)
    test_data = extractor.transform(problem.test.X)

    return train_data, test_data


best = [
    (
        SVMClassifier,
        (
            FeatureExtractor,
            {'ngram_range': (2, 2), 'analyzer': 'word', 'use_scaler': True,
             'use_tfidf': False, 'tokenizer': None},
            'fex config 10',
        ),
        (
            FeatureExtractor,
            {'ngram_range': (2, 2), 'analyzer': 'word', 'use_scaler': False,
             'use_tfidf': True, 'tokenizer': 'pos'},
            'fex config 12',
        ),
    )
]


def baseline(path, outpath):
    problems = ProblemLoader(path)
    for clf_cls, fex_a, fex_b in best:
        scores = []
        for problem in problems.iter():
            data = []
            for fex, args, _ in [fex_a, fex_b]:
                data.append(
                    extract(fex, hashabledict(args), problem)
                )
            train_data, test_data = [np.concatenate(item, axis=1) for item in zip(*data)]
            clf = clf_cls()
            clf.fit(train_data, problem.train.y)
            predictions = clf.predict(test_data)
            scores.append(f1_score(predictions, problem.test.y, average='macro'))
        print('f1 = {:.3f}'.format(np.mean(scores)))
    print(extract.cache_info())
    print('Nothing more to run')


if __name__ == '__main__':
    start_time = time.time()
    baseline('./data/problems-pan18', './data/answers')
    print('elapsed time:', time.time() - start_time)
