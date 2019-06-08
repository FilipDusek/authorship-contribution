import time
import numpy as np
from feature_extractors import (
    PosFeatureExtractor, BaselineFeatureExtractor, PunctFeatureExtractor,
    FunctionWFeatureExtractor, BaseFeatureExtractor, BOWFeatureExtractor
)
from classifiers import BaselineClassifier as Classifier
from problems import ProblemLoader
from utils import save_answers, print_result_report, print_fex_report, hashabledict
from itertools import combinations, chain, product, islice
from functools import lru_cache

feature_extractors = [
    (PunctFeatureExtractor, {}, 'punctuation'),
    (FunctionWFeatureExtractor, {}, 'function words'),
    (BOWFeatureExtractor, {}, 'bag of words (custom)'),
    (
        BaseFeatureExtractor,
        {'use_tfidf': False, 'analyzer': 'word', 'lowercase': False},
        'bag of words (sklearn)'
    ),
    (
        BaseFeatureExtractor,
        {'ngram_range': (3,3), 'use_tfidf': False, 'analyzer': 'word', 'lowercase': False},
        'word 3-gram without tfidf'
    ),
    (
        BaseFeatureExtractor,
        {'ngram_range': (2,2), 'use_tfidf': False, 'analyzer': 'word', 'lowercase': False},
        'word 2-gram without tfidf'
    ),
    (
        BaseFeatureExtractor,
        {'ngram_range': (3, 3), 'use_tfidf': True, 'analyzer': 'char', 'lowercase': False},
        'char 3-gram with tfidf'
    ),
    (
        PosFeatureExtractor,
        {'ngram_range': (2, 2), 'use_tfidf': True},
        'part of speech tags 2-gram with tfidf'
    ),
]


@lru_cache(maxsize=128)
def extract(extractor_cls, extractor_args, problem):
    extractor = extractor_cls(**extractor_args)
    extractor.fit(problem.train.X)
    train_data = extractor.transform(problem.train.X)
    test_data = extractor.transform(problem.test.X)

    extractor_name = extractor_cls.__name__
    error_ndarray = '{} needs to return numpy array. '.format(extractor_name)
    assert isinstance(train_data, np.ndarray), error_ndarray
    assert isinstance(test_data, np.ndarray), error_ndarray

    error_dim = '{} needs to return two-dimensional numpy array. Returned {}.'
    assert len(test_data.shape) == 2, \
        error_dim.format(extractor_name, test_data.shape)
    assert len(train_data.shape) == 2, \
        error_dim.format(extractor_name, train_data.shape)

    assert train_data.shape[1] == test_data.shape[1], \
        '{} needs to return the same amount of features for both testing and' \
        ' training data. It returns {} for training and {} for testing.'.format(
        extractor_name, train_data.shape, test_data.shape
    )

    error_shape = 'First dimension of data returned by {} needs to have same' \
                  'length as the input data. The length is {}, but should be {}.'
    assert train_data.shape[0] == len(problem.train.X), \
        error_shape.format(extractor_name, train_data.shape[0], len(problem.train.X))
    assert test_data.shape[0] == len(problem.test.X), \
        error_shape.format(extractor_name, test_data.shape[0], len(problem.test.X))

    return train_data, test_data


def baseline(path, outpath):
    problems = ProblemLoader(path)
    problem_iterator = islice(problems.iter(), 1)
    fex_combinations = (combinations(feature_extractors, i)
                        for i in range(1, len(feature_extractors) + 1))
    fex_combinations = chain(*fex_combinations)

    for problem, fexs_selected in product(problem_iterator, fex_combinations):
        print('Processing {}...'.format(problem.problem_name))

        # Extract the features [(train, test), (train, test), ...]
        data = [extract(fex, hashabledict(args), problem) for fex, args, _ in fexs_selected]
        # Rotate from rows to clumns and concatenate the features
        train_data, test_data = [np.concatenate(item, axis=1) for item in zip(*data)]

        fexs, _, names = zip(*fexs_selected)
        print_fex_report(fexs, names, train_data)

        clf = Classifier(pt=0.1)
        clf.fit(train_data, problem.train.y)
        predictions = clf.predict(test_data)

        print_result_report(problem, predictions)
        save_answers(problem, predictions, path, outpath)
    print(extract.cache_info())


if __name__ == '__main__':
    start_time = time.time()
    baseline('./data/problems-pan18', './data/answers')
    print('elapsed time:', time.time() - start_time)
