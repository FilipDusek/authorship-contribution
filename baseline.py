import time
import numpy as np
from feature_extractors import POSFeatureExtractor, BaselineFeatureExtractor
from classifiers import BaselineClassifier as Classifier
from problems import ProblemLoader
from utils import save_answers, print_result_report, print_fex_report
from itertools import combinations, chain, product, islice

feature_extractors = [
    BaselineFeatureExtractor,
    POSFeatureExtractor
]


def extract(extractor_cls, problem):
    extractor = extractor_cls()
    extractor.fit(problem.train.X)
    train_data = extractor.transform(problem.train.X)
    test_data = extractor.transform(problem.test.X)

    error = '{} needs to return numpy array'.format(extractor_cls.__name__)
    assert isinstance(train_data, np.ndarray), error
    assert isinstance(test_data, np.ndarray), error

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
        data = [extract(fex, problem) for fex in fexs_selected]
        # Rotate from rows to clumns and concatenate the features
        train_data, test_data = [np.concatenate(item, axis=1) for item in zip(*data)]

        print_fex_report(fexs_selected, train_data)

        clf = Classifier(pt=0.1)
        clf.fit(train_data, problem.train.y)
        predictions = clf.predict(test_data)

        print_result_report(problem, predictions)
        save_answers(problem, predictions, path, outpath)

if __name__ == '__main__':
    start_time = time.time()
    baseline('./data/problems', './data/answers')
    print('elapsed time:', time.time() - start_time)
