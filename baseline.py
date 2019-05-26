import time
from feature_extractors import POSFeatureExtractor as FeatureExtractor
from classifiers import BaselineClassifier as Classifier
from problems import ProblemLoader
from utils import save_answers, print_report

def baseline(path, outpath):
    problems = ProblemLoader(path)
    for problem in problems.iter():
        print('Processing {}...'.format(problem.problem_name))

        extractor = FeatureExtractor()
        extractor.fit(problem.train.X)
        train_data = extractor.transform(problem.train.X)
        test_data = extractor.transform(problem.test.X)

        clf = Classifier(pt=pt)
        clf.fit(train_data, problem.train.y)
        predictions = clf.predict(test_data)

        print_report(problem, predictions)
        save_answers(problem, predictions, path, outpath)

if __name__ == '__main__':
    start_time = time.time()
    baseline('./data/problems', './data/answers')
    print('elapsed time:', time.time() - start_time)
