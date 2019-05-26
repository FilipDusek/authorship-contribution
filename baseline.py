import time
from feature_extractors import POSFeatureExtractor as FeatureExtractor
from classifiers import BaselineClassifier as Classifier
from problems import ProblemLoader
from utils import save_answers, print_report

def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()
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

    print('elapsed time:', time.time() - start_time)


if __name__ == '__main__':
    # Parameters:
    # path: Path to the main folder of a collection of attribution problems
    # outpath: Path to an output folder
    # n: n-gram order (default=3)
    # ft: frequency threshold (default=5)
    # pt: probability threshold for the reject option (default=0.1)
    baseline('./data/problems', './data/answers', 3, 5, 0.1)
