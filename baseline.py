import os
import glob
import json
import time
from feature_extractors import BaselineFeatureExtractor
from classifiers import BaselineClassifier
from problems import Problems


def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()
    problems = Problems(path)
    for problem, train_docs, test_docs, unk_folder, lang in problems.iter():
        x_train, y_train = zip(*train_docs)
        x_test, _ = zip(*test_docs)

        extractor = BaselineFeatureExtractor()
        extractor.fit(x_train)
        train_data = extractor.transform(x_train)
        test_data = extractor.transform(x_test)

        clf = BaselineClassifier(pt=pt)
        clf.fit(train_data, y_train)
        predictions = clf.predict(test_data)

        print('\t', len(set(y_train)), 'candidate authors')
        print('\t', len(x_train), 'known texts')
        print('\t', len(x_test), 'unknown texts')
        print('\t', (predictions == '<UNK>').sum(), 'texts left unattributed')

        # Saving output data
        out_data = []
        unk_filelist = glob.glob(os.path.join(path, problem, unk_folder, '*.txt'))
        pathlen = len(os.path.join(path, problem, unk_folder))
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen + 1:], 'predicted-author': v})
        with open(os.path.join(outpath, 'answers-' + problem + '.json'), 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file', 'answers-' + problem + '.json')
    print('elapsed time:', time.time() - start_time)


if __name__ == '__main__':
    # Parameters:
    # path: Path to the main folder of a collection of attribution problems
    # outpath: Path to an output folder
    # n: n-gram order (default=3)
    # ft: frequency threshold (default=5)
    # pt: probability threshold for the reject option (default=0.1)
    baseline('./data/problems', './data/answers', 3, 5, 0.1)
