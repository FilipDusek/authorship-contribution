import os
import glob
import json
import time
import codecs
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from feature_extractors import BaselineFeatureExtractor


def read_files(path, label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(os.path.join(path, label, '*.txt'))
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        f.close()
    return texts


def load_problem_data(path, problem):
    probpath = os.path.join(path, problem)
    path = os.path.join(probpath, 'problem-info.json')
    with open(path, 'r') as f:
        fj = json.load(f)

    unk_folder = fj['unknown-folder']
    candidates = [atr['author-name'] for atr in fj['candidate-authors']]

    test = read_files(probpath, unk_folder)
    train = []
    for candidate in candidates:
        train.extend(read_files(probpath, candidate))

    return train, test, unk_folder


def iter_problems(path):
    infocollection = os.path.join(path, 'collection-info.json')
    with open(infocollection, 'r') as f:
        problems = [(atr['problem-name'], atr['language']) for atr in json.load(f)]

    for problem, lang in problems:
        yield (problem, *load_problem_data(path, problem))


def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()

    for problem, train_docs, test_docs, unk_folder in iter_problems(path):
        x_train, y_train = zip(*train_docs)
        x_test, _ = zip(*test_docs)

        extractor = BaselineFeatureExtractor()
        extractor.fit(x_train)
        train_data = extractor.transform(x_train)
        test_data = extractor.transform(x_test)

        print('\t', len(set(y_train)), 'candidate authors')
        print('\t', len(x_train), 'known texts')
        print('\t', len(x_test), 'unknown texts')

        clf = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, gamma='auto')), cv=3)

        clf.fit(train_data, y_train)
        predictions = clf.predict(test_data)
        proba = clf.predict_proba(test_data)

        # Reject option (used in open-set cases)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count = count + 1
        print('\t', count, 'texts left unattributed')
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
    baseline('./data', './output', 3, 5, 0.1)
