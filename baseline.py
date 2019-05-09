import os
import glob
import json
import argparse
import time
import codecs
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV


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


def baseline_features(x_train, y_train, x_test, n, min_f):
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(n, n),
        lowercase=False,
        vocabulary=extract_vocabulary(zip(x_train, y_train), n, min_f)
    )

    train_data = vectorizer.fit_transform(x_train).astype(float)
    test_data = vectorizer.transform(x_test).astype(float)

    for i, v in enumerate(x_train):
        train_data[i] = train_data[i] / len(x_train[i])

    for i, v in enumerate(x_test):
        test_data[i] = test_data[i] / len(x_test[i])

    max_abs_scaler = preprocessing.MaxAbsScaler()
    train_data = max_abs_scaler.fit_transform(train_data)
    test_data = max_abs_scaler.transform(test_data)

    return train_data, test_data

def represent_text(text, n):
    # Extracts all character 'n'-grams from  a 'text'
    if n > 0:
        tokens = [text[i:i + n] for i in range(len(text) - n + 1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def extract_vocabulary(texts, n, ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences = defaultdict(int)
    for (text, label) in texts:
        text_occurrences = represent_text(text, n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]
    vocabulary = []
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)
    return vocabulary


def baseline(path, outpath, n=3, ft=5, pt=0.1):
    start_time = time.time()

    for problem, train_docs, test_docs, unk_folder in iter_problems(path):
        x_train, y_train = zip(*train_docs)
        x_test, _ = zip(*test_docs)

        train_data, test_data = baseline_features(
            x_train, y_train, x_test, n, ft
        )

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
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
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
