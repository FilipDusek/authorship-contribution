import os
import codecs
import glob
import json
from types import SimpleNamespace

class Problem:
    def __init__(self, problem, train_docs, test_docs, unk_folder, lang):
        self.problem_name = problem
        self.train_docs = train_docs
        self.unk_folder = unk_folder
        self.lang = lang

        train_x, train_y = zip(*train_docs)
        self.train = SimpleNamespace(X=train_x, y=train_y)

        test_x, test_y = zip(*test_docs)
        self.test = SimpleNamespace(X=test_x, y=test_y)


class ProblemLoader:
    def __init__(self, path):
        self.path = path

    def load_problem_solutions(self, problem_name):
        filename = os.path.join(self.path, problem_name, 'ground-truth.json')
        with open(filename, 'r') as f:
            answers_stupid_format = json.load(f)['ground_truth']

        tups = [(p['unknown-text'], p['true-author']) for p in answers_stupid_format]
        answers_ok_format = dict(tups)

        return answers_ok_format

    def read_files(self, path, label):
        # Reads all text files located in the 'path' and assi gns them to 'label' class
        files = glob.glob(os.path.join(path, label, '*.txt'))
        texts = []
        for i, v in enumerate(files):
            f = codecs.open(v, 'r', encoding='utf-8')
            texts.append((f.read(), label, v))
            f.close()
        return texts

    def load_problem_data(self, problem):
        probpath = os.path.join(self.path, problem)
        path = os.path.join(probpath, 'problem-info.json')
        # print(problem, probpath)
        with open(path, 'r') as f:
            fj = json.load(f)

        unk_folder = fj['unknown-folder']
        candidates = [atr['author-name'] for atr in fj['candidate-authors']]

        test = []
        solutions = self.load_problem_solutions(problem)
        for data, _, path in self.read_files(probpath, unk_folder):
            text_fname = os.path.split(path)[1]
            test.append((data, solutions[text_fname]))

        train = []
        for candidate in candidates:
            files = self.read_files(probpath, candidate)
            files = [f[:2] for f in files]
            train.extend(files)

        return train, test, unk_folder

    def iter(self):
        infocollection = os.path.join(self.path, 'collection-info.json')
        with open(infocollection, 'r') as f:
            problems = [(atr['problem-name'], atr['language']) for atr in json.load(f)]

        for problem, lang in problems:
            yield Problem(problem, *self.load_problem_data(problem), lang)
