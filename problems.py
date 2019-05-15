import os
import codecs
import glob
import json


class Problems:
    def __init__(self, path):
        self.path = path

    def read_files(self, path, label):
        # Reads all text files located in the 'path' and assigns them to 'label' class
        files = glob.glob(os.path.join(path, label, '*.txt'))
        texts = []
        for i, v in enumerate(files):
            f = codecs.open(v, 'r', encoding='utf-8')
            texts.append((f.read(), label))
            f.close()
        return texts

    def load_problem_data(self, problem):
        probpath = os.path.join(self.path, problem)
        path = os.path.join(probpath, 'problem-info.json')
        with open(path, 'r') as f:
            fj = json.load(f)

        unk_folder = fj['unknown-folder']
        candidates = [atr['author-name'] for atr in fj['candidate-authors']]

        test = self.read_files(probpath, unk_folder)
        train = []
        for candidate in candidates:
            train.extend(self.read_files(probpath, candidate))

        return train, test, unk_folder

    def iter(self):
        infocollection = os.path.join(self.path, 'collection-info.json')
        with open(infocollection, 'r') as f:
            problems = [(atr['problem-name'], atr['language']) for atr in json.load(f)]

        for problem, lang in problems:
            yield (problem, *self.load_problem_data(problem), lang)
