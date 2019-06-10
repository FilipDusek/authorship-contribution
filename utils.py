from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings
import os
import glob
import json
import functools


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def fex_signature(cls, args):
    params = ', '.join(['{}={}'.format(k, v) for k, v in args.items()])
    return '{}({})'.format(cls.__name__, params)

def save_answers(problem, predictions, path, outpath):
    # Saving output data
    out_data = []
    unk_filelist = glob.glob(
        os.path.join(
            path, problem.problem_name,
            problem.unk_folder, '*.txt'
        )
    )
    pathlen = len(
        os.path.join(
            path, problem.problem_name, problem.unk_folder
        )
    )

    for i, v in enumerate(predictions):
        out_data.append({'unknown-text': unk_filelist[i][pathlen + 1:], 'predicted-author': v})
    with open(os.path.join(outpath, 'answers-' + problem.problem_name + '.json'), 'w') as f:
        json.dump(out_data, f, indent=4)
    print('\t', 'answers saved to file', 'answers-' + problem.problem_name + '.json')

def print_fex_report(fexs):
    fex_sigs = [fex_signature(fex, args) for fex, args, _ in fexs]
    print('  Fex  ' + '\n  '.join(fex_sigs))

def print_result_report(problem, predictions):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(problem.test.y, predictions, average='macro')
    # f1 score is a bit off, because it's not called exactly as in evaluator
    print('  F1 = {:.3f} is f1 score'.format(f1))

def make_benchmark_row(classifier_cls, fexs, problem, predictions):
    benchmark_row = dict(fexs[0][1])
    benchmark_row['name'] = fexs[0][2]
    benchmark_row['clf'] = classifier_cls.__name__

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = precision_recall_fscore_support(
            problem.test.y, predictions, average='macro'
        )

    scores = dict(zip(['precision', 'recall', 'f1', 'support'], scores))
    print('  F1 = {:.2f}'.format(scores['f1']))
    benchmark_row.update(scores)
    return benchmark_row
