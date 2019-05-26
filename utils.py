from sklearn.metrics import f1_score
import warnings
import os
import glob
import json

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

def print_report(problem, predictions):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1 = f1_score(problem.test.y, predictions, average='macro')
    print('\t', (predictions == '<UNK>').sum(), 'texts left unattributed')
    print('\t', len(set(problem.train.y)), 'candidate authors')
    print('\t', len(problem.train.X), 'known texts')
    print('\t', len(problem.test.X), 'unknown texts')
    # f1 score is a bit off, because it's not called exactly as in evaluator
    print('\t {:.3f} is f1 score'.format(f1))
