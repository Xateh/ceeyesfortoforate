"""CPU-only reconstruction of epoch-5 calibration and BERT official-dev scores."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.squad_eval import evaluate


def read(path):
    return json.loads(path.read_bytes())


def load_split(exports, splits, scores, split):
    payload = read(exports / f'{split}_candidates.json')
    if payload['schema_version'] != 1 or payload['metadata']['split'] != split:
        raise ValueError('Wrong export split/schema')
    data = read(splits / f'{split}.json')['data']
    gold = {q['id']: (a['title'], p['context'], q['question'])
            for a in data for p in a['paragraphs'] for q in p['qas']}
    questions = payload['questions']
    ids = [q['question_id'] for q in questions]
    if len(set(ids)) != len(ids) or set(ids) != set(gold):
        raise ValueError('Candidate coverage/duplicate IDs')
    with (scores / f'{split}_scores.csv').open(encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle))
    by_id = {r['question_id']: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(ids):
        raise ValueError('Score coverage/duplicate IDs')
    features, predictions = [], [{}, {}]
    for q in questions:
        qid = q['question_id']
        article, context, question = gold[qid]
        if (q['article_id'], q['context'], q['question']) != (article, context, question.strip()):
            raise ValueError(f'Source identity mismatch: {qid}')
        first, second = q['candidates'][:2]
        for j, span in enumerate((first, second)):
            if context[span['start']:span['end']] != span['answer']:
                raise ValueError(f'Candidate offset mismatch: {qid}')
            predictions[j][qid] = span['answer']
        r = by_id[qid]
        gap = first['combined_score'] - second['combined_score']
        correction = float(r['score_2']) - float(r['score_1'])
        if gap < 0 or gap != float(r['bert_gap']) or correction != float(r['correction']):
            raise ValueError(f'Score/candidate mismatch: {qid}')
        features.append([gap, correction])
    x = np.asarray(features, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    metrics = [evaluate(data, p) for p in predictions]
    delta = np.array([metrics[1]['f1_per_question'][i] - metrics[0]['f1_per_question'][i] for i in ids])
    return dict(ids=ids, data=data, predictions=predictions, x=x, delta=delta, baseline=metrics[0])


def fit(train, columns):
    informative = train['delta'] != 0
    x = train['x'][informative][:, columns]
    scaler = StandardScaler().fit(x)  # Unweighted scaling, training unequal-F1 pairs only.
    model = LogisticRegression(C=1, max_iter=1000).fit(
        scaler.transform(x), train['delta'][informative] > 0,
        sample_weight=np.abs(train['delta'][informative]))
    if model.n_iter_.max() >= 1000:
        raise RuntimeError('Calibration did not converge')
    return dict(features=[['bert_gap', 'correction'][j] for j in columns],
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                classes=model.classes_.tolist(), informative_pairs=int(informative.sum()),
                n_iter=model.n_iter_.tolist(), C=1, max_iter=1000)


def predict(frozen, x):
    """Inference uses saved parameters only; ties retain candidate 1."""
    return ((x - np.array(frozen['mean'])) / np.array(frozen['scale'])) @ np.array(frozen['coef']) + frozen['intercept'] > 0


def report(split, selected):
    p = {i: split['predictions'][int(s)][i] for i, s in zip(split['ids'], selected)}
    m = evaluate(split['data'], p)
    baseline = split['baseline']
    return dict(questions=len(p), changed=sum(p[i] != split['predictions'][0][i] for i in p),
                fixed=sum(m['exact_per_question'][i] > baseline['exact_per_question'][i] for i in p),
                broken=sum(m['exact_per_question'][i] < baseline['exact_per_question'][i] for i in p),
                exact=m['exact'], f1=m['f1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, default=Path('outputs/issue7-validation-20261104/inference'))
    parser.add_argument('--splits', type=Path, default=Path('data/splits'))
    parser.add_argument('--scores', type=Path, default=Path('outputs/context-top2-span-mean-epoch5-ranking-diagnostic'))
    parser.add_argument('--dev-candidates', type=Path, default=Path('outputs/official-dev/dev_candidates.json'))
    parser.add_argument('--dev-gold', type=Path, default=Path('data/dev-v1.1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train, valid = [load_split(args.exports, args.splits, args.scores, s)
                    for s in ('reranker_train', 'internal_val')]
    if set(train['ids']) & set(valid['ids']):
        raise ValueError('Training/validation overlap')
    if len(train['ids']) != 12516 or len(valid['ids']) != 9131:
        raise ValueError('Unexpected split size')
    if valid['predictions'][0] != read(args.exports / 'internal_val_predictions.json'):
        raise ValueError('Baseline differs from rank one')
    # Both models are frozen before validation reporting. No validation fitting/tuning.
    models = {name: fit(train, columns) for name, columns in [('gap_correction', [0, 1]), ('gap_only', [0])]}
    model_path = args.output / 'frozen_models.json'
    model_path.write_text(json.dumps(models, indent=2), encoding='utf-8')
    models = read(model_path)
    results = {name: report(valid, predict(models[name], valid['x'][:, columns]))
               for name, columns in [('gap_correction', [0, 1]), ('gap_only', [0])]}
    results['uncalibrated_span_mean'] = report(valid, valid['x'][:, 1] > valid['x'][:, 0])
    results['baseline'] = {k: valid['baseline'][k] for k in ('exact', 'f1', 'total')}
    dev = read(args.dev_candidates)
    questions = dev['questions']
    dev_predictions = {q['question_id']: q['candidates'][0]['answer'] for q in questions}
    if len(questions) != 10570 or len(dev_predictions) != 10570:
        raise ValueError('Official-dev must contain 10,570 unique IDs')
    dev_metrics = evaluate(args.dev_gold, dev_predictions)  # Exact ID set checked by evaluator wrapper.
    results['bert_official_dev'] = {k: dev_metrics[k] for k in ('exact', 'f1', 'total')}
    (args.output / 'bert_dev_predictions.json').write_text(json.dumps(dev_predictions), encoding='utf-8')
    expected = dict(changed=344, fixed=81, broken=78, exact=65.436425, f1=79.339035)
    results['calibration_mismatches'] = {k: dict(expected=v, actual=results['gap_correction'][k])
        for k, v in expected.items() if (round(results['gap_correction'][k], 6) != v)}
    files = [args.exports / f'{s}_candidates.json' for s in ('reranker_train', 'internal_val')]
    files += [args.splits / f'{s}.json' for s in ('reranker_train', 'internal_val')]
    files += [args.scores / f'{s}_scores.csv' for s in ('reranker_train', 'internal_val')]
    files += [args.exports / 'internal_val_predictions.json', args.dev_candidates, args.dev_gold,
              Path('evaluate-v2.0.py'), Path('scripts/squad_eval.py')]
    results['input_sha256'] = {}
    for path in files:
        with path.open('rb') as handle:
            results['input_sha256'][path.as_posix()] = hashlib.file_digest(handle, 'sha256').hexdigest()
    results['python'] = sys.version
    results['versions'] = {p: importlib.metadata.version(p) for p in
                           ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'joblib', 'threadpoolctl')}
    (args.output / 'checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    if results['calibration_mismatches']:
        raise SystemExit('Calibration differs from requested result; see checks.json')


if __name__ == '__main__':
    main()
