"""Paired error analysis and article-cluster bootstrap for a frozen comparison."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from offline_handoff import contract as c
from squad_eval import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('outputs/official-dev-final'))
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=4248)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Positive resample count required.')
    out = args.results_dir
    result = c.decode((out / 'complete.json').read_bytes())
    freeze = c.decode((out / 'freeze.json').read_bytes())
    data_raw = (ROOT / freeze['data']['path']).read_bytes()
    c.require(c.sha256(data_raw) == freeze['data']['sha256'], 'dev data changed')
    dataset = c.decode(data_raw)
    articles = [{q['id'] for p in a['paragraphs'] for q in p['qas']} for a in dataset['data']]
    counts = np.array([len(ids) for ids in articles])
    c.require(counts.sum() == len(set().union(*articles)), 'duplicate dev IDs')
    scores, predictions = {}, {}
    for row in result['results']:
        name = row['system']
        raw = (out / (name + '-predictions.json')).read_bytes()
        c.require(c.sha256(raw) == row['predictions_sha256'], 'predictions changed')
        predictions[name] = c.decode(raw)
        scores[name] = evaluate(dataset['data'], predictions[name])
        for key in ('exact', 'f1', 'total'):
            c.require(abs(scores[name][key] - row[key]) < 1e-10, 'score reproduction failed')
    indices = np.random.default_rng(args.seed).integers(0, len(articles), size=(args.resamples, len(articles)))
    denominator = counts[indices].sum(axis=1)
    pairs = [('basic_no_length', 'bert_baseline'), ('full_no_length', 'bert_baseline'),
             ('basic_raw_length_matched_alpha', 'bert_baseline'),
             ('basic_no_length', 'basic_raw_length_matched_alpha'), ('full_no_length', 'basic_no_length')]
    comparisons = []
    for left, right in pairs:
        delta = np.array([[sum(scores[left][key][q] - scores[right][key][q] for q in ids)
                           for key in ('exact_per_question', 'f1_per_question')] for ids in articles])
        sampled = 100 * delta[indices].sum(axis=1) / denominator[:, None]
        point = 100 * delta.sum(axis=0) / counts.sum()
        row = {'left': left, 'right': right,
               'changed_answer_texts': sum(predictions[left][q] != predictions[right][q] for q in predictions[left]),
               'em_corrected': sum(scores[left]['exact_per_question'][q] > scores[right]['exact_per_question'][q] for q in predictions[left]),
               'em_broken': sum(scores[left]['exact_per_question'][q] < scores[right]['exact_per_question'][q] for q in predictions[left])}
        for i, key in enumerate(('exact', 'f1')):
            row[key] = {'delta_pp': float(point[i]),
                        'article_bootstrap_95_percentile_interval_pp': np.percentile(sampled[:, i], [2.5, 97.5]).tolist()}
        comparisons.append(row)
    report = {'method': 'Paired article-cluster bootstrap: sample 48 articles with replacement; retain all their questions; recompute question-weighted metric differences. Percentile intervals are descriptive, not evidence of an untouched test set.',
              'articles': len(articles), 'questions': int(counts.sum()), 'resamples': args.resamples,
              'seed': args.seed, 'comparisons': comparisons,
              'complete_sha256': c.sha256((out / 'complete.json').read_bytes()),
              'script_sha256': c.sha256(Path(__file__).read_bytes())}
    with (out / 'paired-analysis.json').open('xb') as handle:
        handle.write(c.encode(report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
