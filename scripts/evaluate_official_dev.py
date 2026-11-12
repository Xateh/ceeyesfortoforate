"""Evaluate only the preselected models in a pinned official-dev comparison.

All predictions are saved before any dev metrics are computed. No selection or
training is enabled. Run from the repository root after dev candidate generation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_handoff import contract as c
from reranker.official_dev import adapt_official_dev, membership, predict_frozen, source_records


def checked(descriptor):
    path = ROOT / descriptor['path']
    raw = path.read_bytes()
    c.require(c.sha256(raw) == descriptor['sha256'], 'frozen input hash mismatch: ' + str(path))
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=Path('configs/official-dev-frozen.json'))
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--candidates', type=Path, default=Path('outputs/official-dev-candidates/dev_candidates.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/official-dev-final'))
    args = parser.parse_args()
    raw = args.freeze.read_bytes()
    c.require(c.sha256(raw) == args.expected_freeze_sha256, 'freeze changed after selection')
    freeze = c.decode(raw)
    c.require(freeze['status'] == 'frozen_before_local_official_dev_inference', 'comparison not frozen')
    if args.output_dir.exists():
        parser.error('Choose a new output directory.')
    selection = c.decode(checked(freeze['selection_results']))
    selection_path = ROOT / freeze['selection_results']['path']
    manifest_raw = (selection_path.parent / 'manifest.json').read_bytes()
    c.require(c.sha256(manifest_raw) == selection['manifest_sha256'], 'selection manifest changed')
    manifest = c.decode(manifest_raw)
    for name, expected_hash in manifest['software_sha256'].items():
        c.require(c.sha256((ROOT / name).read_bytes()) == expected_hash, 'selection software changed: ' + name)
    source = source_records(c.decode(checked(manifest['inputs']['source_train'])))
    dataset = c.decode(checked(freeze['data']))
    expected = source_records(dataset)
    for descriptor in freeze['baseline']['checkpoint_files'].values(): checked(descriptor)
    for descriptor in freeze['tokenized_dev_files'].values(): checked(descriptor)
    checked(freeze['exporter'])
    for specification in freeze['models']: checked(specification['model'])
    candidates_raw = args.candidates.read_bytes()
    batch = adapt_official_dev(candidates_raw, expected=expected, excluded=membership(source),
                               generation=freeze['generation'])
    c.require(batch['metadata']['checkpoint'] == freeze['baseline']['checkpoint_path'], 'checkpoint path mismatch')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / 'freeze.json').write_bytes(raw)
    predictions = {'bert_baseline': batch['baseline']}
    print('Predicting with frozen models; no refitting or dev-based selection...', flush=True)
    for specification in freeze['models']:
        name = specification['id']
        c.require(name not in predictions, 'duplicate frozen model ID')
        predictions[name] = predict_frozen(ROOT / specification['model']['path'], specification, batch['questions'])
    for name, answers in predictions.items():
        c.keys(answers, expected, 'dev prediction coverage')
        (out / (name + '-predictions.json')).write_bytes(c.encode(answers))
    # Annotations are used here, only after every frozen system has predicted.
    sys.path.insert(0, str(ROOT / 'scripts'))
    from squad_eval import evaluate
    scores = {name: evaluate(dataset['data'], answers) for name, answers in predictions.items()}
    baseline = scores['bert_baseline']
    rows = []
    for name, metrics in scores.items():
        row = {'system': name, **{k: metrics[k] for k in ('exact', 'f1', 'total')},
               'delta_em_vs_bert': metrics['exact'] - baseline['exact'],
               'delta_f1_vs_bert': metrics['f1'] - baseline['f1'],
               'em_corrected_vs_bert': sum(metrics['exact_per_question'][q] > baseline['exact_per_question'][q] for q in expected),
               'em_broken_vs_bert': sum(metrics['exact_per_question'][q] < baseline['exact_per_question'][q] for q in expected),
               'predictions_sha256': c.sha256((out / (name + '-predictions.json')).read_bytes())}
        rows.append(row)
        print(f"{name}: EM {metrics['exact']:.6f}, F1 {metrics['f1']:.6f}, n={metrics['total']}", flush=True)
    with (out / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {'scope': 'frozen_limited_official_dev_comparison', 'freeze_sha256': c.sha256(raw),
              'candidates_sha256': c.sha256(candidates_raw), 'generation': batch['metadata'], 'results': rows,
              'software_sha256': {name: c.sha256((ROOT / name).read_bytes()) for name in
                                 ('reranker/official_dev.py', 'scripts/evaluate_official_dev.py',
                                  'reranker/core.py', 'reranker/team_candidates.py', 'scripts/squad_eval.py', 'evaluate-v2.0.py')},
              'evaluation_history': freeze['evaluation_history'],
              'checks': {'complete_coverage': True, 'source_train_excluded': True, 'frozen_model_hashes_match': True,
                         'no_model_state_changed': True, 'all_predictions_saved_before_scoring': True}}
    (out / 'complete.json').write_bytes(c.encode(result))


if __name__ == '__main__':
    main()
