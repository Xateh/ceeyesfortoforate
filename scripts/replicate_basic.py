"""Run basic alpha-1 experiments with configurable training/evaluation top-k.

Numerical replication only: this does not authenticate the original BERT run
or replace reranker.real_training's provenance manifest.
Run from the repository root: python scripts/replicate_basic.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_handoff import contract as c
from reranker import real_training as r
from reranker.core import FeatureConfig, Reranker, questions_from_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--inference-dir', type=Path, default=Path('inference'))
    parser.add_argument('--train-top-k', type=int, choices=range(1, 21), default=20,
                        metavar='1..20', help='Retain the first k BERT-ranked training candidates.')
    parser.add_argument('--eval-top-k', type=int, choices=range(1, 21), default=20,
                        metavar='1..20', help='Retain the first k BERT-ranked validation candidates.')
    parser.add_argument('--drop-feature', action='append', choices=c.BASIC, default=[],
                        help='Exclude a basic feature; repeat to drop multiple features.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/replication-basic'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new --output-dir.')
    kept_features = [name for name in c.BASIC if name not in args.drop_feature]
    if not kept_features:
        parser.error('Keep at least one basic feature.')

    paths = {
        'source_train': args.data_dir / 'train-v1.1.json',
        **{role: args.data_dir / 'splits' / (role + '.json') for role in r.ROLES},
        'train_candidates': args.inference_dir / 'reranker_train_candidates.json',
        'validation_candidates': args.inference_dir / 'internal_val_candidates.json',
        'validation_baseline': args.inference_dir / 'internal_val_predictions.json',
    }
    policy = json.loads((ROOT / 'configs/reranker-real.example.json').read_text())['policy']
    policy['decision_reference'] = 'scripts/replicate_basic.py: fixed alpha-1 numerical replication'
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    print('Validating splits, source identities, candidates, and baseline parity...', flush=True)
    splits, training, validation, baseline = r.validate_inputs(blobs, policy)
    input_hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in blobs.items()}
    del blobs

    # Restrict only after validating the unchanged producer exports. Candidate
    # order is BERT order, never gold quality or learned reranker order.
    train_records = [dict(q, candidates=q['candidates'][:args.train_top_k])
                     for q in training['questions']]
    val_records = [dict(q, candidates=q['candidates'][:args.eval_top_k])
                   for q in validation['questions']]
    evaluate = r.load_scorer()
    print('Computing candidate F1 targets from reranker_train golds only...', flush=True)
    targets = r.targets_from_training(evaluate, splits['reranker_train'][0], train_records)
    config = c.feature_config('basic')
    model = Reranker(FeatureConfig(kept_features, config['compatibility'], 'basic'),
                     alpha=1.0, no_signal='error')
    train_questions = questions_from_records(train_records)
    val_questions = questions_from_records(val_records)
    print(f'Fitting {len(kept_features)}-feature basic reranker on CPU '
          f'(train top {args.train_top_k}, evaluate top {args.eval_top_k})...', flush=True)
    model.fit(train_questions, targets)
    predictions = model.predict(val_questions)
    baseline_metrics = r.score(evaluate, splits['internal_val'][0], baseline)
    metrics = r.score(evaluate, splits['internal_val'][0], predictions)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / 'basic-model.json'
    model.save(model_path)
    restored = Reranker.load(model_path)
    if restored.predict(val_questions) != predictions or restored.scores(val_questions) != model.scores(val_questions):
        raise RuntimeError('Saved-model reload differs from fitted model.')
    (args.output_dir / 'basic-predictions.json').write_bytes(c.encode(predictions))
    report = {
        'status': 'local_numerical_replication_original_run_provenance_unverified',
        'alpha': 1.0, 'features': list(model.features.names),
        'dropped_features': [name for name in c.BASIC if name in args.drop_feature],
        'train_top_k': args.train_top_k, 'eval_top_k': args.eval_top_k,
        'training_questions': len(train_questions),
        'training_candidates': sum(len(q.candidates) for q in train_questions),
        'validation_candidates': sum(len(q.candidates) for q in val_questions),
        'baseline': {key: baseline_metrics[key] for key in ('exact', 'f1', 'total')},
        'basic': {key: metrics[key] for key in ('exact', 'f1', 'total')},
        'inputs': {key: {'path': str(paths[key]), 'sha256': value}
                   for key, value in input_hashes.items()},
        'software_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in (*r.SOFTWARE_FILES, *r.SOURCE_FILES, 'scripts/replicate_basic.py')},
        'reload_equivalence': 'exact',
    }
    (args.output_dir / 'metrics.json').write_bytes(c.encode(report))
    print(f"Baseline: EM {baseline_metrics['exact']:.6f}, F1 {baseline_metrics['f1']:.6f}")
    print(f"Basic:    EM {metrics['exact']:.6f}, F1 {metrics['f1']:.6f} ({metrics['total']} questions)")
    print(f'Saved model, predictions, and metrics to {args.output_dir}')


if __name__ == '__main__':
    main()
