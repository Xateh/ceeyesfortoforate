"""Reproducible local feature/alpha sweep over existing label-free BERT exports.

This numerical experiment does not authenticate upstream BERT provenance.
Run from the repository root. The output directory must be new.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from offline_handoff import contract as c
from reranker import real_training as r
from reranker.core import Candidate, FeatureConfig, Question, Reranker, questions_from_records


def transformed(questions):
    """Label-free transforms; mirror the producer's existing question types."""
    result = []
    for q in questions:
        candidates = []
        for candidate in q.candidates:
            f = dict(candidate.features)
            length = f['span_token_count']
            f['log_span_tokens'] = math.log1p(length)
            bucket = ('1' if length <= 1 else '2_3' if length <= 3 else
                      '4_7' if length <= 7 else '8_15' if length <= 15 else '16_plus')
            for name in ('1', '2_3', '4_7', '8_15', '16_plus'):
                f['length_bucket_' + name] = int(bucket == name)
            for kind in c.TYPES:
                f[kind + '_log_span_tokens'] = f['q_' + kind] * f['log_span_tokens']
            candidates.append(Candidate(candidate.candidate_id, candidate.answer, f))
        result.append(Question(q.question_id, candidates))
    return result


def cutoff(questions, k):
    return [Question(q.question_id, q.candidates[:k]) for q in questions]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/reranker-exploration.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--inference-dir', type=Path, default=Path('inference'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/reranker-exploration'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory.')
    config_raw = args.config.read_bytes()
    config = c.decode(config_raw)
    if config['selection'] != 'validation_f1_then_em_then_first_alpha':
        parser.error('Unsupported selection rule.')
    if not config['alphas'] or len(set(config['alphas'])) != len(config['alphas']):
        parser.error('Provide distinct positive alphas.')
    for alpha in config['alphas']:
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            parser.error('Alphas must be finite and positive.')
    if not config['profiles'] or len({p['id'] for p in config['profiles']}) != len(config['profiles']):
        parser.error('Provide distinct profile IDs.')
    for profile in config['profiles']:
        if not profile['id'].replace('_', '').isalnum():
            parser.error('Profile IDs must be alphanumeric with optional underscores.')
        FeatureConfig(profile['basic'], profile['compatibility'], profile['group'])
    for k in config['training_top_k']:
        if type(k) is not int or not 1 <= k <= 20:
            parser.error('Training top-k must be between 1 and 20.')
        for ek in config['evaluation_top_k'][str(k)]:
            if type(ek) is not int or not 1 <= ek <= 20:
                parser.error('Evaluation top-k must be between 1 and 20.')

    started = time.monotonic()
    paths = {'source_train': args.data_dir / 'train-v1.1.json',
             **{role: args.data_dir / 'splits' / (role + '.json') for role in r.ROLES},
             'train_candidates': args.inference_dir / 'reranker_train_candidates.json',
             'validation_candidates': args.inference_dir / 'internal_val_candidates.json',
             'validation_baseline': args.inference_dir / 'internal_val_predictions.json'}
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    policy = c.decode((ROOT / 'configs/reranker-real.example.json').read_bytes())['policy']
    policy['decision_reference'] = 'configs/reranker-exploration.json: predeclared numerical sweep'
    print('Validating unchanged source splits, candidates, and baseline...', flush=True)
    splits, training, validation, baseline = r.validate_inputs(blobs, policy)
    inputs = {key: {'path': str(paths[key]), 'sha256': c.sha256(raw)} for key, raw in blobs.items()}
    del blobs
    evaluate = r.load_scorer()
    print('Computing training-only F1 targets once for the original candidate pool...', flush=True)
    targets = r.targets_from_training(evaluate, splits['reranker_train'][0], training['questions'])
    train_all = transformed(questions_from_records(training['questions']))
    val_all = transformed(questions_from_records(validation['questions']))
    available = set(train_all[0].candidates[0].features)
    for profile in config['profiles']:
        if not set(FeatureConfig(profile['basic'], profile['compatibility'], profile['group']).names) <= available:
            parser.error('Unknown feature in profile ' + profile['id'])
    del training, validation
    val_sets = {k: cutoff(val_all, k) for ks in config['evaluation_top_k'].values() for k in ks}
    baseline_metrics = r.score(evaluate, splits['internal_val'][0], baseline)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_bytes(config_raw)
    (out / 'targets.json').write_bytes(c.encode(targets))
    software = (*r.SOFTWARE_FILES, *r.SOURCE_FILES, 'scripts/reranker_experiments.py')
    manifest = {'scope': 'exploratory_internal_validation_numerical_sweep', 'inputs': inputs,
                'config_sha256': c.sha256(config_raw),
                'software_sha256': {name: c.sha256((ROOT / name).read_bytes()) for name in software},
                'python': platform.python_version(), 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                'packages': {name: version(name) for name in ('torch', 'numpy')},
                'target_policy': r.TARGET, 'selection': config['selection']}
    (out / 'manifest.json').write_bytes(c.encode(manifest))
    rows = []
    total_fits = len(config['profiles']) * len(config['training_top_k']) * len(config['alphas'])
    fits = 0
    for train_k in config['training_top_k']:
        train = cutoff(train_all, train_k)
        train_targets = {q.question_id: {v.candidate_id: targets[q.question_id][v.candidate_id]
                                       for v in q.candidates} for q in train}
        for profile in config['profiles']:
            features = FeatureConfig(profile['basic'], profile['compatibility'], profile['group'])
            for alpha_order, alpha in enumerate(config['alphas']):
                model = Reranker(features, alpha=alpha, no_signal='error').fit(train, train_targets)
                fits += 1
                trial = out / f"{profile['id']}-train{train_k}-alpha{alpha:g}"
                trial.mkdir()
                model.save(trial / 'model.json')
                before = r.state_bytes(model)
                restored = Reranker.load(trial / 'model.json')
                if r.state_bytes(restored) != before:
                    raise RuntimeError('Model save/reload state differs.')
                for eval_k in config['evaluation_top_k'][str(train_k)]:
                    val = val_sets[eval_k]
                    predictions = model.predict(val)
                    if restored.predict(val) != predictions or restored.scores(val) != model.scores(val):
                        raise RuntimeError('Reloaded scores/predictions differ.')
                    if r.state_bytes(model) != before:
                        raise RuntimeError('Evaluation mutated fitted state.')
                    metrics = r.score(evaluate, splits['internal_val'][0], predictions)
                    corrected = sum(metrics['exact_per_question'][q] > baseline_metrics['exact_per_question'][q] for q in predictions)
                    broken = sum(metrics['exact_per_question'][q] < baseline_metrics['exact_per_question'][q] for q in predictions)
                    (trial / f'eval{eval_k}-predictions.json').write_bytes(c.encode(predictions))
                    row = {'profile': profile['id'], 'train_k': train_k, 'eval_k': eval_k,
                           'alpha': alpha, 'alpha_order': alpha_order,
                           'exact': metrics['exact'], 'f1': metrics['f1'], 'total': metrics['total'],
                           'delta_em_vs_bert': metrics['exact'] - baseline_metrics['exact'],
                           'delta_f1_vs_bert': metrics['f1'] - baseline_metrics['f1'],
                           'em_corrected_vs_bert': corrected, 'em_broken_vs_bert': broken,
                           'changed_answers_vs_bert': sum(predictions[q] != baseline[q] for q in predictions),
                           'features': list(features.names), 'trial_dir': trial.name,
                           'model_sha256': c.sha256(before), 'reload_equivalence': 'exact'}
                    rows.append(row)
                    (trial / f'eval{eval_k}-metrics.json').write_bytes(c.encode(row))
                    print(f"[{fits}/{total_fits}] {profile['id']} train{train_k}/eval{eval_k} alpha={alpha:g}: "
                          f"EM {metrics['exact']:.6f}, F1 {metrics['f1']:.6f}", flush=True)
    selected = []
    for profile in config['profiles']:
        for train_k in config['training_top_k']:
            for eval_k in config['evaluation_top_k'][str(train_k)]:
                trials = [v for v in rows if (v['profile'], v['train_k'], v['eval_k']) == (profile['id'], train_k, eval_k)]
                selected.append(max(trials, key=lambda v: (v['f1'], v['exact'], -v['alpha_order'])))
    columns = [key for key in rows[0] if key != 'features']
    with (out / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    report = {'baseline': {k: baseline_metrics[k] for k in ('exact', 'f1', 'total')},
              'fits': fits, 'evaluations': len(rows), 'trials': rows, 'selected_per_condition': selected,
              'wall_seconds': time.monotonic() - started, 'manifest_sha256': c.sha256((out / 'manifest.json').read_bytes())}
    (out / 'complete.json').write_bytes(c.encode(report))
    print(f'Complete: {fits} fits, {len(rows)} evaluations; results in {out}', flush=True)


if __name__ == '__main__':
    main()
