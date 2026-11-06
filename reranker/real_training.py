"""Explicit-policy real-label runner; no BERT generation or automatic downloads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import tempfile

from offline_handoff import contract as c
from . import team_candidates as team
from .core import FeatureConfig, Reranker, questions_from_records
from .team_model import GENERATION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'cs4248.real-reranker-run/0.1.0-draft'
BINDING = 'cs4248.real-reranker-binding/0.1.0-draft'
ROLES = ('bert_train', 'reranker_train', 'internal_val')
INPUTS = ('source_train', *ROLES, 'train_candidates', 'validation_candidates',
          'validation_baseline')
SOURCE_FILES = ('evaluate-v2.0.py', 'scripts/squad_eval.py')
SOFTWARE_FILES = ('reranker/real_training.py', 'reranker/core.py',
                  'reranker/team_candidates.py', 'offline_handoff/contract.py')
TARGET = 'repository-scorer-f1-max-normalized-nonempty-golds/1'


def digest(value):
    c.require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA-256 required')


def policy_check(policy):
    c.keys(policy, ('id', 'decision_reference', 'acceptance', 'features', 'alphas',
                    'target', 'no_signal', 'selection'), 'explicit policy')
    for key in ('id', 'decision_reference'):
        c.text(policy[key], key, True)
        c.require(bool(policy[key].strip()), 'empty policy reference')
    c.require(policy['acceptance'] == 'explicit_run_policy_not_team_agreement', 'policy status required')
    c.require(policy['features'] == {g: c.feature_config(g) for g in ('basic', 'full')},
              'unsupported feature registry/order/spec')
    c.require(policy['target'] == TARGET and policy['no_signal'] == 'error', 'unsupported target/no-signal policy')
    c.require(policy['selection'] == 'internal_val_f1_then_em_then_first_alpha', 'selection policy required')
    alphas = policy['alphas']
    c.require(type(alphas) is list and 1 <= len(alphas) <= 10, 'explicit bounded alpha grid required')
    for alpha in alphas:
        c.number(alpha, 'alpha'); c.require(alpha > 0, 'alpha must be positive')
    c.require(len(set(alphas)) == len(alphas), 'duplicate alpha')


def dataset_index(raw):
    data = c.decode(raw); c.keys(data, ('version', 'data'), 'SQuAD train-derived JSON')
    c.require(data['version'] == '1.1' and type(data['data']) is list and data['data'], 'SQuAD 1.1 required')
    index, articles, contexts = {}, set(), set()
    for article in data['data']:
        c.keys(article, ('title', 'paragraphs'), 'article'); title = article['title']
        c.text(title, 'article title', True); c.require(title not in articles, 'duplicate article title')
        articles.add(title); c.require(type(article['paragraphs']) is list and article['paragraphs'], 'paragraphs required')
        for paragraph in article['paragraphs']:
            c.keys(paragraph, ('context', 'qas'), 'paragraph'); context = paragraph['context']
            c.text(context, 'context', True); contexts.add(c.sha256(context.encode('utf-8')))
            c.require(type(paragraph['qas']) is list and paragraph['qas'], 'questions required')
            for qa in paragraph['qas']:
                c.keys(qa, ('id', 'question', 'answers'), 'question')
                qid = qa['id']; c.text(qid, 'question ID', True)
                c.require(qid == qid.strip() and qid not in index, 'duplicate/invalid question ID')
                c.text(qa['question'], 'question', True)
                c.require(type(qa['answers']) is list and qa['answers'], 'v1.1 gold answers required')
                for answer in qa['answers']:
                    c.keys(answer, ('text', 'answer_start'), 'gold answer')
                    c.text(answer['text'], 'gold text', True); c.integer(answer['answer_start'], 'gold start', 0)
                    start = answer['answer_start']
                    c.require(context[start:start + len(answer['text'])] == answer['text'], 'gold offset/text mismatch')
                index[qid] = dict(article_id=title, context=context, question=qa['question'], answers=qa['answers'])
    return data, index, {'questions': sorted(index), 'articles': sorted(articles), 'context_hashes': sorted(contexts)}


def validate_inputs(blobs, policy):
    """Full supplied training-source/split coverage before any fit or sampling."""
    policy_check(policy); c.keys(blobs, INPUTS, 'input bytes')
    source, source_index, _ = dataset_index(blobs['source_train'])
    splits = {role: dataset_index(blobs[role]) for role in ROLES}
    seen = {key: set() for key in ('questions', 'articles', 'context_hashes')}
    union = {}
    for role, (_, index, membership) in splits.items():
        for key in seen:
            current = set(membership[key]); c.require(not current & seen[key], 'cross-role ' + key + ' overlap')
            seen[key].update(current)
        for qid, item in index.items():
            c.require(qid in source_index and item == source_index[qid], 'split/source identity mismatch')
        union.update(index)
    c.require(union == source_index, 'full training source coverage required')
    training, validation = team.prepare_pair(blobs['train_candidates'], blobs['validation_candidates'], group='full')
    for role, batch in (('reranker_train', training), ('internal_val', validation)):
        index = splits[role][1]
        c.require(set(batch['audit']['membership']['questions']) == set(index), 'candidate/split coverage mismatch')
        for q in batch['audit']['retained_origins']:
            item = index[q['question_id']]
            c.require(all(q[k] == item[k] for k in ('article_id', 'context', 'question')), 'candidate/source identity mismatch')
    baseline = c.decode(blobs['validation_baseline'])
    team.adapt_export(blobs['validation_candidates'], group='basic', baseline=baseline)
    return splits, training, validation, baseline


def checked_file(entry, base):
    c.keys(entry, ('path', 'sha256'), 'file descriptor'); digest(entry['sha256'])
    c.text(entry['path'], 'file path', True)
    path = (base / entry['path']).resolve(); h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    c.require(h.hexdigest() == entry['sha256'], 'artifact hash mismatch: ' + str(path))
    return path


def load_manifest(path):
    path = Path(path); manifest = c.decode(path.read_bytes())
    c.keys(manifest, ('schema', 'status', 'policy', 'inputs', 'scorer', 'lineage', 'baseline_generation'), 'run manifest')
    c.require(manifest['schema'] == SCHEMA, 'run schema mismatch')
    c.require(manifest['status'] in ('synthetic_fixture', 'real_provisional'), 'run status required')
    policy_check(manifest['policy']); c.keys(manifest['inputs'], INPUTS, 'manifest inputs')
    blobs = {}
    for key, entry in manifest['inputs'].items():
        raw = checked_file(entry, path.parent).read_bytes()
        c.require(c.sha256(raw) == entry['sha256'], 'input changed while reading')
        blobs[key] = raw
    c.keys(manifest['baseline_generation'], GENERATION, 'baseline declared generation')
    validation_metadata = c.decode(blobs['validation_candidates'])['metadata']
    for key in GENERATION:
        supplied, expected = manifest['baseline_generation'][key], validation_metadata[key]
        c.require(type(supplied) is type(expected) and supplied == expected, 'baseline generation mismatch: ' + key)
    lineage = manifest['lineage']
    if manifest['status'] == 'synthetic_fixture':
        c.require(lineage == {'status': 'synthetic_no_checkpoint'}, 'synthetic lineage required')
    else:
        c.keys(lineage, ('status', 'relationship_reference', 'checkpoint_export_value',
                        'checkpoint_files', 'tokenizer_files', 'run_info', 'train_log', 'environment'), 'real lineage')
        c.require(lineage['status'] == 'caller_attested_lineage_not_origin_authentication', 'real lineage attestation required')
        for key in ('relationship_reference', 'checkpoint_export_value'):
            c.text(lineage[key], key, True); c.require(bool(lineage[key].strip()), 'unresolved lineage')
        for group in ('checkpoint_files', 'tokenizer_files'):
            c.require(type(lineage[group]) is dict and lineage[group], 'lineage files required')
            for entry in lineage[group].values(): checked_file(entry, path.parent)
        for key in ('run_info', 'train_log', 'environment'): checked_file(lineage[key], path.parent)
        for key in ('train_candidates', 'validation_candidates'):
            c.require(c.decode(blobs[key])['metadata']['checkpoint'] == lineage['checkpoint_export_value'], 'checkpoint declaration mismatch')
    c.keys(manifest['scorer'], ('source_status', 'source_reference', 'hashes'), 'scorer identity')
    scorer = manifest['scorer']
    c.require(scorer['source_status'] == 'repository_bytes_course_origin_unverified', 'scorer status must be explicit')
    c.text(scorer['source_reference'], 'scorer reference', True)
    c.keys(scorer['hashes'], SOURCE_FILES, 'scorer files')
    for name, expected in scorer['hashes'].items():
        digest(expected); c.require(c.sha256((ROOT / name).read_bytes()) == expected, 'scorer source hash mismatch')
    return manifest, blobs


def load_scorer():
    """Only fixed, manifest-verified repository wrapper; no injected code path."""
    spec = importlib.util.spec_from_file_location('_reranker_team_scorer', ROOT / 'scripts/squad_eval.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.evaluate


def score(evaluate, dataset, predictions):
    ids = {q['id'] for a in dataset['data'] for p in a['paragraphs'] for q in p['qas']}
    c.keys(predictions, ids, 'prediction coverage')
    result = evaluate(dataset['data'], predictions)
    for key in ('exact_per_question', 'f1_per_question'):
        c.keys(result[key], ids, 'scorer coverage')
        for value in result[key].values():
            c.number(value, 'scorer value'); c.require(0 <= value <= 1, 'scorer target units')
    c.require(result['total'] == len(ids), 'scorer total mismatch')
    return result


def targets_from_training(evaluate, dataset, records):
    targets = {q['question_id']: {} for q in records}
    for rank in range(max(len(q['candidates']) for q in records)):
        predictions = {q['question_id']: q['candidates'][min(rank, len(q['candidates']) - 1)]['answer'] for q in records}
        result = score(evaluate, dataset, predictions)
        for q in records:
            if rank < len(q['candidates']):
                targets[q['question_id']][q['candidates'][rank]['candidate_id']] = result['f1_per_question'][q['question_id']]
    return targets


def state_bytes(model):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.json'; model.save(path)
        return path.read_bytes()


def reload_predict(directory, group, expected_binding_sha256, candidate_bytes):
    """Label-free new batches; an external binding pin is mandatory, never fit."""
    c.require(group in ('basic', 'full'), 'feature group required'); digest(expected_binding_sha256)
    directory = Path(directory); raw = (directory / (group + '-binding.json')).read_bytes()
    c.require(c.sha256(raw) == expected_binding_sha256, 'binding hash mismatch')
    binding = c.decode(raw)
    c.keys(binding, ('schema', 'status', 'profile', 'producer_ref', 'features', 'alpha', 'policy',
                     'generation', 'exclusions', 'inputs_sha256', 'targets_sha256', 'scorer',
                     'lineage', 'model_sha256', 'software_sha256'), 'model binding')
    c.require(binding['schema'] == BINDING and binding['status'] in ('synthetic_fixture', 'real_provisional'), 'model schema/status')
    c.require(binding['profile'] == team.PROFILE and binding['producer_ref'] == team.PRODUCER_REF, 'adapter version mismatch')
    policy_check(binding['policy']); c.require(binding['features'] == c.feature_config(group), 'feature mismatch')
    c.number(binding['alpha'], 'selected alpha')
    c.require(binding['alpha'] in binding['policy']['alphas'], 'selected alpha mismatch')
    c.keys(binding['inputs_sha256'], INPUTS, 'saved input hashes')
    for value in binding['inputs_sha256'].values(): digest(value)
    digest(binding['targets_sha256'])
    c.keys(binding['software_sha256'], SOFTWARE_FILES, 'software identity')
    for name, expected in binding['software_sha256'].items():
        digest(expected); c.require(c.sha256((ROOT / name).read_bytes()) == expected, 'software identity mismatch')
    c.keys(binding['generation'], GENERATION, 'generation identity')
    c.keys(binding['exclusions'], ('questions', 'articles', 'context_hashes'), 'training exclusions')
    batch = team.adapt_export(candidate_bytes, group=group)
    c.require(batch['audit']['metadata']['split'] == 'internal_val', 'inference role forbidden')
    for key in GENERATION:
        supplied, expected = batch['audit']['metadata'][key], binding['generation'][key]
        c.require(type(supplied) is type(expected) and supplied == expected, 'generation mismatch: ' + key)
    for key, excluded in binding['exclusions'].items():
        c.require(type(excluded) is list and all(type(x) is str for x in excluded) and len(excluded) == len(set(excluded)), 'malformed exclusions')
        c.require(not set(excluded) & set(batch['audit']['membership'][key]), 'training overlap: ' + key)
    model_path = directory / (group + '-model.json'); model_raw = model_path.read_bytes()
    digest(binding['model_sha256']); c.require(c.sha256(model_raw) == binding['model_sha256'], 'model hash mismatch')
    model = Reranker.load(model_path)
    config = binding['features']
    c.require(model.features.to_dict() == {k: config[k] for k in ('basic', 'compatibility', 'group')}, 'core feature configuration mismatch')
    c.require(model.alpha == binding['alpha'] and model.no_signal == 'error', 'core policy mismatch')
    questions = questions_from_records(batch['questions']); predictions = model.predict(questions)
    c.require(state_bytes(model) == model_raw, 'inference changed serialized state')
    return predictions, model.scores(questions)


def run(manifest_path, output):
    start = time.monotonic(); manifest, blobs = load_manifest(manifest_path)
    policy = manifest['policy']; splits, training, validation, baseline = validate_inputs(blobs, policy)
    evaluate = load_scorer()
    targets = targets_from_training(evaluate, splits['reranker_train'][0], training['questions'])
    # Validation gold is used only for metrics/selection after each training-only fit.
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    def save(name, value):
        with (output / name).open('xb') as handle: handle.write(c.encode(value))
    save('manifest.json', manifest); save('targets.json', targets)
    save('training-first-answers-derived.json', {q['question_id']: q['candidates'][0]['answer'] for q in training['questions']})
    save('baseline-predictions.json', baseline)
    baseline_metrics = score(evaluate, splits['internal_val'][0], baseline)
    report = {'schema': SCHEMA, 'status': manifest['status'], 'policy': policy,
              'origin_authenticated': False, 'course_scorer_origin_verified': False,
              'scorer': manifest['scorer'], 'lineage': manifest['lineage'],
              'training_baseline_status': 'derived_first_candidates_not_independent_baseline',
              'baseline_generation': manifest['baseline_generation'],
              'baseline_parity_scope': 'declared_settings_and_exact_first_answers_not_run_authentication',
              'input_sha256': {k: c.sha256(v) for k, v in blobs.items()},
              'split_counts': {role: len(value[1]) for role, value in splits.items()},
              'baseline': baseline_metrics, 'variants': {}}
    exclusions = {key: sorted(set(splits['bert_train'][2][key]) | set(splits['reranker_train'][2][key]))
                  for key in ('questions', 'articles', 'context_hashes')}
    for group in ('basic', 'full'):
        train = team.adapt_export(blobs['train_candidates'], group=group)
        valid = team.adapt_export(blobs['validation_candidates'], group=group)
        config = train['feature_registry']; trials = []; best = None
        train_questions = questions_from_records(train['questions']); val_questions = questions_from_records(valid['questions'])
        for order, alpha in enumerate(policy['alphas']):
            model = Reranker(FeatureConfig(config['basic'], config['compatibility'], group), alpha=alpha, no_signal='error')
            model.fit(train_questions, targets); before = state_bytes(model)
            predictions = model.predict(val_questions); scores = model.scores(val_questions)
            metrics = score(evaluate, splits['internal_val'][0], predictions)
            c.require(state_bytes(model) == before, 'validation changed preprocessing/model')
            trials.append({'alpha': alpha, 'exact': metrics['exact'], 'f1': metrics['f1'], 'total': metrics['total']})
            rank = (metrics['f1'], metrics['exact'], -order)
            if best is None or rank > best[0]: best = (rank, alpha, before, predictions, scores, metrics)
        _, alpha, model_raw, predictions, scores, metrics = best
        with (output / (group + '-model.json')).open('xb') as handle: handle.write(model_raw)
        binding = {'schema': BINDING, 'status': manifest['status'], 'profile': team.PROFILE,
                   'producer_ref': team.PRODUCER_REF, 'features': config, 'alpha': alpha, 'policy': policy,
                   'generation': {k: training['audit']['metadata'][k] for k in GENERATION},
                   'exclusions': exclusions, 'inputs_sha256': report['input_sha256'],
                   'targets_sha256': c.sha256(c.encode(targets)), 'scorer': manifest['scorer'],
                   'lineage': manifest['lineage'], 'model_sha256': c.sha256(model_raw),
                   'software_sha256': {name: c.sha256((ROOT / name).read_bytes()) for name in SOFTWARE_FILES}}
        save(group + '-binding.json', binding); pin = c.sha256(c.encode(binding))
        reloaded_predictions, reloaded_scores = reload_predict(output, group, pin, blobs['validation_candidates'])
        c.require(predictions == reloaded_predictions and scores == reloaded_scores, 'reload equivalence failed')
        save(group + '-predictions.json', predictions)
        report['variants'][group] = {'selected_alpha': alpha, 'trials': trials, 'metrics': metrics,
                                    'binding_sha256': pin, 'reload_equivalence': 'exact',
                                    'delta_em': metrics['exact'] - baseline_metrics['exact'],
                                    'delta_f1': metrics['f1'] - baseline_metrics['f1']}
    report['wall_seconds'] = time.monotonic() - start
    save('complete.json', report)  # Only a successful complete run gets this receipt.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv); result = run(args.manifest, args.output)
    print(json.dumps({'status': result['status'], 'split_counts': result['split_counts'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
