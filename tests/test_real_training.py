"""Invented data through the actual checked-in scorer; no release/data download."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_handoff import contract as c
from reranker import real_training as r
from reranker.core import Reranker
from reranker.team_model import TeamCandidateModel
from test_team_candidates import fixture


def policy():
    return {'id': 'test-policy', 'decision_reference': 'synthetic:test-only',
            'acceptance': 'explicit_run_policy_not_team_agreement',
            'features': {g: c.feature_config(g) for g in ('basic', 'full')},
            'alphas': [1.0, 0.1], 'target': r.TARGET, 'no_signal': 'error',
            'selection': 'internal_val_f1_then_em_then_first_alpha'}


def squad(export):
    q = export['questions'][0]; answer = q['candidates'][-1]['answer']
    return {'version': '1.1', 'data': [{'title': q['article_id'], 'paragraphs': [
        {'context': q['context'], 'qas': [{'id': q['question_id'], 'question': q['question'],
          'answers': [{'text': answer, 'answer_start': q['context'].index(answer)}]}]}]}]}


def make_inputs(path):
    train = fixture(); valid = fixture('internal_val', 'validation'); bert = fixture(tag='bert')
    q = valid['questions'][0]; q['context'] = q['context'].replace('Paris', 'París')
    q['candidates'][-1]['answer'] = 'París'
    q['context_id'] = c.sha256(json.dumps([q['article_id'], q['context']], ensure_ascii=False).encode())
    splits = {'bert_train': squad(bert), 'reranker_train': squad(train), 'internal_val': squad(valid)}
    source = {'version': '1.1', 'data': [copy.deepcopy(v['data'][0]) for v in splits.values()]}
    values = dict(source_train=source, **splits, train_candidates=train, validation_candidates=valid,
                  validation_baseline={'validation-q': 'Bob'})
    manifest = {'schema': r.SCHEMA, 'status': 'synthetic_fixture', 'policy': policy(), 'inputs': {},
                'lineage': {'status': 'synthetic_no_checkpoint'},
                'baseline_generation': {k: valid['metadata'][k] for k in r.GENERATION},
                'scorer': {'source_status': 'repository_bytes_course_origin_unverified',
                           'source_reference': 'repository:main-d11a5ea',
                           'hashes': {f: c.sha256((r.ROOT / f).read_bytes()) for f in r.SOURCE_FILES}}}
    for key, value in values.items():
        raw = c.encode(value); (path / (key + '.json')).write_bytes(raw)
        manifest['inputs'][key] = {'path': key + '.json', 'sha256': c.sha256(raw)}
    (path / 'manifest.json').write_bytes(c.encode(manifest))
    return manifest, values


class RealTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name); self.manifest, self.values = make_inputs(self.path)
        self.manifest_path = self.path / 'manifest.json'

    def write_manifest(self):
        self.manifest_path.write_bytes(c.encode(self.manifest))

    def blobs(self):
        return {k: c.encode(v) for k, v in self.values.items()}

    def test_end_to_end_train_only_selection_unicode_and_reload(self):
        original = {p.name: p.read_bytes() for p in self.path.iterdir()}
        fit = Reranker.fit; calls = []
        def checked(model, questions, targets):
            self.assertEqual({q.question_id for q in questions}, {'train-q'})
            self.assertEqual(set(targets), {'train-q'}); calls.append(model.features.group)
            return fit(model, questions, targets)
        with patch.object(Reranker, 'fit', checked):
            report = r.run(self.manifest_path, self.path / 'out')
        self.assertEqual(calls, ['basic', 'basic', 'full', 'full'])
        self.assertEqual(report['status'], 'synthetic_fixture')
        for group in ('basic', 'full'):
            item = report['variants'][group]
            self.assertEqual(item['selected_alpha'], 1.0)
            self.assertEqual(item['metrics']['f1'], 100.0)
            raw = (self.path / 'out' / (group + '-predictions.json')).read_bytes()
            self.assertIn('París'.encode(), raw)
            model = c.decode((self.path / 'out' / (group + '-model.json')).read_bytes())
            self.assertAlmostEqual(model['state']['mean'][0], 10 / 3)
            with patch.object(Reranker, 'fit', side_effect=AssertionError('reload must not fit')):
                answers, _ = r.reload_predict(self.path / 'out', group, item['binding_sha256'], c.encode(fixture('internal_val', 'new')))
            self.assertEqual(set(answers), {'new-q'})
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.path.iterdir() if p.is_file()})
        with self.assertRaises(FileExistsError): r.run(self.manifest_path, self.path / 'out')

    def refresh(self):
        for key, value in self.values.items():
            raw = c.encode(value); (self.path / (key + '.json')).write_bytes(raw)
            self.manifest['inputs'][key]['sha256'] = c.sha256(raw)
        self.write_manifest()

    def test_full_identity_coverage_overlap_and_baseline_before_fit(self):
        pristine = copy.deepcopy(self.values)
        for kind in ('missing', 'extra', 'duplicate', 'article', 'context', 'question', 'gold', 'baseline', 'role', 'overlap'):
            self.values = copy.deepcopy(pristine)
            q = self.values['validation_candidates']['questions'][0]
            if kind == 'missing': self.values['source_train']['data'].pop()
            elif kind == 'extra': self.values['source_train']['data'].append(squad(fixture(tag='extra'))['data'][0])
            elif kind == 'duplicate': self.values['internal_val']['data'][0]['paragraphs'][0]['qas'] *= 2
            elif kind in ('article', 'context', 'question'): q[{'article':'article_id', 'context':'context', 'question':'question'}[kind]] += ' changed'
            elif kind == 'gold': self.values['internal_val']['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] += 1
            elif kind == 'baseline': self.values['validation_baseline']['unknown'] = 'Bob'
            elif kind == 'role': self.values['train_candidates']['metadata']['split'] = 'internal_val'
            else: self.values['internal_val'] = copy.deepcopy(self.values['reranker_train'])
            with self.subTest(kind=kind), patch.object(Reranker, 'fit') as fit:
                with self.assertRaises(ValueError): r.validate_inputs(self.blobs(), policy())
                fit.assert_not_called()

    def test_manifest_hash_policy_scorer_and_real_lineage_fail_closed(self):
        with self.assertRaises(ValueError): r.load_manifest(r.ROOT / 'configs/reranker-real.example.json')
        original = copy.deepcopy(self.manifest)
        for kind in ('hash', 'scorer', 'basic', 'spec', 'bool', 'nonfinite', 'fallback', 'lineage', 'baseline-settings', 'baseline-type'):
            self.manifest = copy.deepcopy(original)
            if kind == 'hash': self.manifest['inputs']['train_candidates']['sha256'] = '0' * 64
            elif kind == 'scorer': self.manifest['scorer']['hashes']['evaluate-v2.0.py'] = '0' * 64
            elif kind == 'basic': self.manifest['policy']['features']['basic']['basic'][0] = 'q_where'
            elif kind == 'spec': self.manifest['policy']['features']['full']['spec_id'] = 'changed'
            elif kind == 'bool': self.manifest['policy']['alphas'] = [True]
            elif kind == 'nonfinite':
                with self.assertRaises(ValueError): r.policy_check(dict(policy(), alphas=[float('inf')]))
                continue
            elif kind == 'fallback': self.manifest['policy']['no_signal'] = 'first'
            elif kind == 'lineage': self.manifest['status'] = 'real_provisional'
            elif kind == 'baseline-type': self.manifest['baseline_generation']['max_context_only'] = 0
            else: self.manifest['baseline_generation']['max_answer_length'] = 1
            self.write_manifest()
            with self.subTest(kind=kind), patch.object(Reranker, 'fit') as fit:
                with self.assertRaises(ValueError): r.run(self.manifest_path, self.path / 'out')
                fit.assert_not_called()
            self.assertFalse((self.path / 'out').exists())

    def test_actual_repository_scorer_semantics_and_target_units(self):
        evaluate = r.load_scorer(); data = copy.deepcopy(self.values['reranker_train'])
        qa = data['data'][0]['paragraphs'][0]['qas'][0]
        qa['answers'].append({'text': 'Ada', 'answer_start': 2})
        self.assertEqual(r.score(evaluate, data, {'train-q':'the PARIS!'})['exact'], 100)
        self.assertEqual(r.score(evaluate, data, {'train-q':'Ada'})['f1'], 100)
        qa['answers'] = [{'text': 'Ada met', 'answer_start': 2}]
        self.assertAlmostEqual(r.score(evaluate, data, {'train-q':'Ada'})['f1_per_question']['train-q'], 2 / 3)
        qa['answers'] = [{'text': 'a', 'answer_start': 4}]
        self.assertEqual(r.score(evaluate, data, {'train-q':''})['f1'], 100)
        with self.assertRaises(ValueError): r.score(evaluate, data, {})
        _, training, _, _ = r.validate_inputs(self.blobs(), policy())
        targets = r.targets_from_training(evaluate, self.values['reranker_train'], training['questions'])
        self.assertEqual(list(targets['train-q'].values()), [0, 0, 1])

    def test_no_signal_fails_and_synthetic_guard_remains(self):
        self.values['train_candidates']['questions'][0]['candidates'] = self.values['train_candidates']['questions'][0]['candidates'][:1]
        self.refresh()
        with self.assertRaises(ValueError): r.run(self.manifest_path, self.path / 'out')
        self.assertFalse((self.path / 'out' / 'complete.json').exists())
        with self.assertRaises(ValueError):
            TeamCandidateModel.fit_synthetic(c.encode(self.values['train_candidates']), {}, group='basic', target_status='real_provisional')

    def test_equal_zero_and_single_groups_retain_coverage(self):
        for tag in ('zero', 'single'):
            export = fixture(tag=tag); q = export['questions'][0]
            for candidate in q['candidates']: candidate['window_index'] += 100 if tag == 'zero' else 200
            if tag == 'single': q['candidates'] = q['candidates'][:1]
            data = squad(export)
            if tag == 'zero':
                qa = data['data'][0]['paragraphs'][0]['qas'][0]
                qa['answers'] = [{'text':'zero', 'answer_start':q['context'].index('zero')}]
            self.values['train_candidates']['questions'].append(q)
            self.values['reranker_train']['data'].extend(copy.deepcopy(data['data']))
            self.values['source_train']['data'].extend(copy.deepcopy(data['data']))
        self.refresh(); r.run(self.manifest_path, self.path / 'out')
        targets = c.decode((self.path / 'out/targets.json').read_bytes())
        self.assertEqual(set(targets), {'train-q', 'zero-q', 'single-q'})
        self.assertEqual(set(targets['zero-q'].values()), {0})
        self.assertEqual(set(targets['single-q'].values()), {1})
        state = c.decode((self.path / 'out/basic-model.json').read_bytes())['state']
        self.assertEqual(state['training_questions'], 3)
        self.assertEqual(state['informative_questions'], 1)
        self.assertAlmostEqual(state['mean'][0], 24 / 7)

    def test_exact_ties_after_reload_preserve_export_order(self):
        qa = self.values['reranker_train']['data'][0]['paragraphs'][0]['qas'][0]
        qa['answers'] = [{'text':'Ada', 'answer_start':2}]
        self.values['source_train']['data'][1] = copy.deepcopy(self.values['reranker_train']['data'][0])
        self.refresh(); report = r.run(self.manifest_path, self.path / 'out')
        new = fixture('internal_val', 'unseen')
        for group in ('basic', 'full'):
            pin = report['variants'][group]['binding_sha256']
            pred, scores = r.reload_predict(self.path / 'out', group, pin, c.encode(new))
            self.assertEqual(pred['unseen-q'], 'Bob'); self.assertEqual(scores[0][0], scores[0][1])
            reverse = copy.deepcopy(new); reverse['questions'][0]['candidates'][:2] = list(reversed(reverse['questions'][0]['candidates'][:2]))
            pred, _ = r.reload_predict(self.path / 'out', group, pin, c.encode(reverse))
            self.assertEqual(pred['unseen-q'], 'Ada')

    def test_reload_rejects_role_overlap_and_repinned_feature_change(self):
        report = r.run(self.manifest_path, self.path / 'out'); pin = report['variants']['basic']['binding_sha256']
        for kind in ('role', 'overlap', 'label', 'generation'):
            new = fixture('internal_val', 'new')
            if kind == 'role': new['metadata']['split'] = 'reranker_train'
            elif kind == 'overlap': new['questions'][0]['question_id'] = 'train-q'
            elif kind == 'label': new['questions'][0]['gold'] = 'Paris'
            else: new['metadata']['checkpoint'] = 'different'
            with self.subTest(kind=kind), patch.object(Reranker, 'fit', side_effect=AssertionError):
                with self.assertRaises(ValueError): r.reload_predict(self.path / 'out', 'basic', pin, c.encode(new))
        path = self.path / 'out/basic-binding.json'; original = path.read_bytes()
        for field in ('software_sha256', 'generation'):
            binding = c.decode(original)
            if field == 'software_sha256': binding[field]['reranker/core.py'] = '0' * 64
            else: binding[field]['max_context_only'] = 0
            path.write_bytes(c.encode(binding))
            with self.assertRaises(ValueError):
                r.reload_predict(self.path / 'out', 'basic', c.sha256(path.read_bytes()), c.encode(fixture('internal_val', 'new')))
        binding = c.decode(original)
        binding['features']['basic'] = list(reversed(binding['features']['basic']))
        path.write_bytes(c.encode(binding))
        for expected in (pin, c.sha256(path.read_bytes())):
            with self.assertRaises(ValueError):
                r.reload_predict(self.path / 'out', 'basic', expected, c.encode(fixture('internal_val', 'new')))


if __name__ == '__main__':
    unittest.main()
