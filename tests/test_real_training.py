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

    def test_producer_question_strip_accepts_only_source_boundary(self):
        pristine = copy.deepcopy(self.values)
        for prefix, suffix in ((' ', ''), ('', ' '), (' \t', '\n '), ('\u2003', '\u00a0')):
            self.values = copy.deepcopy(pristine)
            for role, source_pos in (('reranker_train', 1), ('internal_val', 2)):
                qa = self.values[role]['data'][0]['paragraphs'][0]['qas'][0]
                qa['question'] = prefix + qa['question'] + suffix
                self.values['source_train']['data'][source_pos] = copy.deepcopy(self.values[role]['data'][0])
            before = self.blobs()
            with self.subTest(prefix=prefix, suffix=suffix), patch.object(Reranker, 'fit') as fit:
                r.validate_inputs(before, policy())
                fit.assert_not_called()
            self.assertEqual(before, self.blobs())

    def test_raw_source_split_question_identity_remains_exact(self):
        self.values['internal_val']['data'][0]['paragraphs'][0]['qas'][0]['question'] += ' '
        with self.assertRaisesRegex(ValueError, 'split/source identity'):
            r.validate_inputs(self.blobs(), policy())

    def test_export_question_is_not_normalized_or_content_changed(self):
        pristine = copy.deepcopy(self.values)
        for change in (lambda q: ' ' + q, lambda q: q + ' ',
                       lambda q: q.replace(' ', '  ', 1), lambda q: q + ' changed'):
            self.values = copy.deepcopy(pristine)
            q = self.values['validation_candidates']['questions'][0]
            q['question'] = change(q['question'])
            with self.subTest(question=q['question']), self.assertRaisesRegex(ValueError, 'question preprocessing'):
                r.validate_inputs(self.blobs(), policy())

    def test_question_fix_keeps_article_and_context_identity_exact(self):
        pristine = copy.deepcopy(self.values)
        for field in ('article_id', 'context'):
            self.values = copy.deepcopy(pristine)
            q = self.values['validation_candidates']['questions'][0]; q[field] += ' '
            q['context_id'] = c.sha256(json.dumps([q['article_id'], q['context']], ensure_ascii=False).encode())
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'candidate/source identity'):
                r.validate_inputs(self.blobs(), policy())

    def enable_provisional(self):
        archive = self.path / 'fixture-archive.bin'
        archive.write_bytes(b'Invented test bytes, not a team release or run record.')
        descriptor = {'path': archive.name, 'sha256': c.sha256(archive.read_bytes())}
        self.manifest['status'] = 'real_provisional'
        self.manifest['lineage'] = {
            'schema': r.PROVENANCE_SCHEMA, 'status': r.PROVISIONAL,
            'approval_reference': 'synthetic test of explicit provisional authorization',
            'evidence_references': ['synthetic fixture declaration, not production evidence'],
            'checkpoint_export_value': self.values['train_candidates']['metadata']['checkpoint'],
            'archives': {k: dict(descriptor) for k in ('data', 'checkpoint', 'candidates')},
            'original_records': {k: 'unavailable' for k in ('run_info', 'train_log', 'environment')},
            'unverified': {k: 'unverified' for k in r.UNVERIFIED}}
        settings = {k: self.values['validation_candidates']['metadata'][k]
                    for k in ('checkpoint', 'max_answer_length', 'n_best', 'batch_size')}
        self.manifest['baseline_generation'] = {
            'status': 'supplied_predictions_execution_unverified', 'declared_settings': settings,
            'unverified_fields': [k for k in r.BASELINE_FIELDS if k not in settings],
            'evidence_references': ['synthetic test declaration only']}
        self.write_manifest()

    def test_provisional_requires_optin_and_persists_uncertainties(self):
        self.enable_provisional()
        for optin in (False, 1):
            with self.assertRaisesRegex(ValueError, 'opt-in'):
                r.load_manifest(self.manifest_path, allow_provisional_provenance=optin)
        report = r.run(self.manifest_path, self.path / 'out', allow_provisional_provenance=True)
        self.assertEqual(report['status'], 'real_provisional')
        self.assertEqual(report['provenance_mode'], 'explicit_provisional')
        self.assertFalse(report['origin_authenticated'])
        for group in ('basic', 'full'):
            binding = c.decode((self.path / 'out' / (group + '-binding.json')).read_bytes())
            self.assertEqual(binding['lineage'], self.manifest['lineage'])
            self.assertEqual(binding['provisional_baseline_generation'], self.manifest['baseline_generation'])
            pin = report['variants'][group]['binding_sha256']; raw = c.encode(fixture('internal_val', 'unseen'))
            with self.assertRaisesRegex(ValueError, 'opt-in'):
                r.reload_predict(self.path / 'out', group, pin, raw)
            with patch.object(Reranker, 'fit', side_effect=AssertionError('reload must not fit')):
                pred, _ = r.reload_predict(self.path / 'out', group, pin, raw, allow_provisional_provenance=True)
            self.assertEqual(set(pred), {'unseen-q'})

    def test_provisional_honesty_settings_and_archive_guards(self):
        self.enable_provisional(); pristine = copy.deepcopy(self.manifest)
        for kind in ('record', 'verified', 'unknown', 'extra', 'boolean', 'setting', 'approval', 'archive', 'checkpoint', 'synthetic'):
            self.manifest = copy.deepcopy(pristine); lineage = self.manifest['lineage']; baseline = self.manifest['baseline_generation']
            if kind == 'record': lineage['original_records']['run_info'] = {'path': 'invented.json'}
            elif kind == 'verified': lineage['unverified']['training_lineage'] = 'verified'
            elif kind == 'unknown': baseline['unverified_fields'].pop()
            elif kind == 'extra': baseline['declared_settings']['unknown'] = 1
            elif kind == 'boolean': baseline['declared_settings']['max_answer_length'] = True
            elif kind == 'setting': baseline['declared_settings']['max_answer_length'] += 1
            elif kind == 'approval': lineage['approval_reference'] = ' '
            elif kind == 'archive': lineage['archives']['checkpoint']['sha256'] = '0' * 64
            elif kind == 'checkpoint': lineage['checkpoint_export_value'] = 'different'
            else: self.manifest['status'] = 'synthetic_fixture'
            self.write_manifest()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.load_manifest(self.manifest_path, allow_provisional_provenance=True)

    def test_provisional_flag_does_not_weaken_strict_mode(self):
        self.manifest['status'] = 'real_provisional'; self.write_manifest()
        for optin in (False, True):
            with self.subTest(optin=optin), patch.object(Reranker, 'fit') as fit:
                with self.assertRaises(ValueError):
                    r.run(self.manifest_path, self.path / 'out', allow_provisional_provenance=optin)
                fit.assert_not_called()
        self.assertFalse((self.path / 'out').exists())

    def test_provisional_keeps_input_hash_identity_and_parity_gates(self):
        self.enable_provisional(); pristine_values = copy.deepcopy(self.values); pristine = copy.deepcopy(self.manifest)
        for kind in ('hash', 'baseline', 'question', 'gold', 'coverage'):
            self.values = copy.deepcopy(pristine_values); self.manifest = copy.deepcopy(pristine)
            if kind == 'baseline': self.values['validation_baseline']['validation-q'] = 'wrong'
            elif kind == 'question': self.values['validation_candidates']['questions'][0]['question'] += ' changed'
            elif kind == 'gold': self.values['internal_val']['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] += 1
            elif kind == 'coverage': self.values['source_train']['data'].pop()
            self.refresh()
            if kind == 'hash': self.manifest['inputs']['train_candidates']['sha256'] = '0' * 64; self.write_manifest()
            with self.subTest(kind=kind), patch.object(Reranker, 'fit') as fit:
                with self.assertRaises(ValueError):
                    r.run(self.manifest_path, self.path / 'out', allow_provisional_provenance=True)
                fit.assert_not_called()
            self.assertFalse((self.path / 'out').exists())

    def test_provisional_reload_rejects_relabelled_evidence(self):
        self.enable_provisional(); r.run(self.manifest_path, self.path / 'out', allow_provisional_provenance=True)
        path = self.path / 'out/basic-binding.json'; original = path.read_bytes()
        for kind in ('lineage', 'baseline', 'unknown', 'checkpoint', 'setting'):
            binding = c.decode(original)
            if kind == 'lineage': binding['lineage']['unverified']['training_lineage'] = 'verified'
            elif kind == 'baseline': binding['provisional_baseline_generation']['status'] = 'verified'
            elif kind == 'checkpoint': binding['lineage']['checkpoint_export_value'] = 'different'
            elif kind == 'setting': binding['provisional_baseline_generation']['declared_settings']['max_answer_length'] += 1
            else: binding['provisional_baseline_generation']['unverified_fields'].pop()
            path.write_bytes(c.encode(binding))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                r.reload_predict(self.path / 'out', 'basic', c.sha256(path.read_bytes()),
                                 c.encode(fixture('internal_val', 'unseen')), allow_provisional_provenance=True)

    def test_cli_provisional_optin_is_explicit(self):
        for optin in (False, True):
            argv = ['--manifest', str(self.manifest_path), '--output', str(self.path / 'out')]
            if optin: argv.append('--allow-provisional-provenance')
            with self.subTest(optin=optin), patch.object(r, 'run', return_value={'status':'test','split_counts':{}}) as run:
                r.main(argv)
                self.assertIs(run.call_args.kwargs['allow_provisional_provenance'], optin)

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
