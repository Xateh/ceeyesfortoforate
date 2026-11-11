"""Frozen official-dev inference boundaries, using invented data only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline_handoff import contract as c
from reranker.core import FeatureConfig, Reranker, questions_from_records
from reranker.official_dev import adapt_official_dev, membership, predict_frozen, source_records
from reranker.team_candidates import adapt_export
from test_team_candidates import fixture

ROOT = Path(__file__).resolve().parents[1]
GENERATION = ('max_answer_length', 'n_best', 'max_candidates', 'batch_size',
              'device', 'max_context_only', 'tie_break')


def dataset(export):
    q = export['questions'][0]
    return {'version': '1.1', 'data': [{'title': q['article_id'], 'paragraphs': [
        {'context': q['context'], 'qas': [{'id': q['question_id'], 'question': q['question'],
          'answers': [{'text': 'Bob', 'answer_start': q['context'].index('Bob')}]}]}]}]}


class OfficialDevTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture('dev', 'official')
        self.expected = source_records(dataset(self.data))
        self.excluded = membership(source_records(dataset(fixture())))
        self.generation = {k: self.data['metadata'][k] for k in GENERATION}

    def adapt(self, data=None, expected=None, excluded=None):
        return adapt_official_dev(c.encode(self.data if data is None else data),
                                  expected=self.expected if expected is None else expected,
                                  excluded=self.excluded if excluded is None else excluded,
                                  generation=self.generation)

    def test_separate_dev_path_preserves_unicode_ties_and_original_role_boundary(self):
        original = copy.deepcopy(self.data)
        result = self.adapt()
        self.assertEqual(self.data, original)
        self.assertEqual(result['metadata']['split'], 'dev')
        self.assertEqual(result['baseline'], {'official-q': 'Bob'})
        self.assertEqual([c['answer'] for c in result['questions'][0]['candidates']], ['Bob', 'Ada', 'Paris'])
        with self.assertRaises(ValueError): adapt_export(c.encode(self.data), group='basic')

    def test_source_records_do_not_depend_on_annotations(self):
        data = dataset(self.data); a = source_records(data)
        data['data'][0]['paragraphs'][0]['qas'][0]['answers'] = [{'text': 'invented other annotation'}]
        self.assertEqual(source_records(data), a)

    def test_role_coverage_identity_and_generation_mismatches_rejected(self):
        for kind in ('role', 'missing', 'duplicate', 'question', 'context', 'span', 'labels', 'generation'):
            with self.subTest(kind=kind):
                data = copy.deepcopy(self.data)
                if kind == 'role': data['metadata']['split'] = 'internal_val'
                elif kind == 'missing': data['questions'] = []
                elif kind == 'duplicate': data['questions'].append(copy.deepcopy(data['questions'][0]))
                elif kind == 'question': data['questions'][0]['question'] += ' changed'
                elif kind == 'context': data['questions'][0]['context'] += ' changed'
                elif kind == 'span': data['questions'][0]['candidates'][0]['answer'] = 'not a slice'
                elif kind == 'labels': data['questions'][0]['candidates'][0]['gold'] = 'Bob'
                elif kind == 'generation': data['metadata']['max_answer_length'] = 31
                with self.assertRaises(ValueError): self.adapt(data)

    def test_each_source_overlap_is_rejected(self):
        own = membership(self.expected)
        for key in own:
            excluded = {k: set(v) for k, v in self.excluded.items()}
            excluded[key].update(own[key])
            with self.assertRaises(ValueError): self.adapt(excluded=excluded)

    def fit_model(self, path):
        records = self.adapt()['questions']
        q = questions_from_records(records)
        targets = {q[0].question_id: {v.candidate_id: float(i == 0) for i, v in enumerate(q[0].candidates)}}
        model = Reranker(FeatureConfig(c.BASIC), alpha=0.1).fit(q, targets)
        model.save(path)
        return {'id': 'basic', 'model': {'path': str(path), 'sha256': c.sha256(path.read_bytes())},
                'features': list(model.features.names), 'alpha': 0.1, 'eval_top_k': 20}

    def test_predict_never_fits_or_changes_model_and_checks_pin_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'; spec = self.fit_model(path)
            before = path.read_bytes()
            with patch.object(Reranker, 'fit', side_effect=AssertionError('no fitting on dev')):
                result = predict_frozen(path, spec, self.adapt()['questions'])
            self.assertEqual(set(result), {'official-q'})
            self.assertEqual(path.read_bytes(), before)
            for key, value in [('features', ['different']), ('alpha', 1.0), ('eval_top_k', 5)]:
                bad = dict(spec, **{key: value})
                with self.assertRaises(ValueError): predict_frozen(path, bad, self.adapt()['questions'])
            spec['model']['sha256'] = '0' * 64
            with self.assertRaises(ValueError): predict_frozen(path, spec, self.adapt()['questions'])

    def test_frozen_cli_saves_complete_comparison_and_refuses_changed_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(name, value):
                p = root / name; p.write_bytes(c.encode(value))
                return {'path': str(p), 'sha256': c.sha256(p.read_bytes())}
            spec = self.fit_model(root / 'model.json')
            source = save('source.json', dataset(fixture()))
            manifest = save('manifest.json', {'inputs': {'source_train': source}, 'software_sha256': {}})
            selection = save('selection.json', {'manifest_sha256': manifest['sha256']})
            freeze = save('freeze.json', {'status': 'frozen_before_local_official_dev_inference',
                'selection_results': selection, 'data': save('dev.json', dataset(self.data)),
                'models': [spec], 'generation': self.generation,
                'baseline': {'checkpoint_files': {}, 'checkpoint_path': self.data['metadata']['checkpoint']},
                'tokenized_dev_files': {}, 'exporter': save('exporter.json', {}), 'evaluation_history': 'synthetic'})
            candidates = save('candidates.json', self.data)
            command = [sys.executable, str(ROOT / 'scripts/evaluate_official_dev.py'),
                       '--freeze', freeze['path'], '--expected-freeze-sha256', freeze['sha256'],
                       '--candidates', candidates['path'], '--output-dir', str(root / 'output')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            complete = c.decode((root / 'output/complete.json').read_bytes())
            self.assertEqual(len(complete['results']), 2)
            self.assertTrue(all(v['total'] == 1 for v in complete['results']))
            self.assertTrue(complete['checks']['all_predictions_saved_before_scoring'])
            Path(freeze['path']).write_bytes(b'{}')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
