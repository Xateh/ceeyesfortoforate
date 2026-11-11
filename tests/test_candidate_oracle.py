"""Hand-checked oracle metrics and input-boundary tests; no BERT/data download."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from offline_handoff import contract as c
from scripts.candidate_oracle import diagnose
from scripts.squad_eval import evaluate
from test_team_candidates import fixture


def inputs():
    export = fixture(split='internal_val', tag='val')
    q = export['questions'][0]
    dataset = {'version': '1.1', 'data': [{'title': q['article_id'], 'paragraphs': [
        {'context': q['context'], 'qas': [{'id': q['question_id'], 'question': '  ' + q['question'],
         'answers': [{'text': 'Paris', 'answer_start': q['context'].index('Paris')}]}]}]}]}
    return dataset, export, {'val-q': 'Bob'}


def run(data):
    return diagnose(*(c.encode(x) for x in data))


class CandidateOracleTests(unittest.TestCase):
    def test_later_exact_candidate_and_curve(self):
        data = inputs(); saved = copy.deepcopy(data)
        report = run(data); s = report['summary']
        self.assertEqual(s['baseline'], {'exact': 0.0, 'f1': 0.0})
        self.assertEqual(s['oracle'], {'exact': 100.0, 'f1': 100.0})
        self.assertEqual(s['recoverable_em_errors'], 1)
        self.assertEqual(s['unrecoverable_em_errors'], 0)
        self.assertEqual(s['recoverable_fraction_of_em_errors'], 1)
        self.assertEqual([r['exact'] for r in report['oracle_by_k']], [0, 0, 100])
        self.assertEqual(report['per_question']['val-q']['first_exact_rank'], 3)
        self.assertEqual(data, saved)

    def test_partial_match_without_exact_is_unrecoverable_em(self):
        data = inputs(); qa = data[0]['data'][0]['paragraphs'][0]['qas'][0]
        context = data[1]['questions'][0]['context']
        qa['answers'] = [{'text': 'Bob in Paris', 'answer_start': context.index('Bob')}]
        s = run(data)['summary']
        self.assertEqual(s['oracle']['exact'], 0)
        self.assertAlmostEqual(s['oracle']['f1'], 50)
        self.assertEqual(s['recoverable_em_errors'], 0)
        self.assertEqual(s['unrecoverable_em_errors'], 1)

    def test_multiple_golds_and_shared_normalization(self):
        data = inputs(); q = data[1]['questions'][0]
        qa = data[0]['data'][0]['paragraphs'][0]['qas'][0]
        # Maximum over golds makes the first Bob candidate correct.
        qa['answers'] = [{'text': 'Bob', 'answer_start': q['context'].index('Bob')},
                         {'text': 'Paris', 'answer_start': q['context'].index('Paris')}]
        report = run(data)
        expected = evaluate(data[0]['data'], data[2])
        self.assertEqual(report['summary']['baseline'], {k: expected[k] for k in ('exact', 'f1')})
        self.assertEqual(report['summary']['baseline_em_errors'], 0)
        self.assertIsNone(report['summary']['recoverable_fraction_of_em_errors'])
        self.assertEqual(report['per_question']['val-q']['first_exact_rank'], 1)
        qa['answers'] = [{'text': 'val.', 'answer_start': q['context'].index('val.')}]
        span = q['candidates'][2]; span.update(answer='val', start=q['context'].index('val.'),
                                             end=q['context'].index('val.') + 3)
        self.assertEqual(run(data)['summary']['oracle']['exact'], 100)

    def test_normalized_empty_gold_uses_repository_fallback(self):
        data = inputs(); q = data[1]['questions'][0]
        start = q['context'].index('a')
        data[0]['data'][0]['paragraphs'][0]['qas'][0]['answers'] = [
            {'text': 'a', 'answer_start': start}]
        q['candidates'][2].update(answer='a', start=start, end=start + 1)
        report = run(data)
        self.assertEqual(report['summary']['oracle'], {'exact': 100, 'f1': 100})
        self.assertEqual(report['per_question']['val-q']['first_exact_rank'], 3)

    def test_short_groups_keep_full_denominator(self):
        data = inputs(); other = fixture(split='internal_val', tag='second')['questions'][0]
        other['candidates'] = other['candidates'][:1]
        other['candidates'][0]['window_index'] = 99
        data[1]['questions'].append(other)
        data[0]['data'].append({'title': other['article_id'], 'paragraphs': [
            {'context': other['context'], 'qas': [{'id': other['question_id'], 'question': other['question'],
             'answers': [{'text': 'Bob', 'answer_start': other['context'].index('Bob')}]}]}]})
        data[2]['second-q'] = 'Bob'
        s = run(data)['summary']
        self.assertEqual(s['total'], 2)
        self.assertEqual(s['baseline']['exact'], 50)
        self.assertEqual(s['oracle']['exact'], 100)
        self.assertEqual(s['candidate_count_histogram'], {1: 1, 3: 1})

    def test_wrong_role_coverage_identity_and_baseline_rejected(self):
        for kind in ('role', 'missing', 'extra', 'duplicate', 'question', 'context', 'baseline'):
            with self.subTest(kind=kind):
                data = inputs(); paragraph = data[0]['data'][0]['paragraphs'][0]
                if kind == 'role': data[1]['metadata']['split'] = 'reranker_train'
                elif kind == 'missing': paragraph['qas'][0]['id'] = 'wrong'
                elif kind == 'extra': data[2]['extra'] = 'Bob'
                elif kind == 'duplicate': paragraph['qas'].append(copy.deepcopy(paragraph['qas'][0]))
                elif kind == 'question': paragraph['qas'][0]['question'] = 'different?'
                elif kind == 'context': paragraph['context'] += ' extra'
                elif kind == 'baseline': data[2]['val-q'] = 'Ada'
                with self.assertRaises(ValueError): run(data)

    def test_cli_report_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = [sys.executable, '-m', 'scripts.candidate_oracle']
            for key, value in zip(('dataset', 'candidates', 'baseline'), inputs()):
                path = root / (key + '.json'); path.write_bytes(c.encode(value))
                args.extend(['--' + key, str(path)])
            output = root / 'report.json'; args.extend(['--output', str(output)])
            subprocess.run(args, check=True, capture_output=True)
            original = output.read_bytes(); report = json.loads(original)
            self.assertEqual(report['input_sha256']['dataset'], c.sha256((root / 'dataset.json').read_bytes()))
            self.assertTrue(report['baseline_first_candidate_parity'])
            self.assertFalse(report['course_scorer_equivalence_verified'])
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(original, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
