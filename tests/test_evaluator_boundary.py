"""Evaluator transport/preflight only; no supplied evaluator execution."""
import unittest
from pathlib import Path
from unittest.mock import Mock
from offline_handoff import contract as c
from reranker.evaluation import preflight_predictions, invoke_evaluator, INVOCATION
from integration_fixtures import dataset


class EvaluatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset(['one', 'two'])
        self.predictions = c.encode({'one':'  Thé 😀!  ', 'two':''})
        self.source = b'synthetic test double sentinel; not executable scorer source\n'
        self.provenance = dict(status='synthetic_test_double', source_reference='synthetic:test-double',
                               expected_sha256=c.sha256(self.source), version='synthetic-only',
                               invocation_revision=INVOCATION)

    def test_exact_coverage_and_text(self):
        self.assertEqual(preflight_predictions(self.data, self.predictions), {'one':'  Thé 😀!  ', 'two':''})

    def test_bad_ids_values_and_json_fail_before_dispatch(self):
        for raw in (c.encode({'one':'missing'}), c.encode({'one':'', 'two':'', 'extra':''}),
                    c.encode({'one':False, 'two':''}), b'{"one":"x","one":"y","two":""}'):
            runner = Mock()
            with self.subTest(raw=raw), self.assertRaises(c.ContractError):
                invoke_evaluator(self.data, raw, provenance=self.provenance, source_bytes=self.source, runner=runner)
            runner.assert_not_called()
        for ids in (['one', 'one'], [], ['one', '']):
            with self.assertRaises(c.ContractError):
                preflight_predictions(dataset(ids), self.predictions)

    def test_unresolved_or_incorrect_provenance_refuses(self):
        for provenance in (None, dict(self.provenance, status='source_preview_only'),
                           dict(self.provenance, expected_sha256='0' * 64),
                           dict(self.provenance, invocation_revision='override-na-threshold'),
                           dict(self.provenance, source_reference='')):
            runner = Mock()
            with self.subTest(provenance=provenance), self.assertRaises(c.ContractError):
                invoke_evaluator(self.data, self.predictions, provenance=provenance,
                                 source_bytes=self.source, runner=runner)
            runner.assert_not_called()
        with self.assertRaises(c.ContractError):
            invoke_evaluator(self.data, self.predictions)

    def test_explicit_synthetic_double_dispatch_is_not_conformance(self):
        runner = Mock(return_value={'synthetic_receipt':'called; no metrics computed'})
        result = invoke_evaluator(self.data, self.predictions, provenance=self.provenance,
                                  source_bytes=self.source, runner=runner)
        runner.assert_called_once_with(self.source, self.data, self.predictions)
        self.assertEqual(result['status'], 'synthetic_dispatch_only')
        self.assertFalse(result['official_scorer_conformance'])
        self.assertEqual(result['result'], {'synthetic_receipt':'called; no metrics computed'})

    def test_preview_expectations_are_labelled_unexecuted(self):
        path = Path(__file__).parent / 'fixtures' / 'evaluator-preview-expectations.json'
        expectations = c.decode(path.read_bytes())
        self.assertEqual(expectations['status'], 'preview_derived_expectations_not_executable_conformance')
        self.assertIsNone(expectations['official_source_sha256'])
        self.assertFalse(expectations['executed_against_supplied_evaluator'])
        self.assertEqual(len(expectations['cases']), 6)
        self.assertEqual(expectations['cases'][1]['expected_f1'], .5)
        self.assertEqual(expectations['cases'][2]['expected_f1'], 1)
        self.assertEqual(expectations['cases'][3]['expected_f1'], 0)


if __name__ == '__main__':
    unittest.main()
