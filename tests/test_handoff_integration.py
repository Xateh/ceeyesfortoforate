"""Actual PR20 bundles through PR19 fit/score/predict and bound serialization."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from offline_handoff import contract as c
from reranker.core import Reranker
from reranker.integration import SyntheticRun
from reranker.evaluation import preflight_predictions
from integration_fixtures import assignments, bundle, dataset


def read_model(path):
    return c.decode(Path(path).read_bytes())


class IntegrationTests(unittest.TestCase):
    def test_actual_registry_train_transforms_predictions_ties_and_reload(self):
        for group in ('basic', 'full'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                supplied = assignments(group)
                before = copy.deepcopy(supplied)
                run = SyntheticRun(supplied).fit()
                path = Path(tmp) / 'model.json'
                run.save(path)
                saved = read_model(path)
                core = saved['model']
                names = list(c.BASIC + (c.COMPATIBILITY if group == 'full' else ()))
                self.assertEqual(core['features'], {k:v for k,v in c.feature_config(group).items() if k != 'spec_id'})
                self.assertEqual(saved['binding']['identity']['feature_registry'], c.feature_config(group))
                training = c.decode(supplied['reranker_train'][1]['questions.json'])
                rows = [[candidate['features'][name] for name in names]
                        for question in training for candidate in question['candidates']]
                matrix = torch.tensor(rows, dtype=torch.float64)
                expected_mean = matrix.mean(0).tolist()
                expected_scale = matrix.std(0, correction=0)
                expected_scale[expected_scale == 0] = 1
                self.assertEqual(core['state']['mean'], expected_mean)
                self.assertEqual(core['state']['scale'], expected_scale.tolist())
                self.assertEqual(core['state']['training_questions'], 4)
                self.assertEqual(core['state']['informative_questions'], 1)
                self.assertEqual(core['no_signal'], 'error')
                scores = run.scores()
                selected = run.select()
                predicted = run.predict()
                self.assertEqual(scores[0][0], scores[0][1])
                val = c.decode(supplied['internal_validation'][1]['questions.json'])
                baseline = val[0]['candidates'][0]['candidate_id']
                self.assertEqual(selected['validation-tie-q'].candidate_id, baseline)
                self.assertEqual(predicted['validation-tie-q'], '  Thé 😀!  ')
                self.assertEqual(predicted['validation-single-q'], ' The, Ω 😀 ')
                self.assertEqual(predicted['validation-shift-q'], 'Other')
                illustration = c.decode(supplied['internal_validation'][1]['mapping-illustration.json'])['mapping']
                self.assertNotEqual(predicted, illustration)
                self.assertEqual(set(predicted), {q['question_id'] for q in val})
                self.assertEqual(len(val[0]['candidates']), 2)
                provenance = c.decode(supplied['internal_validation'][1]['provenance.json'])
                self.assertEqual(len(provenance[0]['origins']), 3)
                winner = provenance[0]['selected'][0]
                self.assertEqual((winner['start_logit'], winner['end_logit'], winner['combined_score']), (2, 3, 5))
                output = Path(tmp) / 'predictions.json'
                run.save_predictions(output)
                raw = output.read_bytes()
                self.assertIn('😀'.encode(), raw)
                self.assertEqual(preflight_predictions(dataset([q['question_id'] for q in val]), raw), predicted)
                run.save(Path(tmp) / 'after.json')
                self.assertEqual(saved, read_model(Path(tmp) / 'after.json'))
                loaded = SyntheticRun.load(path, supplied)
                self.assertEqual(loaded.scores(), scores)
                self.assertEqual({k:v.candidate_id for k,v in loaded.select().items()},
                                 {k:v.candidate_id for k,v in selected.items()})
                self.assertEqual(loaded.predict(), predicted)
                self.assertEqual(before, supplied)

    def test_split_gates_run_before_core_execution(self):
        run = SyntheticRun(assignments())
        with patch.object(Reranker, 'fit', side_effect=AssertionError('must not fit')) as fit:
            for role in ('internal_validation', 'official_dev', 'bert_train', 'synthetic'):
                with self.subTest(role=role), self.assertRaises(c.ContractError):
                    run.fit(role=role)
            fit.assert_not_called()
        with patch.object(Reranker, 'scores', side_effect=AssertionError('must not score')):
            with self.assertRaises(c.ContractError):
                run.predict(role='official_dev')
        supplied = assignments()
        supplied['official_dev'] = supplied.pop('internal_validation')
        with self.assertRaises(c.ContractError):
            SyntheticRun(supplied)

    def test_overlap_fails_before_fit(self):
        supplied = assignments()
        supplied['internal_validation'] = supplied['reranker_train']
        with patch.object(Reranker, 'fit', side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(c.ContractError, 'overlap'):
                SyntheticRun(supplied).fit()

    def test_article_and_content_overlap_with_disjoint_question_ids(self):
        original, artifacts = bundle()
        for kind in ('articles', 'content'):
            def rename(value):
                if type(value) is dict:
                    return {rename(k):rename(v) for k,v in value.items()}
                if type(value) is list:
                    return [rename(v) for v in value]
                if type(value) is str and 'train-' in value:
                    if kind == 'articles' and value.endswith('-a'):
                        return value
                    return value.replace('train-', 'other-')
                return value
            cloned = {name:c.encode(rename(c.decode(raw))) for name,raw in artifacts.items()}
            manifest = c.decode(original)
            manifest['artifact_sha256'] = {name:c.sha256(raw) for name,raw in cloned.items()}
            manifest['split_manifest_sha256'] = c.sha256(cloned['split-manifest.json'])
            second = (c.encode(manifest), cloned)
            c.validate_bundle(*second)
            with self.subTest(kind=kind), self.assertRaisesRegex(c.ContractError, f'{kind} overlap'):
                SyntheticRun({'reranker_train':(original, artifacts), 'internal_validation':second})

    def test_equal_target_groups_omit_loss_but_keep_scaling_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = []
            for equal in (0.0, .8):
                run = SyntheticRun(assignments(equal=equal)).fit()
                path = Path(tmp) / str(equal)
                run.save(path)
                states.append(read_model(path)['model'])
                self.assertEqual(len(run.predict(role='reranker_train')), 4)
            self.assertEqual(states[0], states[1])

    def test_no_signal_default_refuses_and_failed_refit_preserves_model(self):
        run = SyntheticRun(assignments(no_signal=True))
        with self.assertRaisesRegex(ValueError, 'no within-question target signal'):
            run.fit()
        with self.assertRaisesRegex(ValueError, 'not fitted'):
            run.predict()
        run = SyntheticRun(assignments()).fit()
        before = run.predict()
        with patch.object(Reranker, 'fit', side_effect=ValueError('synthetic failed refit')):
            with self.assertRaises(ValueError):
                run.fit()
        self.assertEqual(run.predict(), before)

    def test_equal_width_spec_order_and_core_identity_tampering_rejected(self):
        supplied = assignments('full')
        for change in ('spec', 'order', 'group'):
            altered = copy.deepcopy(supplied)
            manifest = c.decode(altered['reranker_train'][0])
            if change == 'spec':
                manifest['feature_registry']['spec_id'] = 'candidate-features/other-draft'
            elif change == 'order':
                manifest['feature_registry']['basic'].reverse()
            else:
                manifest['feature_registry']['basic'][0], manifest['feature_registry']['compatibility'][0] = (
                    manifest['feature_registry']['compatibility'][0], manifest['feature_registry']['basic'][0])
            altered['reranker_train'] = (c.encode(manifest), altered['reranker_train'][1])
            with self.subTest(change=change), self.assertRaises(c.ContractError):
                SyntheticRun(altered).fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            SyntheticRun(supplied).fit().save(path)
            original = read_model(path)
            for change in ('binding_order', 'binding_spec', 'core_order', 'hash', 'fallback', 'provenance'):
                altered = copy.deepcopy(original)
                if change == 'binding_order':
                    altered['binding']['identity']['feature_registry']['basic'].reverse()
                elif change == 'binding_spec':
                    altered['binding']['identity']['feature_spec_id'] = 'candidate-features/other-draft'
                elif change == 'core_order':
                    altered['model']['features']['basic'].reverse()
                    altered['model_sha256'] = c.sha256(c.encode(altered['model']))
                elif change == 'hash':
                    altered['model_sha256'] = '0' * 64
                elif change == 'fallback':
                    altered['model']['no_signal'] = 'first'
                    altered['model_sha256'] = c.sha256(c.encode(altered['model']))
                else:
                    altered['binding']['bundles']['reranker_train']['manifest_sha256'] = '0' * 64
                path.write_bytes(c.encode(altered))
                with self.subTest(change=change), self.assertRaises(c.ContractError):
                    SyntheticRun.load(path, supplied)

    def test_validation_identity_must_match_training(self):
        supplied = assignments('basic')
        supplied['internal_validation'] = bundle('full', 'validation', validation=True)
        with self.assertRaises(c.ContractError):
            SyntheticRun(supplied)

    def test_predictions_fail_complete_id_preflight(self):
        run = SyntheticRun(assignments()).fit()
        actual = run.predict()
        expected = dataset(['validation-tie-q', 'validation-shift-q', 'validation-single-q'])
        for malformed in ({k:v for k,v in actual.items() if k != 'validation-tie-q'},
                          dict(actual, extra='injected')):
            with self.assertRaises(c.ContractError):
                preflight_predictions(expected, c.encode(malformed))


if __name__ == '__main__':
    unittest.main()
