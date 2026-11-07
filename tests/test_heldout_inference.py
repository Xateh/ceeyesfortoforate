"""New label-free synthetic batches through a trained, reloaded reranker."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from offline_handoff import contract as c
from reranker.core import Reranker
from reranker.integration import IDENTITY_FIELDS
from reranker.evaluation import preflight_predictions
from integration_fixtures import bundle, dataset

FILES = ('contexts.json', 'candidates.json', 'questions.json', 'provenance.json', 'split-manifest.json')


def inference(group='basic', prefix='unseen', source=None):
    raw, data = source if source is not None else bundle(group, prefix, validation=True)
    original = c.decode(raw)
    artifacts = {name:data[name] for name in FILES}
    manifest = {'schema_id':'cs4248.label-free-inference/0.1.0-draft', 'status':c.SYNTHETIC,
                'execution_role':'internal_validation', 'dataset_id':prefix,
                'identity':{name:original[name] for name in IDENTITY_FIELDS},
                'artifact_sha256':{name:c.sha256(value) for name,value in artifacts.items()}}
    return c.encode(manifest), artifacts


def repack(pair, name, value):
    raw, artifacts = copy.deepcopy(pair)
    artifacts[name] = c.encode(value)
    manifest = c.decode(raw)
    manifest['artifact_sha256'][name] = c.sha256(artifacts[name])
    return c.encode(manifest), artifacts


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('reranker.heldout'), 'held-out boundary is not implemented')
        self.h = importlib.import_module('reranker.heldout')
        self.i = importlib.import_module('offline_handoff.inference')

    def test_new_batches_after_reload_freeze_state_and_preserve_text(self):
        for group in ('basic', 'full'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                training = bundle(group)
                model = self.h.HeldoutRun(training).fit()
                path = Path(tmp)/'model.json'
                model.save(path)
                before = path.read_bytes()
                with patch.object(Reranker, 'fit', side_effect=AssertionError('reload must not fit')):
                    loaded = self.h.HeldoutRun.load(path, training)
                    for prefix in ('unseen', 'later'):
                        batch = inference(group, prefix)
                        original = copy.deepcopy(batch)
                        with (patch.object(c, 'validate_targets', side_effect=AssertionError('inference targets')),
                             patch.object(c, 'validate_label_audit', side_effect=AssertionError('inference gold'))):
                            predicted = loaded.predict(batch)
                            scores = loaded.scores(batch)
                            selected = loaded.select(batch)
                        self.assertEqual(predicted, {f'{prefix}-tie-q':'  Thé 😀!  ',
                            f'{prefix}-shift-q':'Other', f'{prefix}-single-q':' The, Ω 😀 '})
                        self.assertEqual(scores[0][0], scores[0][1])
                        records = c.decode(batch[1]['questions.json'])
                        self.assertEqual(selected[f'{prefix}-tie-q'].candidate_id, records[0]['candidates'][0]['candidate_id'])
                        self.assertEqual(model.scores(batch), scores)
                        self.assertEqual(batch, original)
                        out = Path(tmp)/f'{prefix}.json'
                        receipt = loaded.save_predictions(out, batch)
                        self.assertIn('😀'.encode(), out.read_bytes())
                        self.assertEqual(preflight_predictions(dataset(predicted), out.read_bytes()), predicted)
                        self.assertEqual(receipt['inference_manifest_sha256'], c.sha256(batch[0]))
                        self.assertEqual(receipt['artifact_sha256'], c.decode(batch[0])['artifact_sha256'])
                        self.assertFalse(receipt['production_valid'])
                loaded.save(path)
                self.assertEqual(path.read_bytes(), before)

    def test_training_registry_scaling_and_no_signal_contract(self):
        training = bundle('full')
        run = self.h.HeldoutRun(training).fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.json'
            run.save(path)
            saved = c.decode(path.read_bytes())
            rows = c.vectors(c.decode(training[1]['questions.json']), c.feature_config('full'))
            matrix = torch.tensor([v for question in rows for v in question], dtype=torch.float64)
            self.assertEqual(saved['model']['state']['mean'], matrix.mean(0).tolist())
            scale = matrix.std(0, correction=0)
            scale[scale == 0] = 1
            self.assertEqual(saved['model']['state']['scale'], scale.tolist())
            self.assertEqual(saved['model']['state']['training_questions'], 4)
            self.assertEqual(saved['model']['state']['informative_questions'], 1)
            self.assertEqual(saved['binding']['identity']['feature_registry'], c.feature_config('full'))
            self.assertEqual(saved['binding']['training_manifest_sha256'], c.sha256(training[0]))
            with self.assertRaises(ValueError):
                run.fit(alpha=-1)
            run.save(path)
            self.assertEqual(c.decode(path.read_bytes()), saved)
        no_signal = self.h.HeldoutRun(bundle(no_signal=True))
        with self.assertRaisesRegex(ValueError, 'no within-question target signal'):
            no_signal.fit()
        with self.assertRaisesRegex(ValueError, 'not fitted'):
            no_signal.predict(inference())

    def test_roles_and_real_provenance_refused_before_execution(self):
        run = self.h.HeldoutRun(bundle()).fit()
        with patch.object(Reranker, 'fit', side_effect=AssertionError('forbidden fit')):
            for role in ('internal_validation', 'official_dev', 'bert_train', 'synthetic', None):
                with self.subTest(role=role), self.assertRaises(c.ContractError):
                    run.fit(role=role)
        for role in ('reranker_train', 'bert_train', 'official_dev', 'synthetic', None):
            raw, artifacts = inference()
            manifest = c.decode(raw)
            manifest['execution_role'] = role
            self.refused(run, (c.encode(manifest), artifacts))
        raw, artifacts = inference()
        manifest = c.decode(raw)
        manifest['status'] = 'production'
        self.refused(run, (c.encode(manifest), artifacts))

    def refused(self, run, batch):
        with patch.object(Reranker, 'scores', side_effect=AssertionError('invalid input reached model')) as score:
            with self.assertRaises(c.ContractError):
                run.predict(batch)
            score.assert_not_called()

    def test_label_files_and_inline_fields_rejected(self):
        run = self.h.HeldoutRun(bundle()).fit()
        for name in ('targets.json', 'label-audit.json', 'mapping-illustration.json', 'gold.json'):
            raw, artifacts = inference()
            artifacts[name] = object()  # Must reject keys, never decode forbidden values.
            self.refused(run, (raw, artifacts))
        for name, field in (('questions.json','gold'), ('candidates.json','target'), ('contexts.json','gold')):
            pair = inference()
            data = c.decode(pair[1][name]); data[0][field] = 'leak'
            self.refused(run, repack(pair, name, data))
        pair = inference(); questions = c.decode(pair[1]['questions.json'])
        questions[0]['candidates'][0]['features']['gold_f1'] = 1
        self.refused(run, repack(pair, 'questions.json', questions))
        pair = inference(); manifest = c.decode(pair[0]); manifest['gold'] = 'leak'
        self.refused(run, (c.encode(manifest), pair[1]))

    def test_identity_schema_and_hash_mismatches_rejected(self):
        run = self.h.HeldoutRun(bundle()).fit()
        for key in ('schema_id','contract_id','feature_spec_id','core_commit','checkpoint_sha256','generator_commit','feature_code_commit'):
            raw, artifacts = inference(); manifest = c.decode(raw)
            manifest['identity'][key] = 'different'
            self.refused(run, (c.encode(manifest), artifacts))
        for change in ('order','registry_spec','tokenizer','policy','hash','outer_schema','missing'):
            raw, artifacts = inference(); manifest = c.decode(raw)
            if change == 'order': manifest['identity']['feature_registry']['basic'].reverse()
            elif change == 'registry_spec': manifest['identity']['feature_registry']['spec_id'] = 'other'
            elif change == 'tokenizer': manifest['identity']['tokenizer_id_and_revision']['revision'] = 'different'
            elif change == 'policy': manifest['identity']['candidate_policy']['max_span_tokens'] = 29
            elif change == 'hash': artifacts['questions.json'] += b' '
            elif change == 'outer_schema': manifest['schema_id'] = c.SCHEMA_ID
            else: artifacts.pop('provenance.json')
            self.refused(run, (c.encode(manifest), artifacts))
        self.refused(run, inference('full'))

    def test_question_order_features_spans_and_coverage_rejected(self):
        run = self.h.HeldoutRun(bundle()).fit()
        for change in ('question_order','candidate_order','feature','boolean','nonfinite','missing_id'):
            pair = inference(); data = c.decode(pair[1]['questions.json'])
            if change == 'question_order': data.reverse()
            elif change == 'candidate_order': data[0]['candidates'].reverse()
            elif change == 'missing_id': data.pop()
            else: data[0]['candidates'][0]['features']['bert_start_logit'] = {'feature':999, 'boolean':True, 'nonfinite':float('inf')}[change]
            if change == 'nonfinite':
                import json
                raw, artifacts = pair; artifacts = dict(artifacts)
                artifacts['questions.json'] = json.dumps(data).encode()
                manifest = c.decode(raw); manifest['artifact_sha256']['questions.json'] = c.sha256(artifacts['questions.json'])
                altered = (c.encode(manifest), artifacts)
            else: altered = repack(pair, 'questions.json', data)
            self.refused(run, altered)
        for change in ('span','baseline'):
            pair = inference(); data = c.decode(pair[1]['candidates.json'])
            if change == 'span': data[0]['occurrences'][0]['end'] -= 1
            else: data[0]['baseline']['answer'] = 'unvalidated'
            self.refused(run, repack(pair, 'candidates.json', data))

    def test_training_question_article_and_content_overlap_rejected(self):
        training = bundle(); run = self.h.HeldoutRun(training).fit()
        self.refused(run, inference(source=training))
        for kind in ('article','article_content','context_content'):
            def rename(value):
                if type(value) is dict: return {rename(k):rename(v) for k,v in value.items()}
                if type(value) is list: return [rename(v) for v in value]
                if type(value) is str and 'train-' in value:
                    return value if kind == 'article' and value.endswith('-a') else value.replace('train-', 'other-')
                return value
            pair = inference(source=training)
            for name in FILES:
                data = rename(c.decode(pair[1][name]))
                if kind == 'context_content' and name == 'split-manifest.json':
                    for article in data['articles']:
                        article['text'] += ' different article suffix'
                        article['sha256'] = c.sha256(article['text'].encode())
                pair = repack(pair, name, data)
            self.i.validate_bundle(*pair)  # Structurally valid; cross-training exclusion must reject.
            self.refused(run, pair)

    def test_saved_identity_core_hash_and_training_binding_checked(self):
        training = bundle(); run = self.h.HeldoutRun(training).fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.json'; run.save(path)
            original = c.decode(path.read_bytes())
            for change in ('binding','hash','order','fallback','count','transform'):
                altered = copy.deepcopy(original)
                if change == 'binding': altered['binding']['training_manifest_sha256'] = '0'*64
                elif change == 'hash': altered['model_sha256'] = '0'*64
                elif change == 'transform': altered['model']['state']['mean'][0] += 1
                else:
                    if change == 'order': altered['model']['features']['basic'].reverse()
                    elif change == 'fallback': altered['model']['no_signal'] = 'first'
                    else: altered['model']['state']['training_questions'] += 1
                    altered['model_sha256'] = c.sha256(c.encode(altered['model']))
                path.write_bytes(c.encode(altered))
                with self.subTest(change=change), self.assertRaises(c.ContractError):
                    self.h.HeldoutRun.load(path, training)
            path.write_bytes(c.encode(original))
            with self.assertRaises(c.ContractError):
                self.h.HeldoutRun.load(path, bundle(prefix='different-training'))

    def test_complete_prediction_ids_and_candidate_text_are_enforced(self):
        run = self.h.HeldoutRun(bundle()).fit(); batch = inference()
        correct = run.predict(batch)
        for malformed in ({k:v for k,v in correct.items() if k != 'unseen-tie-q'},
                          dict(correct, extra='unexpected'), dict(correct, **{'unseen-tie-q':'generated answer'})):
            with patch.object(Reranker, 'predict', return_value=malformed):
                with self.assertRaises(c.ContractError):
                    run.predict(batch)

    def test_malformed_training_and_bundle_container_refused(self):
        with self.assertRaises(c.ContractError):
            self.h.HeldoutRun(inference())
        run = self.h.HeldoutRun(bundle()).fit()
        for value in (None, {}, (), ('missing',), (b'{}', {})):
            self.refused(run, value)


if __name__ == '__main__':
    unittest.main()
