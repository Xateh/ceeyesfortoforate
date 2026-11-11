"""Dev requires explicit opt-in and retains binding/overlap/model checks."""
import tempfile
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from test_real_training import make_inputs
from test_team_candidates import fixture
from offline_handoff import contract as c
from reranker import real_training as r
from reranker.core import Reranker
from reranker.reload_compatibility import compatible_reload_source, LEGACY_LOADER_SHA256, CORRECTED_LOADER_SHA256

class SubmissionInferenceTests(unittest.TestCase):
    def crlf_model(self, root):
        make_inputs(root)
        r.run(root/'manifest.json', root/'models')
        model_path = root/'models/basic-model.json'
        raw = model_path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        model_path.write_bytes(raw)
        binding_path = root/'models/basic-binding.json'
        binding = c.decode(binding_path.read_bytes())
        binding['model_sha256'] = c.sha256(raw)
        binding_path.write_bytes(c.encode(binding))
        return model_path, raw, c.sha256(binding_path.read_bytes()), c.encode(fixture('internal_val', 'unseen'))

    def test_crlf_artifact_reload_on_native_platform_keeps_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model_path, raw, pin, candidates = self.crlf_model(root)
            self.assertIn(b'\r\n', raw)
            native = r.state_bytes(Reranker.load(model_path))
            if os.linesep == '\n':
                self.assertNotEqual(native, raw)  # Actual Linux regression, not a Windows simulation.
                self.assertNotIn(b'\r\n', native)
            with patch.object(Reranker, 'fit', side_effect=AssertionError('reload must not fit')):
                predictions, _ = r.reload_predict(root/'models', 'basic', pin, candidates)
            self.assertEqual(set(predictions), {'unseen-q'})
            self.assertEqual(model_path.read_bytes(), raw)

    def test_crlf_artifact_tampering_still_fails_model_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model_path, raw, pin, candidates = self.crlf_model(root)
            changed = json.loads(raw); changed['state']['coef'][0] += 1
            for altered in (raw.replace(b'\r\n', b'\n'), c.encode(changed)):
                with self.subTest(altered_sha256=c.sha256(altered)):
                    model_path.write_bytes(altered)
                    with self.assertRaisesRegex(ValueError, 'model hash mismatch'):
                        r.reload_predict(root/'models', 'basic', pin, candidates)

    def test_prediction_or_score_mutation_still_fails_state_check(self):
        for method_name in ('predict', 'scores'):
            with self.subTest(method=method_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, _, pin, candidates = self.crlf_model(root)
                original = getattr(Reranker, method_name)
                def mutate(model, questions):
                    answer = original(model, questions)
                    model._state['coef'][0] += 1
                    return answer
                with patch.object(Reranker, method_name, mutate):
                    with self.assertRaisesRegex(ValueError, 'inference changed serialized state'):
                        r.reload_predict(root/'models', 'basic', pin, candidates)

    def test_source_compatibility_accepts_only_exact_audited_loader_transition(self):
        actual = c.sha256((r.ROOT/'reranker/real_training.py').read_bytes())
        self.assertEqual(actual, CORRECTED_LOADER_SHA256)
        self.assertTrue(compatible_reload_source('reranker/real_training.py', LEGACY_LOADER_SHA256, actual))
        for name, expected, supplied in [('reranker/core.py', LEGACY_LOADER_SHA256, actual),
                                         ('reranker/real_training.py', '0'*64, actual),
                                         ('reranker/real_training.py', LEGACY_LOADER_SHA256, '0'*64)]:
            self.assertFalse(compatible_reload_source(name, expected, supplied))

    def test_dev_optin_reload_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); make_inputs(root)
            result=r.run(root/'manifest.json',root/'models')
            for group in ('basic','full'):
                pin=result['variants'][group]['binding_sha256']
                export=fixture('internal_val','unseen'); export['metadata']['split']='dev'
                raw=c.encode(export)
                with self.assertRaises(ValueError): r.reload_predict(root/'models',group,pin,raw)
                with patch.object(Reranker,'fit',side_effect=AssertionError('inference must not fit')):
                    answers,_=r.reload_predict(root/'models',group,pin,raw,split='dev')
                self.assertEqual(set(answers),{'unseen-q'})
                with self.assertRaises(ValueError): r.reload_predict(root/'models',group,pin,raw,split='reranker_train')
                overlapping=fixture(); overlapping['metadata']['split']='dev'
                with self.assertRaisesRegex(ValueError,'training overlap'):
                    r.reload_predict(root/'models',group,pin,c.encode(overlapping),split='dev')
                with self.assertRaisesRegex(ValueError,'binding hash'):
                    r.reload_predict(root/'models',group,'0'*64,raw,split='dev')

if __name__=='__main__': unittest.main()
