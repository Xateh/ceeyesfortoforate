"""Dev requires explicit opt-in and retains binding/overlap/model checks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_real_training import make_inputs
from test_team_candidates import fixture
from offline_handoff import contract as c
from reranker import real_training as r
from reranker.core import Reranker

class SubmissionInferenceTests(unittest.TestCase):
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
