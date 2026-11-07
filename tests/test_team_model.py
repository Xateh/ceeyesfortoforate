"""Synthetic PR28 persistence tests; no BERT, real data or scorer execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_handoff import contract as c
from reranker.core import Reranker
from reranker import team_candidates as team
from reranker.team_model import TeamCandidateModel, TARGET_STATUS
from test_team_candidates import fixture, labels


def fit(group='basic'):
    raw = c.encode(fixture())
    return raw, TeamCandidateModel.fit_synthetic(raw, labels(raw), group=group, target_status=TARGET_STATUS)


def heldout(tag):
    data = fixture('internal_val',tag); q=data['questions'][0]
    q['context']=q['context'].replace('Bob','Bób'); q['candidates'][0]['answer']='Bób'
    q['context_id']=c.sha256(json.dumps([q['article_id'],q['context']],ensure_ascii=False).encode())
    data['metadata']['features_path']='synthetic/new-windows/'+tag
    data['metadata']['examples_path']='synthetic/new-source/'+tag
    return c.encode(data)


class TeamModelTests(unittest.TestCase):
    def test_both_groups_reload_without_fit_accepts_new_batches_and_frozen_state(self):
        for group in ('basic','full'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                train,model=fit(group); path=Path(tmp)/'model.json'; pin=model.save(path)
                original=c.decode(path.read_bytes()); first,second=heldout('first'),heldout('second')
                self.assertNotEqual(c.sha256(first),c.sha256(second))
                expected_scores=model.scores(first); expected_selection=model.select(first)
                with patch.object(Reranker,'fit',side_effect=AssertionError('reload/predict must not fit')):
                    loaded=TeamCandidateModel.load(path,train,expected_sha256=pin)
                    self.assertEqual(loaded.scores(first),expected_scores)
                    self.assertEqual(loaded.select(first),expected_selection)
                    for raw,tag in [(first,'first'),(second,'second')]:
                        predictions,receipt=loaded.predict_with_receipt(raw)
                        self.assertEqual(predictions,{tag+'-q':'Bób'})
                        self.assertIn('Bób'.encode(),c.encode(predictions))
                        self.assertEqual(receipt['inference_export_sha256'],c.sha256(raw))
                        self.assertEqual(receipt['training_export_sha256'],c.sha256(train))
                        self.assertFalse(receipt['origin_authenticated'])
                    re_saved=Path(tmp)/'restored.json'; loaded.save(re_saved)
                self.assertEqual(c.decode(re_saved.read_bytes()),original)
                config=original['binding']['identity']['feature_registry']
                self.assertEqual(config,c.feature_config(group))
                self.assertNotIn('targets',original); self.assertNotIn('targets',original['binding'])
                self.assertEqual(original['training_targets_sha256'],c.sha256(c.encode(labels(train))))

    def test_exact_score_tie_follows_supplied_order_before_and_after_reload(self):
        for group in ('basic', 'full'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                train, model = fit(group)
                path = Path(tmp)/'model.json'; pin = model.save(path)
                raw = heldout('tie')
                reversed_data = c.decode(raw)
                spans = reversed_data['questions'][0]['candidates']
                spans[:2] = reversed(spans[:2])
                reversed_raw = c.encode(reversed_data)
                with patch.object(Reranker, 'fit', side_effect=AssertionError('reload must not fit')):
                    loaded = TeamCandidateModel.load(path, train, expected_sha256=pin)
                    for batch in (raw, reversed_raw):
                        records = team.adapt_export(batch, group=group)['questions'][0]
                        first = records['candidates'][0]
                        self.assertEqual(records['candidates'][0]['features'],
                                         records['candidates'][1]['features'])
                        for run in (model, loaded):
                            scores = run.scores(batch)[0]
                            self.assertEqual(scores[0], scores[1])
                            self.assertEqual(scores[0], max(scores))
                            selected = run.select(batch)['tie-q']
                            self.assertEqual(selected.candidate_id, first['candidate_id'])
                            self.assertEqual(run.predict(batch), {'tie-q': first['answer']})
                        self.assertEqual(loaded.scores(batch), model.scores(batch))

    def test_independent_pin_detects_joint_core_and_binding_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; pin=model.save(path)
            data=c.decode(path.read_bytes()); data['model']['state']['coef'][0]+=1
            data['model_sha256']=c.sha256(c.encode(data['model']))
            data['binding']['training_metadata']['checkpoint']='substitute'
            path.write_bytes(c.encode(data))
            with self.assertRaisesRegex(ValueError,'artifact pin mismatch'):
                TeamCandidateModel.load(path,train,expected_sha256=pin)
            with self.assertRaises(ValueError): TeamCandidateModel.load(path,train,expected_sha256='bad')

    def test_changed_training_bytes_rejected_even_when_records_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; pin=model.save(path)
            for changed in (train+b'\n',c.encode(fixture(tag='another'))):
                with self.assertRaisesRegex(ValueError,'binding mismatch'):
                    TeamCandidateModel.load(path,changed,expected_sha256=pin)

    def test_repinned_malformed_versions_registry_binding_and_core_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit('full'); path=Path(tmp)/'model.json'; model.save(path); original=c.decode(path.read_bytes())
            kinds=('schema','status','adapter','producer','core_commit','candidate_version','boolean_version',
                   'spec','feature_order','group','generation','membership','extra_binding',
                   'core_features','core_hash','negative_scale','training_count','fallback','target_hash')
            for kind in kinds:
                with self.subTest(kind=kind):
                    data=copy.deepcopy(original); binding=data['binding']; identity=binding['identity']
                    if kind=='schema': data['schema_id']='other'
                    elif kind=='status': data['status']='production'
                    elif kind=='adapter': identity['adapter_profile']='other'
                    elif kind=='producer': identity['inspected_producer_ref']='0'*40
                    elif kind=='core_commit': identity['core_commit']='0'*40
                    elif kind=='candidate_version': identity['candidate_schema_version']=2
                    elif kind=='boolean_version': identity['candidate_schema_version']=True
                    elif kind=='spec': identity['feature_registry']['spec_id']='other'
                    elif kind=='feature_order': identity['feature_registry']['basic'].reverse()
                    elif kind=='group': identity['feature_registry']['group']='basic'
                    elif kind=='generation': identity['declared_generation']['checkpoint']='other'
                    elif kind=='membership': binding['training_membership']['questions']=['different']
                    elif kind=='extra_binding': binding['gold']='forbidden'
                    elif kind=='core_features': data['model']['features']['basic'].reverse()
                    elif kind=='core_hash': data['model_sha256']='0'*64
                    elif kind=='negative_scale': data['model']['state']['scale'][0]=-1
                    elif kind=='training_count': data['model']['state']['training_questions']=2
                    elif kind=='fallback': data['model']['no_signal']='first'
                    else: data['training_targets_sha256']='unknown'
                    if kind!='core_hash': data['model_sha256']=c.sha256(c.encode(data['model']))
                    raw=c.encode(data); path.write_bytes(raw)
                    with self.assertRaises(ValueError): TeamCandidateModel.load(path,train,expected_sha256=c.sha256(raw))

    def test_reload_then_rejects_inference_roles_overlap_labels_and_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; pin=model.save(path)
            loaded=TeamCandidateModel.load(path,train,expected_sha256=pin)
            for kind in ('train_role','dev','qid','article','context','checkpoint','dtype','labels','schema'):
                with self.subTest(kind=kind):
                    data=fixture('internal_val','valid'); q=data['questions'][0]
                    source=c.decode(train)['questions'][0]
                    if kind=='train_role': data['metadata']['split']='reranker_train'
                    elif kind=='dev': data['metadata']['split']='dev'
                    elif kind=='qid': q['question_id']=source['question_id']
                    elif kind=='article': q['article_id']=source['article_id']
                    elif kind=='context': q['context']=source['context']
                    elif kind=='checkpoint': data['metadata']['checkpoint']='changed'
                    elif kind=='dtype': data['metadata']['logit_dtype']='float64'
                    elif kind=='labels': data['targets']={}
                    else: data['schema_version']=2
                    q['context_id']=c.sha256(json.dumps([q['article_id'],q['context']],ensure_ascii=False).encode())
                    with self.assertRaises(ValueError): loaded.predict(c.encode(data))
            with patch.object(team,'PROFILE','changed-adapter'):
                with self.assertRaisesRegex(ValueError,'identity mismatch'): loaded.predict(heldout('new'))
            with patch.object(c,'FEATURE_SPEC_ID','changed-feature-spec'):
                with self.assertRaisesRegex(ValueError,'identity mismatch'): loaded.predict(heldout('new'))

    def test_core_nonfinite_state_and_unknown_envelope_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; model.save(path); original=c.decode(path.read_bytes())
            for kind in ('nonfinite','extra'):
                data=copy.deepcopy(original)
                if kind=='nonfinite': data['model']['state']['mean'][0]=float('nan')
                else: data['gold']='not allowed'
                raw=json.dumps(data).encode(); path.write_bytes(raw)
                with self.assertRaises(ValueError): TeamCandidateModel.load(path,train,expected_sha256=c.sha256(raw))

    def test_exclusive_save_and_failure_preserve_existing_model_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; pin=model.save(path); before=path.read_bytes()
            with self.assertRaises(FileExistsError): model.save(path)
            link=Path(tmp)/'link.json'; link.symlink_to(path)
            with self.assertRaises(FileExistsError): model.save(link)
            self.assertEqual(path.read_bytes(),before)
            zeros={qid:{cid:0. for cid in row} for qid,row in labels(train).items()}; saved=copy.deepcopy(zeros)
            with self.assertRaises(ValueError):
                TeamCandidateModel.fit_synthetic(train,zeros,group='basic',target_status=TARGET_STATUS)
            self.assertEqual(zeros,saved); self.assertEqual(model.predict(heldout('valid')),{'valid-q':'Bób'})
            with self.assertRaises(ValueError):
                TeamCandidateModel.fit_synthetic(train,labels(train),group='basic',target_status='real-unverified')
            unfitted=TeamCandidateModel(train,group='basic')
            absent=Path(tmp)/'unfitted.json'
            with self.assertRaises(ValueError): unfitted.save(absent)
            self.assertFalse(absent.exists()); self.assertEqual(c.sha256(path.read_bytes()),pin)

    def test_binding_json_key_order_does_not_change_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            train,model=fit(); path=Path(tmp)/'model.json'; model.save(path)
            data=c.decode(path.read_bytes())
            # Only reorder binding keys; core payload ordering still has its own byte digest.
            data['binding']=json.loads(json.dumps(data['binding'],sort_keys=True))
            raw=c.encode(data); path.write_bytes(raw)
            loaded=TeamCandidateModel.load(path,train,expected_sha256=c.sha256(raw))
            self.assertEqual(loaded.predict(heldout('valid')),{'valid-q':'Bób'})


if __name__ == '__main__':
    unittest.main()
