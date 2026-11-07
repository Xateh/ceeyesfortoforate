"""Tiny invented PR28-shaped exports; no model generation, real data or scorer."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from offline_handoff import contract as c
from reranker.core import Reranker, questions_from_records
from reranker.team_candidates import adapt_export, prepare_pair, fit_synthetic_pair, TIE

STATUS = 'synthetic_hand_authored_not_scorer_outputs'


def fixture(split='reranker_train', tag='train'):
    context = '\U0001f680 Ada met Bob in Paris ' + tag + '.'
    title = 'Synthetic: ' + tag
    spans = []
    # Tied Bob precedes earlier-offset Ada: the older draft would reorder them.
    for answer, start_logit, end_logit, token in [('Bob', 4., 3., 3), ('Ada', 4., 3., 1), ('Paris', 2., 1., 5)]:
        start = context.index(answer)
        spans.append({'answer': answer, 'start': start, 'end': start + len(answer),
                      'window_index': 7, 'start_token_index': token, 'end_token_index': token,
                      'span_token_count': 1, 'start_logit': start_logit, 'end_logit': end_logit,
                      'combined_score': float(np.float32(start_logit) + np.float32(end_logit))})
    q = {'question_id': tag + '-q', 'question': 'Where did they meet?', 'article_id': title,
         'context_id': c.sha256(json.dumps([title, context], ensure_ascii=False).encode()),
         'context': context, 'candidates': spans}
    return {'schema_version': 1, 'metadata': {
        'split': split, 'checkpoint': 'synthetic:no-model', 'examples_path': 'synthetic/' + split,
        'features_path': 'synthetic/windows/' + split, 'max_answer_length': 30, 'n_best': 20,
        'max_candidates': 20, 'batch_size': 1, 'device': 'cpu', 'numpy_version': 'synthetic-version',
        'torch_version': 'synthetic-version', 'logit_dtype': 'float32', 'tie_break': TIE,
        'max_context_only': False}, 'questions': [q]}


def labels(raw):
    # Hand-authored synthetic preferences, not evaluator output.
    return {q['question_id']: {x['candidate_id']: float(i == 0) for i, x in enumerate(q['candidates'])}
            for q in adapt_export(raw, group='basic')['questions']}


class TeamCandidatesTests(unittest.TestCase):
    def test_preserves_team_ties_triplets_identity_and_unicode(self):
        data = fixture(); raw = c.encode(data); out = adapt_export(raw, group='basic', baseline={'train-q':'Bob'})
        self.assertEqual([s['answer'] for s in out['questions'][0]['candidates']], ['Bob', 'Ada', 'Paris'])
        self.assertEqual(out['audit']['retained_origins'], data['questions'])
        self.assertEqual(out['audit']['input_sha256'], c.sha256(raw))
        self.assertEqual(out['audit']['baseline_answer_comparison'], 'matched')
        self.assertFalse(out['audit']['source_checkpoint_tokenizer_verified'])
        self.assertEqual(data, c.decode(raw))
        out['audit']['metadata']['checkpoint'] = 'changed copy'
        self.assertEqual(data, c.decode(raw))

    def test_float32_sum_is_not_python_sum_and_float64_also_supported(self):
        for dtype, expected in [('float32', 16777216.), ('float64', 16777217.)]:
            data = fixture(); data['metadata']['logit_dtype'] = dtype
            data['questions'][0]['candidates'] = data['questions'][0]['candidates'][:1]
            span = data['questions'][0]['candidates'][0]
            span.update(start_logit=16777216., end_logit=1., combined_score=expected)
            self.assertEqual(adapt_export(c.encode(data), group='basic')['audit']['retained_origins'][0]['candidates'][0], span)
            span['combined_score'] = 16777217. if dtype == 'float32' else 16777216.
            with self.assertRaises(ValueError): adapt_export(c.encode(data), group='basic')

    def test_basic_full_exact_vectors_and_question_type_exclusion(self):
        raw = c.encode(fixture()); basic = adapt_export(raw, group='basic'); full = adapt_export(raw, group='full')
        a = basic['questions'][0]['candidates'][0]['features']; b = full['questions'][0]['candidates'][0]['features']
        self.assertEqual(tuple(a), c.BASIC); self.assertEqual(list(a.values()), [4,3,1,0,1])
        self.assertEqual(tuple(b), c.BASIC + c.COMPATIBILITY)
        self.assertEqual(list(b.values())[5:], [0,0,0,1,0,0,0,0,0,0])
        changed = fixture(); changed['questions'][0]['question'] = 'Who met them?'
        self.assertEqual(basic['questions'], adapt_export(c.encode(changed), group='basic')['questions'])
        self.assertEqual(adapt_export(c.encode(changed), group='full')['questions'][0]['candidates'][0]['features']['who_initial_upper_match'], 1)

    def test_duplicate_text_at_distinct_spans_survives(self):
        data = fixture(); q = data['questions'][0]
        q['context'] += ' Bob'; q['context_id'] = c.sha256(json.dumps([q['article_id'], q['context']],ensure_ascii=False).encode())
        another = copy.deepcopy(q['candidates'][0]); another.update(start=len(q['context'])-3,end=len(q['context']),window_index=99)
        q['candidates'].insert(1,another)
        items = adapt_export(c.encode(data),group='basic')['questions'][0]['candidates']
        self.assertEqual(items[0]['answer'],items[1]['answer']); self.assertNotEqual(items[0]['candidate_id'],items[1]['candidate_id'])

    def test_bad_schema_injection_ids_bounds_order_and_numbers_fail(self):
        for kind in ('top_gold','features','question_gold','duplicate_qid','context_id','span','tokens','duplicate_span','order','boolean','nonfinite','dtype','unrepresentable','empty','too_many'):
            with self.subTest(kind=kind):
                data = fixture(); q = data['questions'][0]; s = q['candidates'][0]
                if kind == 'top_gold': data['targets'] = {}
                elif kind == 'features': s['features'] = {'innocuous_gold_value': 1}
                elif kind == 'question_gold': q['gold'] = 'Bob'
                elif kind == 'duplicate_qid': data['questions'].append(copy.deepcopy(q))
                elif kind == 'context_id': q['context_id'] = '0'*64
                elif kind == 'span': s['end'] -= 1
                elif kind == 'tokens': s['span_token_count'] = 2
                elif kind == 'duplicate_span': q['candidates'].insert(1,copy.deepcopy(s))
                elif kind == 'order': q['candidates'].reverse()
                elif kind == 'boolean': s['start_logit'] = True
                elif kind == 'nonfinite': s['start_logit'] = 'nan'
                elif kind == 'dtype': data['metadata']['logit_dtype'] = 'float16'
                elif kind == 'unrepresentable': s['start_logit'] = .1
                elif kind == 'empty': q['candidates'] = []
                else: q['candidates'] = [copy.deepcopy(s) for _ in range(21)]
                before = copy.deepcopy(data)
                with self.assertRaises(ValueError): adapt_export(c.encode(data),group='basic')
                self.assertEqual(data,before)
        raw = c.encode(fixture()).replace(b'4.0',b'NaN',1)
        with self.assertRaises(ValueError): adapt_export(raw,group='basic')

    def test_baseline_exact_coverage_and_text(self):
        for mapping in ({},{'extra':'Bob'},{'train-q':'Ada'},{'train-q':'Bob','extra':'Bob'}):
            with self.assertRaises(ValueError): adapt_export(c.encode(fixture()),group='basic',baseline=mapping)

    def test_role_and_overlap_guards_run_before_fit(self):
        train = fixture(); valid = fixture('internal_val','valid')
        for kind in ('fit_role','dev','question','article','context','checkpoint'):
            with self.subTest(kind=kind):
                t,v = copy.deepcopy(train),copy.deepcopy(valid)
                if kind == 'fit_role': t['metadata']['split'] = 'internal_val'
                elif kind == 'dev': v['metadata']['split'] = 'dev'
                elif kind == 'question': v['questions'][0]['question_id'] = 'train-q'
                elif kind == 'article': v['questions'][0]['article_id'] = t['questions'][0]['article_id']
                elif kind == 'context': v['questions'][0]['context'] = t['questions'][0]['context']
                else: v['metadata']['checkpoint'] = 'different'
                q=v['questions'][0]; q['context_id']=c.sha256(json.dumps([q['article_id'],q['context']],ensure_ascii=False).encode())
                with patch.object(Reranker,'fit') as fit:
                    with self.assertRaises(ValueError):
                        fit_synthetic_pair(c.encode(t),labels(c.encode(t)),c.encode(v),group='basic',target_status=STATUS)
                    fit.assert_not_called()

    def test_actual_core_fit_unicode_ties_scaling_and_reload_both_groups(self):
        train = c.encode(fixture()); val = fixture('internal_val','valid')
        q=val['questions'][0]; q['context']=q['context'].replace('Bob','Bób')
        q['candidates'][0]['answer']='Bób'; q['context_id']=c.sha256(json.dumps([q['article_id'],q['context']],ensure_ascii=False).encode())
        for group in ('basic','full'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                model,t,v = fit_synthetic_pair(train,labels(train),c.encode(val),group=group,target_status=STATUS)
                records = questions_from_records(v['questions']); first = records[0].candidates[0]
                self.assertEqual(model.select(records)['valid-q'].candidate_id, first.candidate_id)
                self.assertEqual(model.predict(records),{'valid-q':'Bób'})
                path=Path(tmp)/'model.json'; model.save(path); saved=path.read_bytes()
                reloaded=Reranker.load(path)
                self.assertEqual(reloaded.features.names,c.validate_config(t['feature_registry']))
                self.assertEqual(reloaded.scores(records),model.scores(records))
                self.assertEqual(reloaded.select(records),model.select(records))
                reloaded.save_predictions(records,Path(tmp)/'predictions.json')
                self.assertIn('Bób'.encode(),(Path(tmp)/'predictions.json').read_bytes())
                # Held-out extreme values do not refit or change learned transforms.
                shifted=copy.deepcopy(val)
                for s in shifted['questions'][0]['candidates']:
                    s.update(start_logit=10000.,end_logit=10000.,combined_score=20000.)
                _, shifted_batch=prepare_pair(train,c.encode(shifted),group=group)
                reloaded.predict(questions_from_records(shifted_batch['questions']))
                reloaded.save(path); self.assertEqual(path.read_bytes(),saved)

    def test_variable_counts_equal_groups_and_no_signal_targets(self):
        data=fixture(); single=fixture(tag='single')['questions'][0]; equal=fixture(tag='equal')['questions'][0]
        single['candidates']=single['candidates'][:1]
        for q,window in [(single,8),(equal,9)]:
            for s in q['candidates']: s['window_index']=window
        data['questions'] += [single,equal]; raw=c.encode(data); targets=labels(raw)
        targets['equal-q']={key:0. for key in targets['equal-q']}
        valid=c.encode(fixture('internal_val','valid'))
        with tempfile.TemporaryDirectory() as tmp:
            model,t,v=fit_synthetic_pair(raw,targets,valid,group='basic',target_status=STATUS)
            path=Path(tmp)/'model.json'; model.save(path); state=c.decode(path.read_bytes())['state']
            self.assertEqual(state['training_questions'],3); self.assertEqual(state['informative_questions'],1)
            self.assertEqual([len(q['candidates']) for q in t['questions']],[3,1,3])
        zeros={qid:{cid:0. for cid in row} for qid,row in targets.items()}; before=copy.deepcopy(zeros)
        with self.assertRaisesRegex(ValueError,'no within-question target signal'):
            fit_synthetic_pair(raw,zeros,valid,group='basic',target_status=STATUS)
        self.assertEqual(zeros,before); self.assertEqual(c.decode(raw),data)
        for bad in ({}, {**targets,'extra':{}}, {**targets,'train-q':{}}):
            with self.assertRaises(ValueError): fit_synthetic_pair(raw,bad,valid,group='basic',target_status=STATUS)

    def test_shared_global_window_index_rejected(self):
        data=fixture(); data['questions'].append(fixture(tag='other')['questions'][0])
        with self.assertRaisesRegex(ValueError,'global window'):
            adapt_export(c.encode(data),group='basic')

    def test_synthetic_attestation_required_before_fitting(self):
        raw=c.encode(fixture()); valid=c.encode(fixture('internal_val','valid'))
        with patch.object(Reranker,'fit') as fit:
            with self.assertRaises(ValueError): fit_synthetic_pair(raw,labels(raw),valid,group='basic',target_status='unverified_real')
            fit.assert_not_called()

    def test_cli_basic_full_and_existing_file_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.json'; source.write_bytes(c.encode(fixture()))
            baseline=Path(tmp)/'baseline.json'; baseline.write_bytes(c.encode({'train-q':'Bob'}))
            for group in ('basic','full'):
                output=Path(tmp)/(group+'.json')
                result=subprocess.run([sys.executable,'-m','reranker.team_candidates','--input',str(source),
                    '--output',str(output),'--group',group,'--baseline',str(baseline)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(c.decode(output.read_bytes())['feature_registry']['group'],group)
            before=source.read_bytes()
            from reranker.team_candidates import main
            with self.assertRaises(FileExistsError): main(['--input',str(source),'--output',str(source),'--group','basic'])
            self.assertEqual(source.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
