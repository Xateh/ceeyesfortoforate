"""Hand-authored synthetic receipts, never source data or tokenizer outputs."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import squad_preparation as prep

receipt = __import__('preparation_receipt') if importlib.util.find_spec('preparation_receipt') else None


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')


def seal(root):
    write_json(root/'artifact-hashes.json', {p.name:prep.file_hash(p) for p in root.iterdir()
                                           if p.name != 'artifact-hashes.json'})
    return prep.file_hash(root/'artifact-hashes.json')


def fixture(root):
    rows = [{'id':str(i), 'title':f'synthetic-{i}', 'context':f'😀 Ada {i}.',
             'question':'Who?', 'answers':{'text':['Ada'], 'answer_start':[2]}} for i in range(20)]
    manifest = prep.build_manifest(rows)
    selected = prep.select_pilot(manifest, 10)
    contexts, inputs, audits = {}, [], []
    for qid in selected:
        r = rows[int(qid)]; identity = {'question_id':qid, **manifest['questions'][qid]}
        contexts[identity['context_sha256']] = r['context']
        window = {'question_id':qid, 'window_index':0, 'input_ids':[101,200,102,100,1011,1012,102],
                  'attention_mask':[1]*7, 'token_type_ids':[0,0,0,1,1,1,1],
                  'offsets':[[0,0],[0,3],[0,0],[0,1],[2,5],[6,len(r['context'])],[0,0]],
                  'sequence_ids':[None,0,None,1,1,1,None]}
        inputs.append({**identity, 'question':r['question'], 'windows':[window]})
        audits.append({**identity, 'golds':[{'start':2,'end':5,'text':'Ada'}],
                       'windows':[{'window_index':0,'answers':[{'status':'exact','start_token':4,'end_token':4}]}],
                       'coverage':[{'status':'exact','exact_windows':1,'full_context':{'status':'exact','start_token':1,'end_token':1}}],
                       'context_capacity':380})
    write_json(root/'split-manifest.json',manifest)
    write_json(root/'selection.json',{'question_ids':selected,'selection_uses_gold':False})
    write_json(root/'pilot-contexts.json',contexts)
    for name, records in [('pilot-inputs.jsonl',inputs),('pilot-gold-audit.jsonl',audits)]:
        (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write_json(root/'audit-summary.json', {
        'status':'complete','processed_questions':10,'sample_counts':dict(Counter(x['role'] for x in inputs)),
        'window_gold_counts':{'exact':10},'gold_coverage':{'exact':10},'exceptions':[],
        'split_reproduced':True,'article_question_context_disjoint':True,
        'max_article_component':1,'component_fraction_max':0.05,
        'readiness':'preparation audit only; no trained checkpoint or QA accuracy'})
    # These are pinned identity declarations in a synthetic test, not real execution evidence.
    write_json(root/'provenance.json', {
        'dataset_id':prep.DATASET,'dataset_revision':prep.DATASET_REVISION,'configuration':'plain_text',
        'source_split':'train','source_path':'plain_text/train-00000-of-00001.parquet',
        'tokenizer_id':prep.TOKENIZER,'tokenizer_revision':prep.TOKENIZER_REVISION,
        'assets':{n:{'sha256':h,'bytes':s} for (n,h),s in zip(prep.ASSET_HASHES.items(),[14458314,466062,48,570,231508])},
        'code_sha256':prep.file_hash(Path(prep.__file__)), 'python':'synthetic-test',
        'packages':{'pyarrow':'synthetic-test','tokenizers':'synthetic-test'},
        'max_length':384,'stride':128,'truncation':'only_second','pair_order':['question','context'],
        'offsets':'Unicode code points; half-open','padding':False,'model_weights_loaded':False,'scorer_executed':False})
    return seal(root)


class Availability(unittest.TestCase):
    def test_api_exists(self):
        self.assertIsNotNone(receipt, 'offline receipt verifier missing')


@unittest.skipUnless(receipt, 'implementation not present yet')
class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pin = fixture(self.root)

    def verify(self):
        return receipt.verify_receipt(self.root, self.pin)

    def edit(self, name, change):
        path = self.root/name
        if name.endswith('.jsonl'):
            value = [json.loads(line) for line in path.read_text().splitlines()]
            change(value)
            path.write_text(''.join(json.dumps(v)+'\n' for v in value))
        else:
            value = json.loads(path.read_text()); change(value); write_json(path,value)
        self.pin = seal(self.root)

    def test_valid_receipt_is_read_only_and_imports_no_optional_packages(self):
        before = {p.name:p.read_bytes() for p in self.root.iterdir()}
        with patch.object(prep,'iter_rows',side_effect=AssertionError('source access')), \
             patch.object(prep,'tokenize_question',side_effect=AssertionError('tokenization')):
            result = self.verify()
        self.assertEqual(result['processed_questions'],10)
        self.assertEqual(result['windows'],10)
        self.assertEqual(result['status'],'verified_preparation_receipt')
        self.assertFalse(result['training_readiness_established'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})
        self.assertTrue({'torch','transformers','tokenizers','pyarrow'}.isdisjoint(sys.modules))

    def test_external_pin_and_changed_artifact_fail(self):
        self.pin = '0'*64
        with self.assertRaisesRegex(ValueError,'inventory'): self.verify()
        self.pin = prep.file_hash(self.root/'artifact-hashes.json')
        (self.root/'selection.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'hash'): self.verify()

    def test_missing_extra_failure_and_symlink_fail(self):
        p = self.root/'failure.json'; p.write_text('{}')
        with self.assertRaises(ValueError): self.verify()
        p.unlink()
        p = self.root/'selection.json'; content=p.read_bytes(); p.unlink()
        with self.assertRaises(ValueError): self.verify()
        p.symlink_to(self.root/'audit-summary.json')
        with self.assertRaises(ValueError): self.verify()
        p.unlink(); p.write_bytes(content)
        (self.root/'unexpected').mkdir()
        with self.assertRaises(ValueError): self.verify()

    def test_inventory_path_injection_fails(self):
        p=self.root/'artifact-hashes.json'; v=json.loads(p.read_text()); v['../escape']='0'*64
        write_json(p,v); self.pin=prep.file_hash(p)
        with self.assertRaises(ValueError): self.verify()

    def test_duplicate_json_keys_nonfinite_and_size_limit_fail(self):
        for text in ('{"x":1,"x":2}', '[NaN]', '[1e400]'):
            with self.subTest(text=text):
                (self.root/'selection.json').write_text(text); self.pin=seal(self.root)
                with self.assertRaises(ValueError): self.verify()
        self.pin=fixture(self.root)
        with patch.object(receipt,'MAX_TOTAL_BYTES',1), self.assertRaisesRegex(ValueError,'size'):
            self.verify()

    def test_selection_missing_extra_duplicate_and_reordered_fail(self):
        for action in (lambda v:v.pop(), lambda v:v.append('unknown'),
                       lambda v:v.append(v[0]), lambda v:v.reverse()):
            self.pin=fixture(self.root); self.edit('selection.json',lambda v:action(v['question_ids']))
            with self.assertRaises(ValueError): self.verify()

    def test_record_id_coverage_and_role_fail(self):
        for name in ('pilot-inputs.jsonl','pilot-gold-audit.jsonl'):
            for action in (lambda v:v.pop(), lambda v:v.append(v[0]),
                           lambda v:v[0].update(question_id='unknown'),
                           lambda v:v[0].update(role='official_dev')):
                self.pin=fixture(self.root); self.edit(name,action)
                with self.subTest(name=name), self.assertRaises(ValueError): self.verify()

    def test_input_gold_injection_window_lengths_and_boolean_offsets_fail(self):
        for action in (lambda v:v[0].update(golds=['Ada']),
                       lambda v:v[0]['windows'][0].update(target=1),
                       lambda v:v[0]['windows'][0]['input_ids'].pop(),
                       lambda v:v[0]['windows'][0]['offsets'][4].__setitem__(0,True)):
            self.pin=fixture(self.root); self.edit('pilot-inputs.jsonl',action)
            with self.assertRaises(ValueError): self.verify()

    def test_manifest_identity_overlap_counts_and_seed_fail(self):
        def overlap(m):
            q=next(iter(m['questions'].values())); q['role']=next(r for r in prep.ROLES if r!=q['role'])
        for action in (overlap,lambda m:m.update(seed=True),
                       lambda m:m['counts']['bert_train'].update(questions=0),
                       lambda m:next(iter(m['articles'].values())).update(title='changed')):
            self.pin=fixture(self.root); self.edit('split-manifest.json',action)
            with self.assertRaises(ValueError): self.verify()

    def test_context_hash_gold_text_alignment_and_coverage_fail(self):
        for name,action in (
            ('pilot-contexts.json',lambda v:v.update({next(iter(v)):'changed'})),
            ('pilot-gold-audit.jsonl',lambda v:v[0]['golds'][0].update(text='ada')),
            ('pilot-gold-audit.jsonl',lambda v:v[0]['windows'][0]['answers'][0].update(start_token=3)),
            ('pilot-gold-audit.jsonl',lambda v:v[0]['coverage'][0].update(exact_windows=0))):
            self.pin=fixture(self.root); self.edit(name,action)
            with self.assertRaises(ValueError): self.verify()

    def test_summary_and_provenance_fail_closed(self):
        for name,action in (
            ('audit-summary.json',lambda v:v.update(processed_questions=True)),
            ('audit-summary.json',lambda v:v.update(status='blocked')),
            ('audit-summary.json',lambda v:v.update(exceptions=[{'question_id':'unknown'}])),
            ('selection.json',lambda v:v.update(selection_uses_gold=True)),
            ('provenance.json',lambda v:v.update(source_split='validation')),
            ('provenance.json',lambda v:v.update(code_sha256='0'*64)),
            ('provenance.json',lambda v:v.update(scorer_executed=True))):
            self.pin=fixture(self.root); self.edit(name,action)
            with self.assertRaises(ValueError): self.verify()

    def install_boundary_exception(self):
        audit=json.loads((self.root/'pilot-gold-audit.jsonl').read_text().splitlines()[0])
        coverage={'status':'unrepresentable_boundary','exact_windows':0,
                  'full_context':{'status':'unrepresentable_boundary'}}
        def exception(v):
            v[0]['golds'][0].update(start=3,text='da')
            v[0]['windows'][0]['answers']=[{'status':'unrepresentable_boundary'}]
            v[0]['coverage']=[coverage]
        self.edit('pilot-gold-audit.jsonl',exception)
        self.edit('audit-summary.json',lambda v:v.update(
            status='complete_with_alignment_exceptions',gold_coverage={'exact':9,'unrepresentable_boundary':1},
            window_gold_counts={'exact':9,'unrepresentable_boundary':1},
            exceptions=[{'question_id':audit['question_id'],'coverage':[coverage]}]))

    def test_explicit_boundary_exception_is_retained(self):
        self.install_boundary_exception()
        before=(self.root/'pilot-inputs.jsonl').read_bytes()
        self.assertEqual(self.verify()['alignment_exception_questions'],1)
        self.assertEqual(before,(self.root/'pilot-inputs.jsonl').read_bytes())

    def test_zero_question_tokens_allowed_by_producer(self):
        def inputs(v):
            v[0]['question']=' '
            w=v[0]['windows'][0]
            for k in ('input_ids','attention_mask','token_type_ids','offsets','sequence_ids'):
                w[k].pop(1)
        def audit(v):
            v[0]['context_capacity']=381
            v[0]['windows'][0]['answers'][0].update(start_token=3,end_token=3)
        self.edit('pilot-inputs.jsonl',inputs); self.edit('pilot-gold-audit.jsonl',audit)
        self.assertEqual(self.verify()['processed_questions'],10)

    def test_other_nonexact_full_context_receipts_follow_producer_policy(self):
        # Full-context offsets are not present: these remain producer assertions.
        for status in ('outside','partial'):
            self.pin=fixture(self.root)
            self.install_boundary_exception()
            self.edit('pilot-gold-audit.jsonl',lambda v:v[0]['coverage'][0]['full_context'].update(status=status))
            self.edit('audit-summary.json',lambda v:v['exceptions'][0]['coverage'][0]['full_context'].update(status=status))
            self.assertEqual(self.verify()['alignment_exception_questions'],1)

    def test_shared_components_and_coherent_wrong_allocation(self):
        rows=[{'id':str(i),'title':f'article-{i}','context':f'context-{i}',
               'question':'Who?', 'answers':{'text':['context'],'answer_start':[0]}} for i in range(20)]
        rows += [{**rows[0],'id':'shared-a','context':'link-a'},
                 {**rows[1],'id':'shared-b','context':'link-a'},
                 {**rows[1],'id':'shared-c','context':'link-b'},
                 {**rows[2],'id':'shared-d','context':'link-b'}]
        m=prep.build_manifest(rows)
        receipt.manifest(m)
        self.assertEqual(m['component_sizes'][-1],3)
        candidates=[prep.digest(f'article-{i}') for i in range(3,20)]
        a=candidates[0]; b=next(b for b in candidates if m['articles'][b]['role']!=m['articles'][a]['role'])
        m['articles'][a]['role'],m['articles'][b]['role']=m['articles'][b]['role'],m['articles'][a]['role']
        for q in m['questions'].values(): q['role']=m['articles'][q['article_id']]['role']
        # Swap equal singleton components: counts and disjointness still agree.
        with self.assertRaisesRegex(ValueError,'allocation'): receipt.manifest(m)

    def test_cli_success_and_failure_exit_status(self):
        import subprocess
        command=[sys.executable, str(Path(receipt.__file__)), '--receipt',str(self.root),'--inventory-sha256']
        result=subprocess.run(command+[self.pin],capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['training_readiness_established'])
        result=subprocess.run(command+['0'*64],capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,1)
        self.assertIn('inventory pin mismatch',result.stderr)
