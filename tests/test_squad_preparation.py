"""Synthetic unit fixtures only; no downloads or model imports."""
import importlib.util
import unittest

spec = importlib.util.find_spec('squad_preparation')
prep = __import__('squad_preparation') if spec else None


def row(i, title=None, context=None):
    return {'id': str(i), 'title': title or f'article-{i}',
            'context': context or f'Ada wrote {i}.', 'question': 'Who wrote?',
            'answers': {'text': ['Ada'], 'answer_start': [0]}}


class Availability(unittest.TestCase):
    def test_preparation_api_exists(self):
        self.assertIsNotNone(prep, 'preparation API is not implemented')


@unittest.skipUnless(prep, 'implementation not present yet')
class PreparationTests(unittest.TestCase):
    def test_reproducible_disjoint_split_and_counts(self):
        rows = [row(i) for i in range(20)]
        a = prep.build_manifest(iter(rows))
        self.assertEqual(a, prep.build_manifest(iter(reversed(rows))))
        self.assertEqual([a['counts'][r]['articles'] for r in prep.ROLES], [16, 2, 2])
        self.assertEqual(sum(x['questions'] for x in a['counts'].values()), 20)
        self.assertEqual(a['seed'], 42)
        self.assertEqual(a['status'], 'provisional_not_team_agreed')

    def test_shared_context_unions_articles_transitively(self):
        rows = [row(i) for i in range(20)]
        rows += [row('a','article-0','same'), row('b','article-1','same'),
                 row('c','article-1','other'), row('d','article-2','other')]
        m = prep.build_manifest(iter(rows))
        self.assertEqual(len({m['questions'][str(i)]['role'] for i in range(3)}), 1)
        self.assertEqual(m['component_sizes'][-1], 3)

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prep.build_manifest(iter([row(0), row(0)]))

    def test_too_few_components_fail(self):
        with self.assertRaisesRegex(ValueError, 'components'):
            prep.build_manifest(iter([row(0)]))

    def test_strict_row_and_gold_validation(self):
        for edit in ({'id': ''}, {'answers': {'text':['Ada'], 'answer_start':[True]}},
                     {'answers': {'text':['Ada'], 'answer_start':[1]}},
                     {'answers': {'text':[''], 'answer_start':[0]}},
                     {'answers': {'text':['Ada'], 'answer_start':[]}}, {'extra': 1}):
            r = row(1); r.update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                prep.validate_row(r, gold=True)

    def test_unicode_and_all_gold_references(self):
        r = row(1, context='😀 Ada, Ada!')
        r['answers'] = {'text':['Ada','Ada'], 'answer_start':[2,7]}
        prep.validate_row(r, gold=True)
        r['answers']['answer_start'][1] = 6
        with self.assertRaises(ValueError): prep.validate_row(r, gold=True)

    def test_sampling_is_deterministic_exact_and_bounded(self):
        m = prep.build_manifest(iter(row(i) for i in range(100)))
        a = prep.select_pilot(m, 20)
        self.assertEqual(a, prep.select_pilot(m, 20))
        self.assertEqual([sum(m['questions'][q]['role']==r for q in a) for r in prep.ROLES], [16,2,2])
        for n in (True,0,1001):
            with self.assertRaises(ValueError): prep.select_pilot(m,n)

    def test_disjointness_gate_rejects_tampered_roles(self):
        m = prep.build_manifest(iter([row(i) for i in range(20)] + [row('x','article-0')]))
        m['questions']['x']['role'] = next(r for r in prep.ROLES if r != m['questions']['0']['role'])
        with self.assertRaisesRegex(ValueError,'overlap|role'): prep.verify_disjoint(m)

    def test_alignment_api_exists(self):
        self.assertTrue(hasattr(prep, 'audit_window'), 'alignment API missing')

    @unittest.skipUnless(hasattr(prep, 'audit_window'), 'alignment not present yet')
    def test_alignment_exact_partial_outside_and_boundary(self):
        context = '😀 Ada wrote notes.'
        offsets = [(0,0),(0,1),(2,5),(6,11),(12,17),(17,18),(0,0)]
        seq = [None,1,1,1,1,1,None]
        golds = [{'start':2,'end':5}, {'start':3,'end':5}]
        a = prep.audit_window(context, golds, offsets, seq)
        self.assertEqual(a[0], {'status':'exact','start_token':2,'end_token':2})
        self.assertEqual(a[1]['status'], 'unrepresentable_boundary')
        a = prep.audit_window(context,[{'start':2,'end':11},{'start':12,'end':17}],[(2,5)],[1])
        self.assertEqual([x['status'] for x in a], ['partial','outside'])
        with self.assertRaises(ValueError): prep.audit_window(context,golds,[(0,0)],[1])
        with self.assertRaises(ValueError): prep.audit_window(context,golds,[(0,30)],[1])

    @unittest.skipUnless(hasattr(prep, 'audit_window'), 'alignment not present yet')
    def test_question_offsets_do_not_count_as_context(self):
        a = prep.audit_window('Ada',[{'start':0,'end':3}],[(0,3),(0,3)],[0,1])
        self.assertEqual(a[0]['start_token'],1)

    def test_asset_verification_api_exists(self):
        self.assertTrue(hasattr(prep, 'verified_assets'), 'asset verification API missing')

    @unittest.skipUnless(hasattr(prep, 'verified_assets'), 'verification not present yet')
    def test_unknown_or_changed_bytes_fail_closed(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,FileNotFoundError)): prep.verified_assets(Path(d))
            for name in prep.ASSET_HASHES: (Path(d)/name).write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError,'hash'): prep.verified_assets(Path(d))

    def test_tokenize_api_exists(self):
        self.assertTrue(hasattr(prep, 'tokenize_question'), 'tokenization API missing')

    @unittest.skipUnless(hasattr(prep, 'tokenize_question'), 'tokenization not present yet')
    def test_real_tokenizer_library_overflow_and_label_separation(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        tok = Tokenizer(models.WordPiece({'[UNK]':0,'[CLS]':1,'[SEP]':2,'who':3,'Ada':4,'x':5}))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',1),('[SEP]',2)])
        r = row(0, context='x '*400+'Ada'); r['answers']['answer_start']=[800]
        inputs,audit = prep.tokenize_question(r,tok)
        self.assertGreater(len(inputs),1)
        self.assertTrue(any(w['answers'][0]['status']=='exact' for w in audit['windows']))
        self.assertTrue(all(len(w['input_ids'])<=384 for w in inputs))
        self.assertTrue(all('answers' not in w and 'gold' not in w for w in inputs))
        self.assertEqual(audit['golds'][0]['text'],'Ada')
        r['question']='who '*300
        with self.assertRaisesRegex(ValueError,'capacity'): prep.tokenize_question(r,tok)

    def test_pipeline_api_exists(self):
        self.assertTrue(hasattr(prep, 'prepare'), 'pipeline API missing')

    @unittest.skipUnless(hasattr(prep, 'prepare'), 'pipeline not present yet')
    def test_pipeline_preserves_existing_output_and_rejects_limit(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with self.assertRaises(FileExistsError): prep.prepare(p,p,1000)
            with self.assertRaisesRegex(ValueError,'limit'): prep.prepare(p,p/'new',1001)
            self.assertFalse((p/'new').exists())

    def test_parquet_schema_gate(self):
        import tempfile
        from pathlib import Path
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'fixture.parquet'
            pq.write_table(pa.table({'id':['x']}),path)
            with self.assertRaisesRegex(ValueError,'schema'): list(prep.iter_rows(path))

    def test_io_failure_before_first_question_keeps_original_error(self):
        self._check_io_failure('pilot-inputs.jsonl')

    def test_finalization_failure_receipt_marks_incomplete(self):
        self._check_io_failure('artifact-hashes.json')

    def _check_io_failure(self, failing_name):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        tok = Tokenizer(models.WordPiece({'[UNK]':0,'[CLS]':1,'[SEP]':2,'Ada':3}))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',1),('[SEP]',2)])
        original = Path.open
        def fail_open(path,*args,**kwargs):
            if path.name == failing_name: raise OSError('simulated output failure')
            return original(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'output'
            with patch.object(prep,'verified_assets',return_value={}), \
                 patch.object(prep,'iter_rows',side_effect=lambda p:iter(row(i) for i in range(20))), \
                 patch('tokenizers.Tokenizer.from_file',return_value=tok), \
                 patch.object(Path,'open',fail_open):
                with self.assertRaisesRegex(OSError,'simulated output failure'):
                    prep.prepare(Path(d),output,1)
            import json
            failure = json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['status'],'blocked')
            self.assertIn('simulated output failure',failure['error'])
            if failing_name == 'pilot-inputs.jsonl':
                self.assertIsNone(failure['failed_question_id'])
