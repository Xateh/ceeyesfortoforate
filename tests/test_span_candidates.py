"""Offline regression tests, including the original baseline as an oracle."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from span_candidates import candidate_records, select_candidates


def original_baseline(contexts, ids, offsets, starts, ends, limit):
    best = {}
    for i, (qid, window) in enumerate(zip(ids, offsets)):
        for s in np.argsort(starts[i])[-20:]:
            for e in np.argsort(ends[i])[-20:]:
                if window[s] is None or window[e] is None:
                    continue
                if e < s or e - s + 1 > limit:
                    continue
                score = starts[i][s] + ends[i][e]
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, contexts[qid][window[s][0]:window[e][1]])
    return {qid: best[qid][1] if qid in best else '' for qid in contexts}


class SpanCandidatesTest(unittest.TestCase):
    def select(self, offsets, starts, ends, context='A B C', limit=30):
        return select_candidates({'q': context}, ['q'] * len(offsets), offsets,
                                 np.asarray(starts, dtype=np.float32),
                                 np.asarray(ends, dtype=np.float32), limit)['q']

    def test_overlap_keeps_complete_winning_triplet(self):
        rows = self.select([[(0, 1)], [(0, 1)]], [[4], [2]], [[1], [7]])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['start_logit'], rows[0]['end_logit'],
                          rows[0]['combined_score'], rows[0]['window_index']), (2, 7, 9, 1))

    def test_repeated_text_is_not_deduplicated(self):
        rows = self.select([[(0, 1), (2, 3)]], [[2, 1]], [[2, 1]], 'A A', 1)
        self.assertEqual([r['answer'] for r in rows], ['A', 'A'])
        self.assertEqual([r['start'] for r in rows], [0, 2])

    def test_filter_noncontext_reversed_and_overlength(self):
        rows = self.select([[None, (0, 1), (2, 3), (4, 5), None]],
                           [[100, 1, 2, 3, 100]], [[100, 3, 2, 1, 100]], limit=1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['start_token_index'] == r['end_token_index'] for r in rows))

    def test_duplicate_upgrade_does_not_take_earlier_tie_position(self):
        rows = self.select([[(0, 1)], [(2, 3)], [(0, 1)]], [[1], [3], [3]], [[1], [3], [3]])
        self.assertEqual([r['answer'] for r in rows], ['B', 'A'])

    def test_equal_duplicate_retains_first_window(self):
        rows = self.select([[(0, 1)], [(0, 1)]], [[1], [2]], [[2], [1]])
        self.assertEqual(rows[0]['window_index'], 0)

    def test_top20_across_windows(self):
        offsets = [[(2*i, 2*i+1)] for i in range(25)]
        rows = self.select(offsets, [[i] for i in range(25)], [[0]] * 25, 'A ' * 25)
        self.assertEqual(len(rows), 20)
        self.assertEqual([r['start'] for r in rows], list(range(48, 8, -2)))

    def test_unicode_export_and_no_labels(self):
        rows = self.select([[(0, 2)]], [[1]], [[2]], '🙂é!')
        examples = [{'id': 'q', 'title': 'Title', 'context': '🙂é!', 'question': 'What?',
                     'answers': {'text': ['secret'], 'answer_start': [0]}}]
        records = candidate_records(examples, {'q': rows})
        self.assertEqual(rows[0]['answer'], '🙂é')
        self.assertNotIn('answers', json.dumps(records))
        self.assertEqual(json.loads(json.dumps(records, ensure_ascii=False)), records)
        self.assertEqual(candidate_records(examples, {'q': rows}), records)

    def test_no_candidates_report_question(self):
        rows = self.select([[None]], [[1]], [[1]])
        self.assertEqual(rows, [])
        with self.assertRaisesRegex(ValueError, 'q'):
            candidate_records([{'id': 'q'}], {'q': rows})

    def test_invalid_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.select([[(0, 1)]], [[np.nan]], [[1]])
        with self.assertRaisesRegex(ValueError, 'character offsets'):
            self.select([[(0, 99)]], [[1]], [[1]])
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.select([[(0, 1)]], [[1]], [[1]], limit=0)
        with self.assertRaisesRegex(ValueError, 'row counts'):
            select_candidates({'q': 'A'}, ['q'], [], [], [])
        with self.assertRaisesRegex(ValueError, 'Unknown question'):
            select_candidates({}, ['q'], [[(0, 1)]], [[1]], [[1]])

    def test_original_baseline_agreement_random_logits_and_ties(self):
        rng = np.random.default_rng(4248)
        contexts = {'a': 'A ' * 50, 'b': 'B ' * 50, 'missing': 'C'}
        ids = ['a', 'b', 'a', 'b']
        offsets = [[None] + [(2*j, 2*j+1) for j in range(k, k+32)] + [None]
                   for k in (0, 0, 10, 10)]
        for dtype in (np.float32, np.float64):
            for ties in (False, True):
                for limit in (1, 5, 30):
                    for _ in range(10):
                        starts = rng.normal(size=(4, 34)).astype(dtype)
                        ends = rng.normal(size=(4, 34)).astype(dtype)
                        if ties:
                            starts = np.round(starts)
                            ends = np.round(ends)
                        rows = select_candidates(contexts, ids, offsets, starts, ends, limit)
                        actual = {qid: spans[0]['answer'] if spans else '' for qid, spans in rows.items()}
                        self.assertEqual(actual, original_baseline(contexts, ids, offsets, starts, ends, limit))
                        for qid, spans in rows.items():
                            self.assertEqual(len(spans), len({(r['start'], r['end']) for r in spans}))
                            self.assertEqual([r['combined_score'] for r in spans],
                                             sorted([r['combined_score'] for r in spans], reverse=True))
                            for r in spans:
                                self.assertEqual(r['answer'], contexts[qid][r['start']:r['end']])


if __name__ == '__main__':
    unittest.main()
