"""Only offline boundary tests: no reranker, torch, tokenizer or scorer import."""
import copy
import json
import unittest

from offline_handoff import contract as c
from offline_handoff.fixtures import demo_sources, synthetic_bundle


class OfflineContractTests(unittest.TestCase):
    def setUp(self):
        self.contexts, self.records = demo_sources()
        self.config, self.policy = c.feature_config('basic'), c.candidate_policy()
        self.manifest, self.artifacts = synthetic_bundle()

    def adapt(self):
        return c.adapt_candidates(self.contexts, self.records, self.config, self.policy)

    def alter_artifact(self, name, mutate):
        obj = c.decode(self.artifacts[name])
        mutate(obj)
        self.artifacts[name] = c.encode(obj)
        manifest = c.decode(self.manifest)
        manifest['artifact_sha256'][name] = c.sha256(self.artifacts[name])
        if name == 'split-manifest.json':
            manifest['split_manifest_sha256'] = c.sha256(self.artifacts[name])
        self.manifest = c.encode(manifest)

    def assert_bundle_rejected(self):
        with self.assertRaises(c.ContractError):
            c.validate_bundle(self.manifest, self.artifacts)

    def test_exact_fixture_vectors_basic_and_full(self):
        for group, width in (('basic', 5), ('full', 15)):
            with self.subTest(group=group):
                manifest, artifacts = synthetic_bundle(group)
                report = c.validate_bundle(manifest, artifacts)
                self.assertEqual(report['feature_width'], width)
                config = c.decode(manifest)['feature_registry']
                suffix = [] if group == 'basic' else [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                self.assertEqual(c.vectors(c.decode(artifacts['questions.json']), config),
                                 [[[4, 3, 1, 0, 1] + suffix, [2, 1, 1, 0, 1] + suffix]])
                self.assertFalse(report['production_valid'])
                self.assertFalse(report['model_execution'])
                self.assertFalse(report['scorer_execution'])

    def test_basic_excludes_all_question_type_fields(self):
        before, _ = self.adapt()
        self.records[0]['question'] = 'When was this?'
        after, _ = self.adapt()
        self.assertEqual(before, after)
        for candidate in after[0]['candidates']:
            self.assertEqual(tuple(candidate['features']), c.BASIC)
        self.config = c.feature_config('full')
        full, _ = self.adapt()
        self.assertEqual(full[0]['candidates'][0]['features']['q_when'], 1)

    def test_question_type_and_ascii_rules(self):
        o = self.records[0]['occurrences'][0]
        for question, kind in [(' WHO?', 'who'), ('\u2003When?', 'when'), ('In which?', 'other'),
                               ('?Who', 'other'), ('Whoever', 'other'), ('whoé', 'who')]:
            values = c.features(question, o, 'full')
            self.assertEqual([values['q_' + t] for t in c.TYPES], [int(t == kind) for t in c.TYPES])
        o['answer'] = '🙂 ９éaB2'
        values = c.features('When?', o, 'full')
        self.assertEqual((values['answer_has_digit'], values['answer_initial_upper'], values['when_digit_match']), (1, 0, 1))
        o['answer'] = '🙂９é'
        values = c.features('Who?', o, 'full')
        self.assertEqual((values['answer_has_digit'], values['answer_initial_upper']), (0, 0))

    def test_extra_keys_and_feature_injection(self):
        for location in ('question', 'candidate', 'feature'):
            with self.subTest(location=location):
                self.setUp()
                def mutate(records):
                    q = records[0]
                    target = q if location == 'question' else q['candidates'][0]
                    if location == 'feature':
                        target = target['features']
                    target['gold_f1'] = 1
                self.alter_artifact('questions.json', mutate)
                self.assert_bundle_rejected()

    def test_disguised_allowed_feature_mismatch(self):
        self.alter_artifact('questions.json', lambda q: q[0]['candidates'][0]['features'].update(bert_start_logit=1))
        self.assert_bundle_rejected()

    def test_boolean_and_nonfinite_features(self):
        for value in (True, float('inf'), float('nan')):
            questions, _ = self.adapt()
            expected = copy.deepcopy(questions)
            questions[0]['candidates'][0]['features']['answer_initial_upper'] = value
            with self.assertRaises(c.ContractError):
                c.validate_questions(questions, expected, self.config)

    def test_raw_schema_numbers_and_ids(self):
        for field, value in [('start', True), ('window_index', -1), ('start_logit', True),
                             ('end_logit', float('inf')), ('combined_score', float('nan')),
                             ('context_id', 'missing'), ('context_id', 'bad:id'), ('answer', 9)]:
            with self.subTest(field=field, value=value):
                self.setUp()
                self.records[0]['occurrences'][0][field] = value
                with self.assertRaises(c.ContractError):
                    self.adapt()
        self.setUp()
        self.records[0]['occurrences'][0]['target'] = 1
        with self.assertRaises(c.ContractError):
            self.adapt()

    def test_exact_slice_and_invalid_baseline(self):
        for field, value in [('end', 23), ('answer', 'paris'), ('start', -1)]:
            self.setUp()
            self.records[0]['baseline'][field] = value
            with self.assertRaises(c.ContractError):
                self.adapt()
        self.setUp()
        self.records[0]['occurrences'][0]['end'] = 23
        with self.assertRaisesRegex(c.ContractError, 'exact context slice'):
            self.adapt()
        self.setUp()
        self.records[0]['occurrences'] = self.records[0]['occurrences'][1:]
        with self.assertRaisesRegex(c.ContractError, 'baseline'):
            self.adapt()

    def test_invalid_span_filters_before_dedup(self):
        base = self.records[0]['occurrences'][1]
        changes = [({'is_context': False}, 'non_context'),
                   ({'start': 3, 'end': 0}, 'invalid_character_bounds'),
                   ({'end': 999}, 'invalid_character_bounds'),
                   ({'start_token_index': -1}, 'invalid_token_bounds'),
                   ({'end_token_index': 100}, 'overlength_tokens')]
        for index, (change, _) in enumerate(changes, 1):
            o = {**base, **change, 'window_index': index}
            self.records[0]['occurrences'].append(o)
        questions, provenance = self.adapt()
        self.assertEqual(len(questions[0]['candidates']), 2)
        self.assertEqual([o['filter_reason'] for o in provenance[0]['origins'][2:]], [r for _, r in changes])

    def test_zero_generation_receipt_and_no_partial_batch(self):
        second = copy.deepcopy(self.records[0])
        second['question_id'] = 'q-empty'
        second['occurrences'] = []
        self.records.append(second)
        with self.assertRaises(c.ContractError) as error:
            self.adapt()
        self.assertEqual(error.exception.receipt, {'status': 'candidate_generation_failure',
                                                 'question_id': 'q-empty', 'origins': []})
        self.records = [self.records[0]]
        for o in self.records[0]['occurrences']:
            o['is_context'] = False
        with self.assertRaises(c.ContractError) as error:
            self.adapt()
        self.assertEqual(len(error.exception.receipt['origins']), 2)

    def test_duplicate_window_winner_whole_triplet_and_tie(self):
        o = self.records[0]['occurrences'][0]
        self.records[0]['occurrences'] += [{**o, 'window_index': 2, 'start_logit': 1.0,
                                         'end_logit': 9.0, 'combined_score': 10.0},
                                        {**o, 'window_index': 1, 'start_logit': 6.0,
                                         'end_logit': 4.0, 'combined_score': 10.0}]
        questions, sidecar = self.adapt()
        self.assertEqual(len(questions[0]['candidates']), 2)
        winner = sidecar[0]['selected'][0]
        self.assertEqual((winner['start_logit'], winner['end_logit'], winner['combined_score']), (6, 4, 10))
        self.assertEqual(len(sidecar[0]['origins']), 4)
        self.records[0]['occurrences'].reverse()
        reversed_questions, reverse_sidecar = self.adapt()
        self.assertEqual(questions, reversed_questions)
        self.assertEqual(sidecar[0]['selected'], reverse_sidecar[0]['selected'])

    def test_conflicting_origin_rejected_exact_duplicate_preserved(self):
        o = self.records[0]['occurrences'][0]
        self.records[0]['occurrences'].append(copy.deepcopy(o))
        _, provenance = self.adapt()
        self.assertEqual(len(provenance[0]['origins']), 3)
        self.records[0]['occurrences'][-1].update(start_logit=3, end_logit=4)
        with self.assertRaisesRegex(c.ContractError, 'conflicting'):
            self.adapt()

    def test_combined_score_not_softmax_or_mixed(self):
        self.records[0]['occurrences'][0]['combined_score'] = 0.7
        with self.assertRaisesRegex(c.ContractError, 'raw logit sum'):
            self.adapt()
        self.records[0]['occurrences'][0].update(start_logit=1e308, end_logit=1e308, combined_score=1e308)
        with self.assertRaisesRegex(c.ContractError, 'finite'):
            self.adapt()

    def test_unicode_codepoints_exact_text_and_empty_answer(self):
        context = 'The 🙂é, A!'
        self.contexts[0]['text'] = context
        o = self.records[0]['occurrences'][0]
        o.update(start=4, end=6, answer='🙂é')
        self.records[0]['baseline'].update(start=4, end=6, answer='🙂é')
        self.records[0]['occurrences'] = [o]
        questions, _ = self.adapt()
        self.assertEqual(c.decode(c.encode(questions)), questions)
        self.assertEqual(questions[0]['candidates'][0]['answer'], '🙂é')
        c.validate_mapping({'demo-q1': '🙂é'}, questions)
        o['end'] = 7  # UTF-16 length would incorrectly include the comma.
        with self.assertRaises(c.ContractError):
            self.adapt()
        o.update(end=6, answer='')
        with self.assertRaises(c.ContractError):
            self.adapt()

    def test_top20_baseline_first_equal_text_distinct_spans(self):
        self.contexts[0]['text'] = 'A ' * 21
        template = self.records[0]['occurrences'][0]
        occurrences = [{**template, 'start': 2 * i, 'end': 2 * i + 1, 'answer': 'A',
                        'start_token_index': i, 'end_token_index': i,
                        'start_logit': 2, 'end_logit': 1, 'combined_score': 3} for i in range(21)]
        occurrences[20].update(start_logit=-5, end_logit=-5, combined_score=-10)
        self.records[0]['occurrences'] = list(reversed(occurrences))
        self.records[0]['baseline'] = {k: occurrences[20][k] for k in ('context_id', 'start', 'end', 'answer')}
        questions, sidecar = self.adapt()
        self.assertEqual([o['start'] for o in sidecar[0]['selected']], [40] + list(range(0, 38, 2)))
        self.assertEqual(len(questions[0]['candidates']), 20)
        self.assertEqual(len(sidecar[0]['origins']), 21)
        self.assertEqual(len({o['candidate_id'] for o in questions[0]['candidates']}), 20)
        self.records[0]['occurrences'].reverse()
        self.assertEqual(self.adapt()[0], questions)

    def test_baseline_reversal_duplicate_question_and_candidate_rejected(self):
        self.alter_artifact('questions.json', lambda q: q[0]['candidates'].reverse())
        self.assert_bundle_rejected()
        self.setUp()
        self.records.append(copy.deepcopy(self.records[0]))
        with self.assertRaisesRegex(c.ContractError, 'duplicate question'):
            self.adapt()
        self.setUp()
        self.alter_artifact('questions.json', lambda q: q[0]['candidates'].append(copy.deepcopy(q[0]['candidates'][0])))
        self.assert_bundle_rejected()

    def test_registry_spec_and_group_classification(self):
        for field, value in [('spec_id', 'candidate-features/0.1.1-draft'),
                             ('basic', list(reversed(c.BASIC))),
                             ('basic', list(c.BASIC[:-1]) + ['q_where']),
                             ('compatibility', list(reversed(c.COMPATIBILITY))), ('group', 'other')]:
            config = c.feature_config('basic')
            config[field] = value
            with self.assertRaises(c.ContractError):
                c.validate_config(config)

    def test_target_exact_coverage_and_domain(self):
        questions, _ = self.adapt()
        targets = c.decode(self.artifacts['targets.json'])
        for invalid in ({}, {**targets, 'extra': {}}, {'demo-q1': {}},
                        {'demo-q1': {**targets['demo-q1'], 'extra': 0}}):
            with self.assertRaises(c.ContractError):
                c.validate_targets(invalid, questions)
        for value in (True, -0.01, 1.01, float('nan'), float('inf')):
            invalid = copy.deepcopy(targets)
            invalid['demo-q1']['demo-context-A:19:24'] = value
            with self.assertRaises(c.ContractError):
                c.validate_targets(invalid, questions)

    def test_zero_targets_preserve_inputs_and_nosignal_classification(self):
        manifest, zero = synthetic_bundle(zero_targets=True)
        for name in c.FILES - {'targets.json'}:
            self.assertEqual(self.artifacts[name], zero[name])
        report = c.validate_bundle(manifest, zero)
        self.assertEqual(report['target_groups']['equal_zero'], ['demo-q1'])
        questions, _ = self.adapt()
        targets = c.decode(zero['targets.json'])
        targets['demo-q1'] = dict.fromkeys(targets['demo-q1'], 0.4)
        before = copy.deepcopy(questions)
        self.assertEqual(c.validate_targets(targets, questions)['equal_nonzero'], ['demo-q1'])
        self.assertEqual(questions, before)
        questions[0]['candidates'].pop()
        targets['demo-q1'].pop('demo-context-A:0:3')
        self.assertEqual(c.validate_targets(targets, questions)['single_candidate'], ['demo-q1'])

    def test_hash_version_artifact_and_provenance_integrity(self):
        self.artifacts['contexts.json'] += b' '
        self.assert_bundle_rejected()
        for field in ('contract_id', 'schema_id', 'feature_spec_id', 'core_commit', 'split_manifest_sha256'):
            self.setUp()
            manifest = c.decode(self.manifest)
            manifest[field] = 'wrong'
            with self.assertRaises(c.ContractError):
                c.validate_bundle(c.encode(manifest), self.artifacts)
        self.setUp()
        self.artifacts['extra.json'] = b'{}'
        self.assert_bundle_rejected()
        self.setUp()
        self.alter_artifact('provenance.json', lambda p: p[0]['origins'].pop())
        self.assert_bundle_rejected()
        self.setUp()
        self.alter_artifact('provenance.json', lambda p: p[0]['selected'][0].update(start_logit=True))
        self.assert_bundle_rejected()

    def test_strict_json_duplicate_keys_nonfinite_and_encoding(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'Infinity', b'\xff', b'{'):
            with self.assertRaises(c.ContractError):
                c.decode(raw)

    def test_unknown_production_and_scorer_refusal(self):
        manifest = c.decode(self.manifest)
        manifest.update(status='production', split_role='reranker_train')
        with self.assertRaisesRegex(c.ContractError, 'provenance required'):
            c.validate_bundle(c.encode(manifest), self.artifacts)
        questions, _ = self.adapt()
        audit = c.decode(self.artifacts['label-audit.json'])
        with self.assertRaisesRegex(c.ContractError, 'official scorer unresolved'):
            c.validate_label_audit(audit, questions, synthetic=False)
        audit['target_origin'] = 'scorer_output'
        with self.assertRaises(c.ContractError):
            c.validate_label_audit(audit, questions, synthetic=True)

    def test_gold_audit_never_changes_features_or_mapping(self):
        before = self.artifacts['questions.json']
        self.alter_artifact('label-audit.json', lambda a: a['gold_references'].update({'demo-q1': ['Entirely different']}))
        c.validate_bundle(self.manifest, self.artifacts)
        self.assertEqual(self.artifacts['questions.json'], before)
        self.alter_artifact('candidates.json', lambda q: q[0].update(gold='Paris'))
        self.assert_bundle_rejected()

    def test_forbidden_roles_without_any_fitting(self):
        c.validate_role('reranker_train', 'fit_boundary')
        c.validate_role('internal_validation', 'selection_boundary')
        c.validate_role('official_dev', 'final_boundary')
        for role in ('bert_train', 'internal_validation', 'official_dev', 'synthetic'):
            with self.assertRaises(c.ContractError):
                c.validate_role(role, 'fit_boundary')
        with self.assertRaises(c.ContractError):
            c.validate_role('official_dev', 'selection_boundary')

    def production_split_evidence(self):
        split = c.decode(self.artifacts['split-manifest.json'])
        split['partitions'][0]['role'] = 'reranker_train'
        for index, role in enumerate(('bert_train', 'internal_validation', 'official_dev')):
            aid, qid, content = 'article-' + role, 'q-' + role, 'Exact complete article ' + str(index)
            split['articles'].append({'article_id': aid, 'text': content, 'sha256': c.sha256(content.encode())})
            split['partitions'].append({'role': role, 'question_ids': [qid], 'article_ids': [aid]})
        return split

    def test_split_disjoint_question_article_and_content(self):
        split = self.production_split_evidence()
        check = lambda s: c.validate_splits(s, self.contexts, self.records, 'reranker_train', synthetic=False)
        check(split)
        mutations = [lambda s: s['partitions'][1]['question_ids'].append('demo-q1'),
                     lambda s: s['partitions'][1]['article_ids'].append('demo-article-A'),
                     lambda s: s['articles'][1].update(text=s['articles'][0]['text'], sha256=s['articles'][0]['sha256']),
                     lambda s: s.update(content_scope='context_snippets_only'),
                     lambda s: s['articles'][0].update(sha256='wrong'),
                     lambda s: s['partitions'].pop()]
        for mutate in mutations:
            broken = copy.deepcopy(split)
            mutate(broken)
            with self.assertRaises(c.ContractError):
                check(broken)

    def test_mapping_is_illustration_exact_coverage_supplied_text(self):
        for mapping in ({}, {'demo-q1': 'paris'}, {'demo-q1': 'Paris', 'extra': 'Ada'}):
            questions, _ = self.adapt()
            with self.assertRaises(c.ContractError):
                c.validate_mapping(mapping, questions)
        self.alter_artifact('mapping-illustration.json', lambda m: m.update(status='prediction_result'))
        self.assert_bundle_rejected()

    def test_only_boundary_modules_imported(self):
        import sys
        for name in ('reranker', 'torch', 'transformers', 'datasets'):
            self.assertNotIn(name, sys.modules)


    def test_unpaired_surrogate_and_exponent_overflow_regressions(self):
        for value in ('dataset-' + chr(0xd800), chr(0xdfff)):
            with self.assertRaises(c.ContractError):
                c.identifier(value, 'id')
            manifest = c.decode(self.manifest)
            manifest['dataset_id'] = value
            with self.assertRaises(c.ContractError):
                c.validate_bundle(json.dumps(manifest).encode(), self.artifacts)
        for raw in (b'{"unexpected":1e999}', b'[-1e999]',
                    json.dumps({'unexpected': chr(0xd800)}).encode()):
            with self.assertRaises(c.ContractError):
                c.decode(raw)
            self.setUp()
            self.artifacts['provenance.json'] = raw
            manifest = c.decode(self.manifest)
            manifest['artifact_sha256']['provenance.json'] = c.sha256(raw)
            with self.assertRaises(c.ContractError):
                c.validate_bundle(c.encode(manifest), self.artifacts)

    def test_variable_candidate_counts_and_token_tie_breaks(self):
        second = copy.deepcopy(self.records[0])
        second['question_id'] = 'demo-q2'
        second['occurrences'] = second['occurrences'][:1]
        self.records.append(second)
        questions, _ = self.adapt()
        self.assertEqual([len(q['candidates']) for q in questions], [2, 1])
        o = self.records[0]['occurrences'][0]
        self.records[0]['occurrences'] += [{**o, 'start_token_index': 2, 'end_token_index': 3},
                                         {**o, 'start_token_index': 2, 'end_token_index': 2}]
        _, sidecar = self.adapt()
        winner = sidecar[0]['selected'][0]
        self.assertEqual((winner['start_token_index'], winner['end_token_index']), (2, 2))

    def test_policy_and_missing_raw_fields(self):
        for key in tuple(self.records[0]['occurrences'][0]):
            self.setUp()
            self.records[0]['occurrences'][0].pop(key)
            with self.assertRaises(c.ContractError):
                self.adapt()
        for value in (True, 20.0, 21):
            policy = c.candidate_policy()
            policy['max_candidates'] = value
            with self.assertRaises(c.ContractError):
                c.validate_policy(policy)


    def test_remaining_order_end_then_context_and_context_identity(self):
        second_context = {**self.contexts[0], 'context_id': 'demo-context-B'}
        self.contexts.append(second_context)
        ada = self.records[0]['occurrences'][1]
        self.records[0]['occurrences'] += [{**ada, 'context_id': 'demo-context-B'},
                                         {**ada, 'end': 9, 'answer': 'Ada wrote', 'end_token_index': 1}]
        questions, _ = self.adapt()
        self.assertEqual([o['candidate_id'] for o in questions[0]['candidates']],
                         ['demo-context-A:19:24', 'demo-context-A:0:3',
                          'demo-context-B:0:3', 'demo-context-A:0:9'])
        self.contexts[1]['article_id'] = 'different-article'
        with self.assertRaises(c.ContractError):
            self.adapt()
        self.contexts[1] = copy.deepcopy(self.contexts[0])
        with self.assertRaisesRegex(c.ContractError, 'duplicate context'):
            self.adapt()


if __name__ == '__main__':
    unittest.main()
