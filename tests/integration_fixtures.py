"""Hand-authored synthetic bundles, using the actual PR20 adapter and schema."""
import copy
from offline_handoff import contract as c
from offline_handoff.fixtures import synthetic_bundle


def bundle(group='basic', prefix='train', *, validation=False, equal=0.0, no_signal=False):
    manifest_bytes, _ = synthetic_bundle(group)
    manifest = c.decode(manifest_bytes)
    if validation:
        rows = [('tie', ['  Thé 😀!  ', '  Thé 😀!  '], [(2, 3), (2, 3)], [0, 0]),
                ('shift', ['Zulu', 'Other'], [(900, 400), (1000, 500)], [0, 0]),
                ('single', [' The, Ω 😀 '], [(0, 0)], [0])]
    else:
        rows = [('signal', ['Paris', 'Ada'], [(4, 3), (2, 1)], [0 if no_signal else 1, 0]),
                ('zero', ['Zero', 'None'], [(20, -3), (15, -1)], [equal, equal]),
                ('single', ['Only'], [(100, 5)], [0]),
                ('equal', ['Same', 'Equal'], [(6, 7), (7, 8)], [.4, .4])]
    contexts, records, targets, articles, golds = [], [], {}, [], {}
    for label, answers, logits, labels in rows:
        qid, cid, aid = (f'{prefix}-{label}-{kind}' for kind in ('q', 'c', 'a'))
        context_text = f'{prefix} {label}: '
        occurrences = []
        for token, (answer, (left, right)) in enumerate(zip(answers, logits)):
            start = len(context_text)
            context_text += answer + ' | '
            occurrences.append(dict(context_id=cid, start=start, end=start + len(answer),
                answer=answer, window_index=0, start_token_index=token, end_token_index=token,
                is_context=True, start_logit=left, end_logit=right, combined_score=left + right))
        baseline_index = 1 if label == 'tie' else 0
        baseline = {k: occurrences[baseline_index][k] for k in ('context_id', 'start', 'end', 'answer')}
        if label == 'tie':
            weak = copy.deepcopy(occurrences[baseline_index])
            weak.update(window_index=2, start_logit=1, end_logit=1, combined_score=2)
            occurrences.insert(0, weak)
        contexts.append(dict(context_id=cid, article_id=aid, text=context_text))
        records.append(dict(question_id=qid, question='Where is it?', article_id=aid,
                            baseline=baseline, occurrences=occurrences))
        articles.append(dict(article_id=aid, text=context_text, sha256=c.sha256(context_text.encode('utf-8'))))
        # Labels are hand-authored; never official evaluator results.
        unique = {c.span_id(o): value for o, value in zip(occurrences[-len(answers):], labels)}
        targets[qid] = unique
        golds[qid] = [answers[0]]
    config, policy = c.feature_config(group), c.candidate_policy()
    questions, provenance = c.adapt_candidates(contexts, records, config, policy)
    split = dict(schema_id=c.SCHEMA_ID, content_scope='complete_exact_article_text',
                 articles=articles, partitions=[dict(role='synthetic',
                 question_ids=[q['question_id'] for q in questions], article_ids=[a['article_id'] for a in articles])])
    audit = dict(status=c.SYNTHETIC, gold_references=golds,
                 scorer={k:c.UNAVAILABLE for k in ('source_reference','sha256','version','invocation_revision')},
                 target_units='[0,1]', multi_gold_rule=c.UNAVAILABLE,
                 target_origin='hand_authored_synthetic_not_scorer_outputs',
                 no_signal_policy='retain_inputs;model_behavior_deferred')
    objects = {'contexts.json': contexts, 'candidates.json': records, 'questions.json': questions,
               'targets.json': targets, 'label-audit.json': audit, 'provenance.json': provenance,
               'split-manifest.json': split, 'mapping-illustration.json': {
               'status':'illustration_not_prediction_result',
               'mapping': {q['question_id']:q['candidates'][0]['answer'] for q in questions}}}
    artifacts = {name:c.encode(value) for name,value in objects.items()}
    manifest.update(dataset_id=prefix, split_manifest_sha256=c.sha256(artifacts['split-manifest.json']),
                    artifact_sha256={name:c.sha256(raw) for name,raw in artifacts.items()})
    raw = c.encode(manifest)
    c.validate_bundle(raw, artifacts)
    return raw, artifacts


def assignments(group='basic', **kwargs):
    return {'reranker_train': bundle(group, **kwargs),
            'internal_validation': bundle(group, 'validation', validation=True)}


def dataset(ids):
    return c.encode({'data':[{'paragraphs':[{'qas':[{'id':qid} for qid in ids]}]}]})
