"""Invented wire fixtures. Token positions and targets are hand-authored."""
from .contract import (CONTRACT_ID, SCHEMA_ID, FEATURE_SPEC_ID, CORE_COMMIT,
                       SYNTHETIC, UNAVAILABLE, adapt_candidates, candidate_policy,
                       feature_config, encode, sha256, validate_bundle)


def demo_sources():
    contexts = [{'context_id': 'demo-context-A', 'article_id': 'demo-article-A',
                 'text': 'Ada wrote notes in Paris.'}]
    occurrences = []
    for start, end, answer, token, left, right in ((19, 24, 'Paris', 4, 4.0, 3.0),
                                                  (0, 3, 'Ada', 0, 2.0, 1.0)):
        occurrences.append({'context_id': 'demo-context-A', 'start': start, 'end': end,
                            'answer': answer, 'window_index': 0, 'start_token_index': token,
                            'end_token_index': token, 'is_context': True,
                            'start_logit': left, 'end_logit': right, 'combined_score': left + right})
    records = [{'question_id': 'demo-q1', 'question': 'Where did Ada write notes?',
                'article_id': 'demo-article-A',
                'baseline': {key: occurrences[0][key] for key in ('context_id', 'start', 'end', 'answer')},
                'occurrences': occurrences}]
    return contexts, records


def synthetic_bundle(group='basic', zero_targets=False):
    """Create exact-byte synthetic bundle; never identify fake production artifacts."""
    contexts, records = demo_sources()
    config, policy = feature_config(group), candidate_policy()
    questions, provenance = adapt_candidates(contexts, records, config, policy)
    qid = 'demo-q1'
    targets = {qid: {'demo-context-A:19:24': 0.0 if zero_targets else 1.0,
                     'demo-context-A:0:3': 0.0}}
    article = contexts[0]['text']
    split = {'schema_id': SCHEMA_ID, 'content_scope': 'complete_exact_article_text',
             'articles': [{'article_id': 'demo-article-A', 'text': article,
                           'sha256': sha256(article.encode('utf-8'))}],
             'partitions': [{'role': 'synthetic', 'question_ids': [qid],
                             'article_ids': ['demo-article-A']}]}
    audit = {'status': SYNTHETIC, 'gold_references': {qid: ['Paris']},
             'scorer': {key: UNAVAILABLE for key in ('source_reference', 'sha256', 'version', 'invocation_revision')},
             'target_units': '[0,1]', 'multi_gold_rule': UNAVAILABLE,
             'target_origin': 'hand_authored_synthetic_not_scorer_outputs',
             'no_signal_policy': 'retain_inputs;model_behavior_deferred'}
    objects = {'contexts.json': contexts, 'candidates.json': records, 'questions.json': questions,
               'targets.json': targets, 'label-audit.json': audit, 'provenance.json': provenance,
               'split-manifest.json': split,
               'mapping-illustration.json': {'status': 'illustration_not_prediction_result',
                                             'mapping': {qid: 'Paris'}}}
    artifacts = {name: encode(value) for name, value in objects.items()}
    manifest = {'schema_id': SCHEMA_ID, 'contract_id': CONTRACT_ID, 'status': SYNTHETIC,
                'dataset_id': 'hand-authored-demo', 'split_role': 'synthetic',
                'split_manifest_sha256': sha256(artifacts['split-manifest.json']),
                'generator_commit': UNAVAILABLE, 'feature_spec_id': FEATURE_SPEC_ID,
                'feature_code_commit': UNAVAILABLE, 'checkpoint_sha256': UNAVAILABLE,
                'tokenizer_id_and_revision': {'id': UNAVAILABLE, 'revision': UNAVAILABLE},
                'candidate_policy': policy, 'core_commit': CORE_COMMIT, 'feature_registry': config,
                'artifact_sha256': {name: sha256(raw) for name, raw in artifacts.items()}}
    raw_manifest = encode(manifest)
    validate_bundle(raw_manifest, artifacts)
    return raw_manifest, artifacts
