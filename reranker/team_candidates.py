"""PR28 label-free conversion and explicit synthetic core fitting; no BERT/scorer."""
import argparse
import json
from pathlib import Path

import numpy as np

from offline_handoff import contract as c

PROFILE = 'cs4248.pr28-candidate-adapter/0.1.0-draft'
PRODUCER_REF = 'e56efaa1aacfa5d547286137375ab398baaeefeb'
META = ('split', 'checkpoint', 'examples_path', 'features_path', 'max_answer_length',
        'n_best', 'max_candidates', 'batch_size', 'device', 'numpy_version',
        'torch_version', 'logit_dtype', 'tie_break', 'max_context_only')
QUESTION = ('question_id', 'question', 'article_id', 'context_id', 'context', 'candidates')
CANDIDATE = ('answer', 'start', 'end', 'window_index', 'start_token_index',
             'end_token_index', 'span_token_count', 'start_logit', 'end_logit', 'combined_score')
TIE = 'first occurrence in baseline traversal'


def _metadata(meta, *, allow_dev=False):
    c.keys(meta, META, 'PR28 metadata')
    allowed = ('reranker_train', 'internal_val', 'dev') if allow_dev else ('reranker_train', 'internal_val')
    c.require(meta['split'] in allowed, 'only train/internal_val adaptation enabled unless dev explicitly enabled')
    for key in ('checkpoint', 'examples_path', 'features_path', 'device', 'numpy_version', 'torch_version'):
        c.text(meta[key], key, True)
    for key in ('max_answer_length', 'n_best', 'max_candidates', 'batch_size'):
        c.integer(meta[key], key, 1)
    c.require(meta['n_best'] == 20 and meta['max_candidates'] == 20, 'unsupported PR28 search limits')
    c.require(meta['tie_break'] == TIE and meta['max_context_only'] is False,
              'unsupported team selection policy')
    c.require(meta['logit_dtype'] in ('float32', 'float64'), 'unsupported logit dtype')
    return np.dtype(meta['logit_dtype']).type


def _score(span, scalar):
    values = [span[k] for k in ('start_logit', 'end_logit', 'combined_score')]
    for value in values:
        c.number(value, 'logit/score')
    try:
        with np.errstate(over='raise', invalid='raise'):
            encoded = [scalar(v) for v in values]
            c.require(all(float(v) == original for v, original in zip(encoded, values)),
                      'exported scalar not representable in declared dtype')
            c.require(float(encoded[0] + encoded[1]) == values[2], 'combined score/dtype mismatch')
    except (OverflowError, FloatingPointError) as exc:
        raise c.ContractError('logit dtype overflow') from exc


def adapt_export(raw, *, group, baseline=None, allow_dev=False):
    """Preserve team order; return strict core records plus separate audit data.

    Baseline comparison is optional and independently supplied. Paths and metadata
    are declarations, not verified checkpoint/source/tokenizer provenance.
    """
    data = c.decode(raw)
    c.keys(data, ('schema_version', 'metadata', 'questions'), 'PR28 export')
    c.require(type(data['schema_version']) is int and data['schema_version'] == 1, 'unsupported candidate schema')
    meta = data['metadata']; scalar = _metadata(meta, allow_dev=allow_dev); config = c.feature_config(group)
    c.require(type(data['questions']) is list and data['questions'], 'nonempty questions required')
    records, origins, seen, articles, contexts = [], [], set(), set(), set()
    window_owners = {}
    for q in data['questions']:
        c.keys(q, QUESTION, 'PR28 question')
        for key in QUESTION[:-1]:
            c.text(q[key], key, True)
        qid = q['question_id']
        c.require(bool(qid.strip()) and qid not in seen, 'empty/duplicate question ID')
        seen.add(qid); articles.add(q['article_id'])
        contexts.add(c.sha256(q['context'].encode('utf-8')))
        identity = json.dumps([q['article_id'], q['context']], ensure_ascii=False).encode('utf-8')
        c.require(q['context_id'] == c.sha256(identity), 'team context identity mismatch')
        spans = q['candidates']
        c.require(type(spans) is list and 1 <= len(spans) <= 20, 'require 1-20 supplied candidates')
        candidates, span_seen, previous = [], set(), float('inf')
        for span in spans:
            c.keys(span, CANDIDATE, 'PR28 candidate'); c.text(span['answer'], 'answer', True)
            for key in ('start', 'end', 'window_index', 'start_token_index', 'end_token_index'):
                c.integer(span[key], key, 0)
            c.integer(span['span_token_count'], 'span_token_count', 1)
            window = span['window_index']
            c.require(window not in window_owners or window_owners[window] == qid,
                      'global window index belongs to multiple questions')
            window_owners[window] = qid
            start, end = span['start'], span['end']
            c.require(0 <= start < end <= len(q['context']) and q['context'][start:end] == span['answer'],
                      'candidate text/Unicode span mismatch')
            c.require(span['span_token_count'] == span['end_token_index'] - span['start_token_index'] + 1
                      and span['span_token_count'] <= meta['max_answer_length'], 'token count/bounds mismatch')
            c.require((start, end) not in span_seen, 'duplicate supplied span')
            span_seen.add((start, end)); _score(span, scalar)
            c.require(span['combined_score'] <= previous, 'candidate scores not descending')
            previous = span['combined_score']
            cid = f"{q['context_id']}:{start}:{end}"
            candidates.append({'candidate_id': cid, 'answer': span['answer'],
                               'features': c.features(q['question'], span, group)})
        records.append({'question_id': qid, 'candidates': candidates})
        origins.append(q)
    if baseline is not None:
        c.keys(baseline, seen, 'baseline question coverage')
        for q in records:
            c.text(baseline[q['question_id']], 'baseline answer')
            c.require(baseline[q['question_id']] == q['candidates'][0]['answer'], 'first answer/baseline mismatch')
    return {'profile_id': PROFILE, 'feature_registry': config, 'questions': records,
            'audit': {'input_sha256': c.sha256(raw), 'inspected_producer_ref': PRODUCER_REF,
                      'metadata': meta, 'retained_origins': origins,
                      'membership': {'questions': sorted(seen), 'articles': sorted(articles),
                                     'context_hashes': sorted(contexts)},
                      'baseline_answer_comparison': 'matched' if baseline is not None else 'not_supplied',
                      'source_checkpoint_tokenizer_verified': False,
                      'feature_policy': 'provisional_not_team_agreed'}}


def prepare_pair(training_raw, validation_raw, *, group):
    """Check role/overlap before synthetic execution; no fitting or label loading."""
    training = adapt_export(training_raw, group=group)
    validation = adapt_export(validation_raw, group=group)
    tm, vm = training['audit']['metadata'], validation['audit']['metadata']
    c.require(tm['split'] == 'reranker_train' and vm['split'] == 'internal_val', 'forbidden fit/validation role')
    for key in ('checkpoint', 'max_answer_length', 'n_best', 'max_candidates', 'logit_dtype',
                'tie_break', 'max_context_only', 'numpy_version', 'torch_version', 'device'):
        c.require(tm[key] == vm[key], f'train/validation metadata mismatch: {key}')
    for key in ('questions', 'articles', 'context_hashes'):
        c.require(not set(training['audit']['membership'][key]) &
                  set(validation['audit']['membership'][key]), f'train/validation {key} overlap')
    return training, validation


def fit_synthetic_pair(training_raw, targets, validation_raw, *, group, target_status, alpha=1.0):
    """Tiny synthetic integration only. Targets are separate, manually authored.

    The caller attests fixture origin; this is not an authenticity detector. No
    real-label training or diagnostic fallback is enabled by this entry point.
    """
    c.require(target_status == 'synthetic_hand_authored_not_scorer_outputs', 'synthetic target attestation required')
    training, validation = prepare_pair(training_raw, validation_raw, group=group)
    from .core import FeatureConfig, Reranker, questions_from_records
    config = training['feature_registry']; c.validate_config(config)
    features = FeatureConfig(config['basic'], config['compatibility'], config['group'])
    model = Reranker(features, alpha=alpha, no_signal='error')
    model.fit(questions_from_records(training['questions']), targets)
    return model, training, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description='Adapt PR28 candidate bytes without model/scorer execution.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=('basic', 'full'), required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args(argv)
    baseline = c.decode(args.baseline.read_bytes()) if args.baseline else None
    result = adapt_export(args.input.read_bytes(), group=args.group, baseline=baseline)
    # Exclusive creation protects source and existing outputs, including symlinks.
    with args.output.open('xb') as out:
        out.write(c.encode(result))


if __name__ == '__main__':
    main()
