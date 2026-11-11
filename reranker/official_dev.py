"""Label-free official-dev validation and frozen-model prediction.

This separate evaluation path leaves the train/internal-validation adapter and
its role restrictions unchanged. It never fits a model or loads training targets.
"""
import json
from pathlib import Path
import tempfile

import numpy as np

from offline_handoff import contract as c
from . import team_candidates as team
from .core import Reranker, questions_from_records


def source_records(dataset):
    """Extract only identifiers, questions and contexts, excluding annotations."""
    records = {}
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qid = qa['id']
                c.require(qid not in records, 'duplicate source question ID')
                records[qid] = {'question': qa['question'].strip(),
                                'article_id': article['title'], 'context': paragraph['context']}
    c.require(bool(records), 'empty source dataset')
    return records


def membership(records):
    return {'questions': set(records),
            'articles': {v['article_id'] for v in records.values()},
            'context_hashes': {c.sha256(v['context'].encode('utf-8')) for v in records.values()}}


def adapt_official_dev(raw, *, expected, excluded, generation):
    data = c.decode(raw)
    c.keys(data, ('schema_version', 'metadata', 'questions'), 'dev export')
    c.require(type(data['schema_version']) is int and data['schema_version'] == 1, 'candidate schema')
    meta = data['metadata']
    c.keys(meta, team.META, 'dev metadata')
    c.require(meta['split'] == 'dev', 'official-dev role required')
    for key in ('checkpoint', 'examples_path', 'features_path', 'device', 'numpy_version', 'torch_version'):
        c.text(meta[key], key, True)
    for key in ('max_answer_length', 'n_best', 'max_candidates', 'batch_size'):
        c.integer(meta[key], key, 1)
    for key in ('max_answer_length', 'n_best', 'max_candidates', 'batch_size', 'device', 'max_context_only', 'tie_break'):
        c.require(type(meta[key]) is type(generation[key]) and meta[key] == generation[key],
                  'frozen generation mismatch: ' + key)
    c.require(meta['n_best'] == meta['max_candidates'] == 20 and meta['tie_break'] == team.TIE
              and meta['max_context_only'] is False, 'unsupported dev selection policy')
    c.require(meta['logit_dtype'] in ('float32', 'float64'), 'unsupported dev dtype')
    scalar = np.dtype(meta['logit_dtype']).type
    c.keys(excluded, ('questions', 'articles', 'context_hashes'), 'excluded membership')
    current = membership(expected)
    for key in current:
        c.require(not current[key] & set(excluded[key]), 'dev/source overlap: ' + key)
    c.require(type(data['questions']) is list and data['questions'], 'nonempty dev questions required')
    records, seen, windows = [], set(), {}
    for q in data['questions']:
        c.keys(q, team.QUESTION, 'dev question')
        for key in team.QUESTION[:-1]:
            c.text(q[key], key, True)
        qid = q['question_id']
        c.require(qid in expected and qid not in seen, 'unknown/duplicate dev question')
        seen.add(qid)
        c.require(all(q[k] == expected[qid][k] for k in ('question', 'article_id', 'context')),
                  'dev/source text mismatch')
        identity = json.dumps([q['article_id'], q['context']], ensure_ascii=False).encode('utf-8')
        c.require(q['context_id'] == c.sha256(identity), 'dev context ID mismatch')
        spans = q['candidates']
        c.require(type(spans) is list and 1 <= len(spans) <= 20, 'require 1-20 dev spans')
        candidates, unique, previous = [], set(), float('inf')
        for span in spans:
            c.keys(span, team.CANDIDATE, 'dev candidate')
            c.text(span['answer'], 'answer', True)
            for key in ('start', 'end', 'window_index', 'start_token_index', 'end_token_index'):
                c.integer(span[key], key, 0)
            c.integer(span['span_token_count'], 'span_token_count', 1)
            window = span['window_index']
            c.require(window not in windows or windows[window] == qid, 'dev window ownership')
            windows[window] = qid
            start, end = span['start'], span['end']
            c.require(0 <= start < end <= len(q['context']) and q['context'][start:end] == span['answer'],
                      'dev text/span mismatch')
            c.require((start, end) not in unique, 'duplicate dev span')
            unique.add((start, end))
            c.require(span['span_token_count'] == span['end_token_index'] - span['start_token_index'] + 1
                      and span['span_token_count'] <= meta['max_answer_length'], 'dev token count')
            team._score(span, scalar)
            c.require(span['combined_score'] <= previous, 'dev candidate order')
            previous = span['combined_score']
            candidates.append({'candidate_id': f"{q['context_id']}:{start}:{end}", 'answer': span['answer'],
                               'features': c.features(q['question'], span, 'full')})
        records.append({'question_id': qid, 'candidates': candidates})
    c.require(seen == set(expected), 'incomplete dev coverage')
    return {'questions': records, 'metadata': meta,
            'baseline': {q['question_id']: q['candidates'][0]['answer'] for q in records}}


def state_bytes(model):
    with tempfile.TemporaryDirectory(prefix='official-dev-state-') as directory:
        path = Path(directory) / 'model.json'
        model.save(path)
        return path.read_bytes()


def predict_frozen(model_path, specification, records):
    """No fitting; the expected model digest comes from the frozen comparison."""
    raw = Path(model_path).read_bytes()
    c.require(c.sha256(raw) == specification['model']['sha256'], 'frozen model hash mismatch')
    model = Reranker.load(model_path)
    c.require(list(model.features.names) == specification['features'], 'frozen feature mismatch')
    c.require(model.alpha == specification['alpha'], 'frozen alpha mismatch')
    c.require(specification['eval_top_k'] == 20, 'only frozen top-20 comparison enabled')
    before = state_bytes(model)
    predictions = model.predict(questions_from_records(records))
    c.require(state_bytes(model) == before, 'dev prediction changed fitted state')
    c.require(Path(model_path).read_bytes() == raw, 'model file changed during dev inference')
    return predictions
