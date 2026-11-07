"""Complete-ID preflight and injectable evaluator transport, not an evaluator.

The supplied source's exact hash is unresolved. There is no default runner or
source fetch. A future owner must independently establish the expected hash
and authorize a runner that invokes the supplied file with default options.
"""
import copy
import re
from offline_handoff import contract as c

INVOCATION = 'evaluate-v2.0-default-na/0.1.0-draft'


def _question_ids(data):
    c.require(type(data) is dict and type(data.get('data')) is list, 'dataset data array required')
    ids = []
    for article in data['data']:
        c.require(type(article) is dict and type(article.get('paragraphs')) is list, 'paragraphs required')
        for paragraph in article['paragraphs']:
            c.require(type(paragraph) is dict and type(paragraph.get('qas')) is list, 'qas required')
            for question in paragraph['qas']:
                c.require(type(question) is dict and 'id' in question, 'question id required')
                qid = question['id']
                c.text(qid, 'evaluation question id', True)
                c.require(qid == qid.strip(), 'question id edge whitespace')
                ids.append(qid)
    c.require(bool(ids) and len(ids) == len(set(ids)), 'nonempty unique evaluation IDs required')
    return ids


def preflight_predictions(data_bytes, prediction_bytes):
    """Require every dataset ID exactly once; preserve answers, including empty text."""
    ids = _question_ids(c.decode(data_bytes))
    predictions = c.decode(prediction_bytes)
    c.keys(predictions, ids, 'evaluation prediction coverage')
    for value in predictions.values():
        c.text(value, 'prediction answer')
    return predictions


def invoke_evaluator(data_bytes, prediction_bytes, *, provenance=None, source_bytes=None, runner=None):
    """Dispatch validated bytes to a trusted injected runner(source, data, predictions).

    Runner contract: invoke data_file pred_file (optional output capture only),
    with default no-answer probabilities 0 and threshold 1; no override flags.
    This function checks transport integrity, never scorer semantics or the
    external attestation that the expected hash identifies the supplied file.
    Synthetic doubles are explicitly labelled and do not compute real metrics.
    """
    predictions = preflight_predictions(data_bytes, prediction_bytes)
    c.keys(provenance, ('status', 'source_reference', 'expected_sha256', 'version',
                        'invocation_revision'), 'unresolved evaluator provenance')
    provenance = copy.deepcopy(provenance)
    status = provenance['status']
    c.require(status in ('verified_exact_file', 'synthetic_test_double'), 'supplied evaluator source unresolved')
    c.text(provenance['source_reference'], 'scorer source reference', True)
    c.require(bool(provenance['source_reference'].strip()), 'source reference unresolved')
    c.require(provenance['version'] is None or
              type(provenance['version']) is str and bool(provenance['version'].strip()), 'invalid scorer version')
    digest = provenance['expected_sha256']
    c.require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None,
              'independently verified exact source SHA-256 required')
    c.require(type(source_bytes) is bytes and bool(source_bytes) and c.sha256(source_bytes) == digest,
              'evaluator bytes missing or hash mismatch')
    c.require(provenance['invocation_revision'] == INVOCATION, 'unsupported evaluator invocation/options')
    c.require(callable(runner), 'explicit authorized evaluator runner required')
    result = runner(source_bytes, data_bytes, prediction_bytes)
    return {'status':'synthetic_dispatch_only' if status == 'synthetic_test_double' else 'supplied_dispatch',
            'question_count':len(predictions), 'provenance':provenance,
            'official_scorer_conformance':False, 'result':result}
