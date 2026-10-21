"""Strict, standard-library-only validation of a provisional software contract."""
import copy
import hashlib
import json
import math
import re

CONTRACT_ID = 'cs4248.reranker-handoff/0.1.0-draft'
SCHEMA_ID = 'cs4248.offline-bundle/0.1.0-draft'
FEATURE_SPEC_ID = 'candidate-features/0.1.0-draft'
CORE_COMMIT = '6c293edf63a0816370c1407a880b33c3ff2a6176'
SYNTHETIC = 'synthetic_illustration_not_production_valid'
UNAVAILABLE = 'unavailable:synthetic-no-model-or-scorer'
BASIC = ('bert_start_logit', 'bert_end_logit', 'span_token_count',
         'answer_has_digit', 'answer_initial_upper')
TYPES = ('who', 'what', 'when', 'where', 'why', 'how', 'which', 'other')
COMPATIBILITY = tuple('q_' + t for t in TYPES) + ('when_digit_match', 'who_initial_upper_match')
ROLES = ('bert_train', 'reranker_train', 'internal_validation', 'official_dev')
FILES = {'contexts.json', 'candidates.json', 'questions.json', 'targets.json',
         'label-audit.json', 'provenance.json', 'split-manifest.json',
         'mapping-illustration.json'}


class ContractError(ValueError):
    """Invalid handoff, optionally with a generation-failure receipt."""
    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt


def require(condition, message):
    if not condition:
        raise ContractError(message)


def keys(value, expected, where):
    require(type(value) is dict and set(value) == set(expected),
            f'{where}: exact keys required: {sorted(expected)}')


def identifier(value, where):
    text(value, where, True)
    require(type(value) is str and bool(value) and value == value.strip()
            and ':' not in value and not any(ord(c) < 32 for c in value),
            f'{where}: nonempty identifier without colon/control/edge whitespace required')


def text(value, where, nonempty=False):
    require(type(value) is str and (bool(value) or not nonempty), f'{where}: text required')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ContractError(f'{where}: unpaired Unicode surrogate') from exc


def integer(value, where, minimum=None):
    require(type(value) is int and (minimum is None or value >= minimum),
            f'{where}: integer required')


def number(value, where):
    require(type(value) in (int, float), f'{where}: numeric value, not boolean, required')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f'{where}: finite number required')


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')


def decode(raw):
    require(type(raw) is bytes, 'artifacts must be exact UTF-8 bytes')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def finite_float(value):
        result = float(value)
        number(result, 'JSON number')
        return result
    def constant(value):
        raise ContractError(f'nonfinite JSON constant: {value}')
    def validate_strings(value):
        if type(value) is str:
            text(value, 'JSON string')
        elif type(value) is list:
            for item in value:
                validate_strings(item)
        elif type(value) is dict:
            for key, item in value.items():
                text(key, 'JSON key')
                validate_strings(item)
    try:
        result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                            parse_constant=constant, parse_float=finite_float)
        validate_strings(result)
        return result
    except ContractError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise ContractError('invalid UTF-8 JSON') from exc


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def feature_config(group):
    require(group in ('basic', 'full'), 'unknown feature group')
    return {'spec_id': FEATURE_SPEC_ID, 'basic': list(BASIC),
            'compatibility': list(COMPATIBILITY), 'group': group}


def validate_config(config):
    keys(config, ('spec_id', 'basic', 'compatibility', 'group'), 'feature registry')
    require(config == feature_config(config['group']), 'feature spec/order/classification mismatch')
    return BASIC if config['group'] == 'basic' else BASIC + COMPATIBILITY


def candidate_policy(max_span_tokens=30):
    return {'offsets': 'unicode-codepoint-half-open', 'scores': 'raw-logit-sum',
            'dedup': 'context-start-end', 'winner_ties': 'window-start_token-end_token',
            'order': 'baseline-first;score-desc,start,end,context-asc',
            'max_candidates': 20, 'max_span_tokens': max_span_tokens}


def validate_policy(policy):
    keys(policy, candidate_policy(), 'candidate policy')
    integer(policy['max_span_tokens'], 'max_span_tokens', 1)
    integer(policy['max_candidates'], 'max_candidates', 1)
    require(policy == candidate_policy(policy['max_span_tokens']), 'candidate policy mismatch')


def context_index(contexts):
    require(type(contexts) is list and bool(contexts), 'contexts: nonempty array required')
    indexed = {}
    for context in contexts:
        keys(context, ('context_id', 'article_id', 'text'), 'context')
        identifier(context['context_id'], 'context_id')
        identifier(context['article_id'], 'article_id')
        text(context['text'], 'context text', True)
        require(context['context_id'] not in indexed, 'duplicate context_id')
        indexed[context['context_id']] = context
    return indexed


def features(question, occurrence, group):
    answer = occurrence['answer']
    alpha = re.search('[A-Za-z]', answer)
    digit = int(any('0' <= c <= '9' for c in answer))
    upper = int(alpha is not None and 'A' <= alpha.group() <= 'Z')
    values = [occurrence['start_logit'], occurrence['end_logit'],
              occurrence['end_token_index'] - occurrence['start_token_index'] + 1, digit, upper]
    if group == 'full':
        token = re.match('[A-Za-z]+', question.lstrip())
        kind = token.group().lower() if token else 'other'
        if kind not in TYPES:
            kind = 'other'
        values += [int(kind == category) for category in TYPES]
        values += [int(kind == 'when') * digit, int(kind == 'who') * upper]
    return dict(zip(BASIC if group == 'basic' else BASIC + COMPATIBILITY, values))


def span_id(occurrence):
    return f"{occurrence['context_id']}:{occurrence['start']}:{occurrence['end']}"


def validate_occurrence(o, contexts, article, policy):
    keys(o, ('context_id', 'start', 'end', 'answer', 'window_index',
             'start_token_index', 'end_token_index', 'is_context',
             'start_logit', 'end_logit', 'combined_score'), 'occurrence')
    identifier(o['context_id'], 'occurrence context_id')
    require(o['context_id'] in contexts, 'unknown context_id')
    context = contexts[o['context_id']]
    require(context['article_id'] == article, 'article identity lost through windowing')
    for name in ('start', 'end', 'start_token_index', 'end_token_index'):
        integer(o[name], name)
    integer(o['window_index'], 'window_index', 0)
    require(type(o['is_context']) is bool, 'is_context: boolean required')
    text(o['answer'], 'occurrence answer')
    for name in ('start_logit', 'end_logit', 'combined_score'):
        number(o[name], name)
    total = o['start_logit'] + o['end_logit']
    number(total, 'logit sum')
    require(o['combined_score'] == total, 'combined score must equal raw logit sum')
    token_count = o['end_token_index'] - o['start_token_index'] + 1
    if not o['is_context']:
        return 'non_context'
    if not 0 <= o['start'] < o['end'] <= len(context['text']):
        return 'invalid_character_bounds'
    if o['start_token_index'] < 0 or token_count <= 0:
        return 'invalid_token_bounds'
    if token_count > policy['max_span_tokens']:
        return 'overlength_tokens'
    require(o['answer'] == context['text'][o['start']:o['end']],
            'answer does not match exact context slice')
    return None


def adapt_candidates(contexts, records, config, policy):
    """Filter supplied occurrences and derive records; never call a model/tokenizer."""
    indexed = context_index(contexts)
    validate_config(config)
    validate_policy(policy)
    require(type(records) is list and bool(records), 'candidates: nonempty array required')
    questions, provenance, seen = [], [], set()
    for record in records:
        keys(record, ('question_id', 'question', 'article_id', 'baseline', 'occurrences'), 'candidate question')
        qid = record['question_id']
        identifier(qid, 'question_id')
        identifier(record['article_id'], 'article_id')
        text(record['question'], 'question text', True)
        require(qid not in seen, 'duplicate question_id')
        seen.add(qid)
        baseline = record['baseline']
        keys(baseline, ('context_id', 'start', 'end', 'answer'), 'baseline')
        identifier(baseline['context_id'], 'baseline context_id')
        integer(baseline['start'], 'baseline start')
        integer(baseline['end'], 'baseline end')
        text(baseline['answer'], 'baseline answer', True)
        require(type(record['occurrences']) is list, 'occurrences: array required')
        winners, origins, coordinates = {}, [], {}
        for o in record['occurrences']:
            reason = validate_occurrence(o, indexed, record['article_id'], policy)
            coordinate = tuple(o[k] for k in ('context_id', 'window_index', 'start_token_index', 'end_token_index'))
            require(coordinate not in coordinates or coordinates[coordinate] == o,
                    'conflicting duplicate window/token origin')
            coordinates[coordinate] = o
            origins.append({'occurrence': copy.deepcopy(o), 'filter_reason': reason})
            if reason is not None:
                continue
            cid = span_id(o)
            rank = lambda item: (-item['combined_score'], item['window_index'],
                                 item['start_token_index'], item['end_token_index'])
            if cid not in winners or rank(o) < rank(winners[cid]):
                winners[cid] = o
        if not winners:
            raise ContractError('candidate_generation_failure: batch incomplete',
                                {'status': 'candidate_generation_failure', 'question_id': qid,
                                 'origins': origins})
        baseline_id = span_id(baseline)
        require(baseline_id in winners and baseline['answer'] == winners[baseline_id]['answer'],
                'baseline missing, filtered, or invalid')
        remaining = sorted((o for cid, o in winners.items() if cid != baseline_id),
                           key=lambda o: (-o['combined_score'], o['start'], o['end'], o['context_id']))
        selected = [winners[baseline_id]] + remaining[:19]
        questions.append({'question_id': qid, 'candidates': [
            {'candidate_id': span_id(o), 'answer': o['answer'],
             'features': features(record['question'], o, config['group'])} for o in selected]})
        provenance.append({'question_id': qid, 'baseline_candidate_id': baseline_id,
                           'origins': origins, 'selected': copy.deepcopy(selected)})
    return questions, provenance


def validate_questions(actual, expected, config):
    names = validate_config(config)
    require(type(actual) is list and len(actual) == len(expected), 'questions: exact batch coverage required')
    for question in actual:
        keys(question, ('question_id', 'candidates'), 'core question')
        identifier(question['question_id'], 'core question_id')
        require(type(question['candidates']) is list and 1 <= len(question['candidates']) <= 20,
                'core candidates: count must be 1..20')
        for candidate in question['candidates']:
            keys(candidate, ('candidate_id', 'answer', 'features'), 'core candidate')
            text(candidate['candidate_id'], 'candidate_id', True)
            text(candidate['answer'], 'answer', True)
            keys(candidate['features'], names, 'features')
            for value in candidate['features'].values():
                number(value, 'feature')
    require(actual == expected, 'questions differ from deterministic adapter output/order/features')


def vectors(questions, config):
    """Project validated records using the explicit ordered registry."""
    names = validate_config(config)
    return [[[c['features'][name] for name in names] for c in q['candidates']]
            for q in questions]


def validate_targets(targets, questions):
    keys(targets, [q['question_id'] for q in questions], 'target question coverage')
    summary = {'single_candidate': [], 'equal_zero': [], 'equal_nonzero': [], 'varying': []}
    for question in questions:
        qid = question['question_id']
        values = targets[qid]
        keys(values, [c['candidate_id'] for c in question['candidates']], 'target candidate coverage')
        for value in values.values():
            number(value, 'target')
            require(0 <= value <= 1, 'target outside [0,1]')
        scores = list(values.values())
        kind = ('single_candidate' if len(scores) == 1 else 'varying' if len(set(scores)) > 1
                else 'equal_zero' if scores[0] == 0 else 'equal_nonzero')
        summary[kind].append(qid)
    return summary


def validate_role(role, purpose):
    """Validate a declared consumer; never perform fitting or model selection."""
    require(role in ROLES + ('synthetic',), 'unknown split role')
    allowed = {'offline_check': ROLES + ('synthetic',), 'fit_boundary': ('reranker_train',),
               'selection_boundary': ('internal_validation',), 'final_boundary': ('official_dev',)}
    require(type(purpose) is str and purpose in allowed and role in allowed[purpose],
            'forbidden split role for intended consumer')


def unique_ids(value, where):
    require(type(value) is list, f'{where}: array required')
    for item in value:
        identifier(item, where)
    require(len(value) == len(set(value)), f'{where}: duplicate IDs')
    return set(value)


def validate_splits(split, contexts, records, role, synthetic):
    keys(split, ('schema_id', 'content_scope', 'articles', 'partitions'), 'split manifest')
    require(split['schema_id'] == SCHEMA_ID, 'split schema mismatch')
    require(split['content_scope'] == 'complete_exact_article_text', 'complete article evidence required')
    require(type(split['articles']) is list and bool(split['articles']), 'articles required')
    articles = {}
    for article in split['articles']:
        keys(article, ('article_id', 'text', 'sha256'), 'article')
        identifier(article['article_id'], 'article_id')
        text(article['text'], 'complete article text', True)
        require(article['article_id'] not in articles, 'duplicate article_id')
        require(article['sha256'] == sha256(article['text'].encode('utf-8')), 'article content hash mismatch')
        articles[article['article_id']] = article
    require(type(split['partitions']) is list and bool(split['partitions']), 'partitions required')
    by_role, used_questions, used_articles, used_content = {}, set(), set(), set()
    for partition in split['partitions']:
        keys(partition, ('role', 'question_ids', 'article_ids'), 'partition')
        part_role = partition['role']
        validate_role(part_role, 'offline_check')
        require(part_role not in by_role, 'duplicate partition role')
        qids = unique_ids(partition['question_ids'], 'partition questions')
        aids = unique_ids(partition['article_ids'], 'partition articles')
        require(aids <= articles.keys(), 'missing complete article evidence')
        require(not qids & used_questions, 'question overlap across partitions')
        require(not aids & used_articles, 'article overlap across partitions')
        hashes = {articles[aid]['sha256'] for aid in aids}
        require(not hashes & used_content, 'identical article content across partitions')
        used_questions |= qids
        used_articles |= aids
        used_content |= hashes
        by_role[part_role] = (qids, aids)
    require(set(by_role) == ({'synthetic'} if synthetic else set(ROLES)), 'split roles incomplete or inconsistent')
    require(used_articles == articles.keys(), 'unassigned complete article evidence')
    require(role in by_role, 'handoff role absent from split manifest')
    qids, aids = by_role[role]
    require({r['question_id'] for r in records} == qids, 'batch question coverage differs from split')
    require({r['article_id'] for r in records} <= aids, 'question article outside declared split')
    for context in contexts:
        require(context['article_id'] in aids, 'context article outside declared split')
        require(context['text'] in articles[context['article_id']]['text'], 'context not in complete article text')
    # Exact snippet duplicates across article IDs are additional evidence, not a
    # semantic near-duplicate detector or proof that supplied articles are complete.
    context_owners = {}
    for context in contexts:
        owner = context_owners.setdefault(context['text'], context['article_id'])
        require(owner == context['article_id'], 'identical context content under different articles')


def validate_mapping(mapping, questions):
    keys(mapping, [q['question_id'] for q in questions], 'illustration mapping coverage')
    for question in questions:
        answer = mapping[question['question_id']]
        text(answer, 'illustration answer')
        require(answer in [c['answer'] for c in question['candidates']], 'mapping text not supplied by candidate')


def validate_label_audit(audit, questions, synthetic):
    keys(audit, ('status', 'gold_references', 'scorer', 'target_units', 'multi_gold_rule',
                 'target_origin', 'no_signal_policy'), 'label audit')
    scorer = audit['scorer']
    keys(scorer, ('source_reference', 'sha256', 'version', 'invocation_revision'), 'scorer provenance')
    require(synthetic, 'supplied official scorer unresolved: real labels refused')
    require(audit['status'] == SYNTHETIC and audit['target_origin'] == 'hand_authored_synthetic_not_scorer_outputs',
            'synthetic label status required')
    require(all(value == UNAVAILABLE for value in scorer.values()), 'synthetic scorer sentinel required')
    require(audit['target_units'] == '[0,1]' and audit['multi_gold_rule'] == UNAVAILABLE,
            'unverified target/scorer rule')
    require(audit['no_signal_policy'] == 'retain_inputs;model_behavior_deferred', 'no-signal policy mismatch')
    keys(audit['gold_references'], [q['question_id'] for q in questions], 'gold reference coverage')
    for references in audit['gold_references'].values():
        require(type(references) is list and bool(references), 'gold reference list required')
        for reference in references:
            text(reference, 'gold reference')


def validate_manifest(manifest, artifacts, purpose):
    keys(manifest, ('schema_id', 'contract_id', 'status', 'dataset_id', 'split_role',
                    'split_manifest_sha256', 'generator_commit', 'feature_spec_id',
                    'feature_code_commit', 'checkpoint_sha256', 'tokenizer_id_and_revision',
                    'candidate_policy', 'core_commit', 'feature_registry', 'artifact_sha256'), 'manifest')
    require(manifest['schema_id'] == SCHEMA_ID and manifest['contract_id'] == CONTRACT_ID,
            'manifest schema/contract version mismatch')
    require(manifest['feature_spec_id'] == FEATURE_SPEC_ID, 'feature spec mismatch')
    require(manifest['core_commit'] == CORE_COMMIT, 'core commit mismatch')
    identifier(manifest['dataset_id'], 'dataset_id')
    require(manifest['status'] in (SYNTHETIC, 'production'), 'unknown provenance status')
    synthetic = manifest['status'] == SYNTHETIC
    validate_role(manifest['split_role'], purpose)
    require(synthetic == (manifest['split_role'] == 'synthetic'), 'synthetic/production role mismatch')
    validate_config(manifest['feature_registry'])
    validate_policy(manifest['candidate_policy'])
    keys(manifest['tokenizer_id_and_revision'], ('id', 'revision'), 'tokenizer provenance')
    for field, width in (('generator_commit', 40), ('feature_code_commit', 40), ('checkpoint_sha256', 64)):
        value = manifest[field]
        require(value == UNAVAILABLE if synthetic else
                type(value) is str and re.fullmatch('[0-9a-f]{' + str(width) + '}', value) is not None,
                f'{field}: explicit provenance required')
    for value in manifest['tokenizer_id_and_revision'].values():
        require(value == UNAVAILABLE if synthetic else
                type(value) is str and bool(value.strip()) and value not in (UNAVAILABLE, 'unknown'),
                'tokenizer identity/revision unresolved')
    keys(manifest['artifact_sha256'], FILES, 'artifact hash map')
    keys(artifacts, FILES, 'artifact set')
    for filename, raw in artifacts.items():
        require(type(raw) is bytes, 'artifact bytes required')
        require(manifest['artifact_sha256'][filename] == sha256(raw), f'artifact hash mismatch: {filename}')
    require(manifest['split_manifest_sha256'] == sha256(artifacts['split-manifest.json']),
            'split manifest hash mismatch')
    return synthetic


def validate_bundle(manifest_bytes, artifacts, purpose='offline_check'):
    """Validate exact bytes; no files, network, model, tokenizer or scorer calls."""
    manifest = decode(manifest_bytes)
    synthetic = validate_manifest(manifest, artifacts, purpose)
    data = {name: decode(raw) for name, raw in artifacts.items()}
    config = manifest['feature_registry']
    expected, provenance = adapt_candidates(data['contexts.json'], data['candidates.json'],
                                            config, manifest['candidate_policy'])
    validate_questions(data['questions.json'], expected, config)
    # Generated sidecar is canonical: same keys/order and complete origins.
    require(encode(data['provenance.json']) == encode(provenance), 'provenance differs from complete source origins/winners')
    validate_splits(data['split-manifest.json'], data['contexts.json'], data['candidates.json'],
                    manifest['split_role'], synthetic)
    signal = validate_targets(data['targets.json'], expected)
    validate_label_audit(data['label-audit.json'], expected, synthetic)
    illustration = data['mapping-illustration.json']
    keys(illustration, ('status', 'mapping'), 'mapping illustration')
    require(illustration['status'] == 'illustration_not_prediction_result', 'mapping must be labeled illustration')
    validate_mapping(illustration['mapping'], expected)
    return {'status': 'valid_synthetic_offline_contract', 'contract_id': CONTRACT_ID,
            'question_count': len(expected), 'candidate_counts': [len(q['candidates']) for q in expected],
            'feature_group': config['group'], 'feature_width': len(validate_config(config)),
            'target_groups': signal, 'model_execution': False, 'scorer_execution': False,
            'production_valid': False}
