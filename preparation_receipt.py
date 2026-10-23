"""Read-only verification of pinned preparation receipts; no tokenization or models."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

import squad_preparation as prep  # This module imports only the standard library.

FILES = frozenset(('split-manifest.json', 'selection.json', 'pilot-contexts.json',
                   'pilot-inputs.jsonl', 'pilot-gold-audit.jsonl', 'audit-summary.json',
                   'provenance.json'))
MAX_TOTAL_BYTES = 64 * 1024 * 1024
PRODUCER_SHA256 = '57272f6f3f79e267486204901cf196e93eb4b4168719212ea52eb94be132aecb'
IDENTITY = {'question_id', 'article_id', 'context_sha256', 'role'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected), 'unexpected keys/schema')


def equal(actual, expected, message):
    # JSON comparison distinguishes bool from int and rejects unexpected fields.
    require(prep.canonical(actual) == prep.canonical(expected), message)


def integer(value, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), 'invalid integer')


def string(value):
    require(isinstance(value, str) and bool(value), 'empty or invalid string')


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA-256')


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def finite(value):
    number = float(value)
    require(math.isfinite(number), 'nonfinite JSON number')
    return number


def parse(text):
    return json.loads(text, object_pairs_hook=pairs, parse_constant=finite, parse_float=finite)


def load(root, pin):
    sha(pin)
    require(not root.is_symlink() and root.is_dir(), 'receipt directory required')
    require({p.name for p in root.iterdir()} == FILES | {'artifact-hashes.json'}, 'incomplete or extra artifacts')
    paths = [root/n for n in sorted(FILES | {'artifact-hashes.json'})]
    require(all(not p.is_symlink() and p.is_file() for p in paths), 'regular artifacts required')
    require(sum(p.stat().st_size for p in paths) <= MAX_TOTAL_BYTES, 'receipt size limit exceeded')
    raw = {}
    for path in paths:
        with path.open('rb') as stream:
            raw[path.name] = stream.read(MAX_TOTAL_BYTES + 1)
        require(sum(map(len, raw.values())) <= MAX_TOTAL_BYTES, 'receipt size limit exceeded')
    require(hashlib.sha256(raw['artifact-hashes.json']).hexdigest() == pin, 'inventory pin mismatch')
    inventory = parse(raw.pop('artifact-hashes.json'))
    keys(inventory, FILES)
    for name, content in raw.items():
        sha(inventory[name])
        require(hashlib.sha256(content).hexdigest() == inventory[name], f'artifact hash mismatch: {name}')
    return {name:([parse(line) for line in data.splitlines()] if name.endswith('.jsonl') else parse(data))
            for name, data in raw.items()}


def provenance(p):
    expected = {
        'dataset_id':prep.DATASET, 'dataset_revision':prep.DATASET_REVISION, 'configuration':'plain_text',
        'source_split':'train', 'source_path':'plain_text/train-00000-of-00001.parquet',
        'tokenizer_id':prep.TOKENIZER, 'tokenizer_revision':prep.TOKENIZER_REVISION,
        'code_sha256':PRODUCER_SHA256, 'max_length':384, 'stride':128, 'truncation':'only_second',
        'pair_order':['question','context'], 'offsets':'Unicode code points; half-open',
        'padding':False, 'model_weights_loaded':False, 'scorer_executed':False}
    keys(p, set(expected) | {'assets','python','packages'})
    equal({k:p[k] for k in expected}, expected, 'incompatible preparation provenance')
    keys(p['assets'], prep.ASSET_HASHES)
    sizes = dict(zip(prep.ASSET_HASHES, (14458314,466062,48,570,231508)))
    for name, digest in prep.ASSET_HASHES.items():
        equal(p['assets'][name], {'sha256':digest,'bytes':sizes[name]}, 'asset identity mismatch')
    string(p['python']); keys(p['packages'], {'pyarrow','tokenizers'})
    for value in p['packages'].values(): string(value)


def manifest(m):
    keys(m, {'schema_id','status','seed','allocation','articles','questions','counts','component_sizes'})
    fixed = {'schema_id':'cs4248.squad-preparation/0.1.0-draft',
             'status':'provisional_not_team_agreed','seed':42,
             'allocation':'80/10/10 by sorted, seeded-shuffled connected article components; floor cut points'}
    equal({k:m[k] for k in fixed}, fixed, 'unsupported split specification')
    articles, questions = m['articles'], m['questions']
    require(isinstance(articles,dict) and articles and isinstance(questions,dict) and questions, 'empty manifest')
    owners, adjacency = {}, {aid:set() for aid in articles}
    for aid, a in articles.items():
        keys(a, {'title','contexts','role'}); string(a['title'])
        require(aid == prep.digest(a['title']), 'article identity mismatch')
        require(a['role'] in prep.ROLES, 'invalid role')
        cs = a['contexts']
        require(isinstance(cs,list) and cs, 'empty article contexts')
        for cid in cs: sha(cid)
        require(cs == sorted(set(cs)), 'duplicate or unordered article contexts')
        for cid in cs:
            if cid in owners:
                other = owners[cid]
                require(articles[other]['role'] == a['role'], 'exact context overlap across roles')
                adjacency[aid].add(other); adjacency[other].add(aid)
            owners[cid] = aid
    actual_contexts = {aid:set() for aid in articles}
    for qid, q in questions.items():
        string(qid); keys(q, {'article_id','context_sha256','role'})
        require(q['article_id'] in articles, 'unknown article')
        a = articles[q['article_id']]
        require(q['role'] == a['role'] and q['context_sha256'] in a['contexts'], 'question identity/role mismatch')
        actual_contexts[q['article_id']].add(q['context_sha256'])
    for aid, cs in actual_contexts.items():
        require(cs == set(articles[aid]['contexts']), 'article context coverage mismatch')
    counts = {r:{'articles':sum(a['role']==r for a in articles.values()),
                 'questions':sum(q['role']==r for q in questions.values()),
                 'contexts':len({q['context_sha256'] for q in questions.values() if q['role']==r})}
              for r in prep.ROLES}
    equal(m['counts'], counts, 'split counts mismatch')
    seen, components = set(), []
    for aid in sorted(articles):
        if aid in seen: continue
        pending, group = [aid], []
        while pending:
            current = pending.pop()
            if current in seen: continue
            seen.add(current); group.append(current)
            pending.extend(adjacency[current] - seen)
        components.append(sorted(group))
    components.sort()
    require(len(components) >= 10, 'too few article components')
    equal(m['component_sizes'], sorted(map(len,components)), 'component sizes mismatch')
    random.Random(42).shuffle(components)
    n = len(components)
    for i, group in enumerate(components):
        role = prep.ROLES[0 if i < n*8//10 else 1 if i < n*9//10 else 2]
        require(all(articles[a]['role']==role for a in group), 'split allocation mismatch')


def classification(value, statuses):
    require(isinstance(value,dict) and value.get('status') in statuses, 'invalid alignment status')
    if value['status'] == 'exact':
        keys(value, {'status','start_token','end_token'})
        integer(value['start_token']); integer(value['end_token'], value['start_token'])
    else:
        keys(value, {'status'})


def window(w, qid, index, question, context):
    keys(w, {'question_id','window_index','input_ids','attention_mask','token_type_ids','offsets','sequence_ids'})
    equal([w['question_id'],w['window_index']], [qid,index], 'window identity/order mismatch')
    arrays = [w[k] for k in ('input_ids','attention_mask','token_type_ids','offsets','sequence_ids')]
    require(all(isinstance(a,list) for a in arrays), 'window arrays required')
    n = len(arrays[0]); integer(n,1,384)
    require(all(len(a)==n for a in arrays), 'unequal window array lengths')
    for token, mask, kind, offset, seq in zip(*arrays):
        integer(token,0,30521); integer(mask,1,1); integer(kind,0,1)
        require(seq is None or type(seq) is int and seq in (0,1), 'invalid sequence ID')
        require(isinstance(offset,list) and len(offset)==2, 'invalid offset pair')
        start,end = offset; integer(start); integer(end,start)
        if seq is None:
            equal(offset,[0,0],'special token offsets must be zero')
        else:
            integer(end,start+1,len(context if seq==1 else question))
            require(kind==seq, 'token/sequence type mismatch')
    require(1 in w['sequence_ids'], 'context token coverage missing')


def records(data, m, selected):
    contexts = data['pilot-contexts.json']
    keys(contexts, {m['questions'][q]['context_sha256'] for q in selected})
    for cid, text in contexts.items():
        string(text); require(prep.digest(text)==cid, 'context text hash mismatch')
    inputs, audits = data['pilot-inputs.jsonl'], data['pilot-gold-audit.jsonl']
    require(len(inputs)==len(audits)==len(selected), 'question coverage mismatch')
    window_counts, coverage_counts, exceptions = Counter(), Counter(), []
    total_windows = 0
    for qid, item, audit in zip(selected,inputs,audits):
        keys(item, IDENTITY | {'question','windows'})
        keys(audit, IDENTITY | {'golds','windows','coverage','context_capacity'})
        identity = {'question_id':qid, **m['questions'][qid]}
        for record in (item,audit):
            equal({k:record[k] for k in IDENTITY}, identity, 'question identity/order/role mismatch')
        string(item['question']); context = contexts[item['context_sha256']]
        golds = audit['golds']
        require(isinstance(golds,list) and golds, 'nonempty gold list required')
        for gold in golds:
            keys(gold, {'start','end','text'}); string(gold['text'])
            integer(gold['start']); integer(gold['end'],gold['start']+1,len(context))
            require(context[gold['start']:gold['end']]==gold['text'], 'gold text mismatch')
        windows = item['windows']
        require(isinstance(windows,list) and windows, 'nonempty window list required')
        require(isinstance(audit['windows'],list) and len(audit['windows'])==len(windows), 'window coverage mismatch')
        exact = [0]*len(golds)
        integer(audit['context_capacity'],129,381)
        for index, (w, wa) in enumerate(zip(windows,audit['windows'])):
            window(w,qid,index,item['question'],context)
            require(audit['context_capacity']==384-w['sequence_ids'].count(0)-3, 'context capacity mismatch')
            expected = prep.audit_window(context,golds,w['offsets'],w['sequence_ids'])
            equal(wa, {'window_index':index,'answers':expected}, 'window alignment mismatch')
            for j, answer in enumerate(expected):
                window_counts[answer['status']] += 1
                exact[j] += answer['status']=='exact'
        total_windows += len(windows)
        require(isinstance(audit['coverage'],list) and len(audit['coverage'])==len(golds), 'gold coverage mismatch')
        for hits, coverage in zip(exact,audit['coverage']):
            keys(coverage, {'status','exact_windows','full_context'})
            integer(coverage['exact_windows'])
            require(coverage['exact_windows']==hits, 'exact window count mismatch')
            full = coverage['full_context']
            classification(full, {'exact','unrepresentable_boundary','outside','partial'})
            if hits:
                require(full['status']=='exact', 'full context contradicts exact window')
                status = 'exact'
            elif full['status']!='exact':
                status = 'unrepresentable_boundary'
            else:
                length = full['end_token']-full['start_token']+1
                require(length>129, 'unexplained alignment failure')
                status = 'answer_longer_than_window' if length>audit['context_capacity'] else 'crosses_window_boundaries'
            require(coverage['status']==status, 'coverage status mismatch')
            coverage_counts[status] += 1
        if any(c['status']!='exact' for c in audit['coverage']):
            exceptions.append({'question_id':qid,'coverage':audit['coverage']})
    return total_windows, window_counts, coverage_counts, exceptions


def _verify(root, pin):
    data = load(root,pin)
    provenance(data['provenance.json'])
    m = data['split-manifest.json']; manifest(m)
    selection = data['selection.json']
    keys(selection, {'question_ids','selection_uses_gold'})
    require(selection['selection_uses_gold'] is False, 'gold-dependent selection refused')
    selected = selection['question_ids']
    require(isinstance(selected,list), 'selection must be an ordered list')
    integer(len(selected),1,1000)
    equal(selected, prep.select_pilot(m,len(selected)), 'selection mismatch')
    total_windows, wc, cc, exceptions = records(data,m,selected)
    expected = {
        'status':'complete_with_alignment_exceptions' if exceptions else 'complete',
        'processed_questions':len(selected),
        'sample_counts':dict(Counter(m['questions'][q]['role'] for q in selected)),
        'window_gold_counts':dict(wc), 'gold_coverage':dict(cc), 'exceptions':exceptions,
        'split_reproduced':True, 'article_question_context_disjoint':True,
        'max_article_component':max(m['component_sizes']),
        'component_fraction_max':max(m['component_sizes'])/len(m['articles']),
        'readiness':'preparation audit only; no trained checkpoint or QA accuracy'}
    equal(data['audit-summary.json'], expected, 'audit summary mismatch')
    return {'status':'verified_preparation_receipt', 'inventory_sha256':pin,
            'processed_questions':len(selected), 'windows':total_windows,
            'alignment_exception_questions':len(exceptions), 'gold_coverage':dict(cc),
            'split_status':m['status'], 'training_readiness_established':False}


def verify_receipt(root, expected_inventory_sha256):
    """Validate existing saved bytes against an independently trusted inventory pin.

    Does not establish authenticity of the caller's pin or retokenize the source.
    Full-context classifications and execution history remain producer assertions.
    """
    try:
        return _verify(Path(root),expected_inventory_sha256)
    except (OSError, TypeError, KeyError, IndexError, RecursionError, UnicodeError) as exc:
        raise ValueError(f'malformed or unreadable preparation receipt: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--inventory-sha256',required=True,help='independently retained trusted inventory hash')
    args = parser.parse_args()
    try:
        result = verify_receipt(args.receipt,args.inventory_sha256)
    except ValueError as exc:
        parser.exit(1,f'Receipt rejected: {exc}\n')
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
