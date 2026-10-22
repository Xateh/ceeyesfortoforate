"""Local-only, provisional SQuAD train preparation. No model/scorer execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

DATASET = 'rajpurkar/squad'
DATASET_REVISION = '7b6d24c440a36b6815f21b70d25016731768db1f'
TOKENIZER = 'google-bert/bert-base-uncased'
TOKENIZER_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
ROLES = ('bert_train', 'reranker_train', 'internal_validation')
SEED = 42
MAX_LENGTH = 384
STRIDE = 128
ASSET_HASHES = {
    'train.parquet': 'ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b',
    'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
    'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
    'config.json': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def validate_row(row, *, gold=False):
    if not isinstance(row, dict) or set(row) != {'id','title','context','question','answers'}:
        raise ValueError('unexpected source row schema')
    if any(not isinstance(row[k], str) or not row[k] for k in ('id','title','context','question')):
        raise ValueError('empty or invalid source string')
    a = row['answers']
    if not isinstance(a, dict) or set(a) != {'text','answer_start'}:
        raise ValueError('unexpected answers schema')
    if not isinstance(a['text'], list) or not isinstance(a['answer_start'], list):
        raise ValueError('answers must be lists')
    if not a['text'] or len(a['text']) != len(a['answer_start']):
        raise ValueError('empty or unequal answer lists')
    for text, start in zip(a['text'], a['answer_start']):
        if not isinstance(text, str) or not text or type(start) is not int or start < 0:
            raise ValueError('invalid gold annotation')
        if gold and row['context'][start:start+len(text)] != text:
            raise ValueError(f'gold character mismatch: {row["id"]}')


def build_manifest(rows):
    articles, questions, contexts, parents = {}, {}, {}, {}
    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a
    for row in rows:
        validate_row(row)
        q, title, context = row['id'], row['title'], row['context']
        if q in questions:
            raise ValueError(f'duplicate question ID: {q}')
        aid, cid = digest(title), digest(context)
        if aid not in articles:
            articles[aid] = {'title': title, 'contexts': set()}
            parents[aid] = aid
        elif articles[aid]['title'] != title:
            raise ValueError('article hash collision')
        articles[aid]['contexts'].add(cid)
        if cid in contexts:
            old_text, old_article = contexts[cid]
            if old_text != context: raise ValueError('context hash collision')
            x, y = find(aid), find(old_article)
            parents[max(x,y)] = min(x,y)
        else:
            contexts[cid] = (context, aid)
        questions[q] = {'article_id': aid, 'context_sha256': cid}
    groups = {}
    for aid in sorted(articles): groups.setdefault(find(aid), []).append(aid)
    components = sorted(groups.values())
    if len(components) < 10:
        raise ValueError('at least ten independent article components required')
    random.Random(SEED).shuffle(components)
    n = len(components); cuts = (n*8//10, n*9//10)
    for i, group in enumerate(components):
        role = ROLES[0 if i<cuts[0] else 1 if i<cuts[1] else 2]
        for aid in group:
            articles[aid]['role'] = role
            articles[aid]['contexts'] = sorted(articles[aid]['contexts'])
    for info in questions.values(): info['role'] = articles[info['article_id']]['role']
    counts = {r: {'articles': sum(a['role']==r for a in articles.values()),
                  'questions': sum(q['role']==r for q in questions.values()),
                  'contexts': len({q['context_sha256'] for q in questions.values() if q['role']==r})}
              for r in ROLES}
    result = {'schema_id':'cs4248.squad-preparation/0.1.0-draft',
              'status':'provisional_not_team_agreed', 'seed':SEED,
              'allocation':'80/10/10 by sorted, seeded-shuffled connected article components; floor cut points',
              'articles':articles, 'questions':questions, 'counts':counts,
              'component_sizes':sorted(map(len, components))}
    verify_disjoint(result)
    return result


def verify_disjoint(manifest):
    seen = {'article_id':{}, 'context_sha256':{}}
    for qid, q in manifest['questions'].items():
        role = q['role']
        if role not in ROLES or manifest['articles'][q['article_id']]['role'] != role:
            raise ValueError(f'article role mismatch: {qid}')
        for key, mapping in seen.items():
            value = q[key]
            if value in mapping and mapping[value] != role:
                raise ValueError(f'{key} overlap across roles')
            mapping[value] = role


def select_pilot(manifest, limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('pilot limit must be 1..1000')
    quotas = (limit*8//10, limit*9//10-limit*8//10, limit-limit*9//10)
    selected = []
    for role, quota in zip(ROLES, quotas):
        ids = sorted(q for q, v in manifest['questions'].items() if v['role']==role)
        random.Random(SEED).shuffle(ids)
        if len(ids) < quota: raise ValueError(f'insufficient questions in {role}')
        selected.extend(ids[:quota])
    return selected


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
    return h.hexdigest()


def verified_assets(root):
    result = {}
    for name, expected in ASSET_HASHES.items():
        path = root / name
        if file_hash(path) != expected: raise ValueError(f'asset hash mismatch: {name}')
        result[name] = {'sha256':expected, 'bytes':path.stat().st_size}
    return result


def audit_window(context, golds, offsets, sequence_ids):
    if len(offsets) != len(sequence_ids): raise ValueError('offset length mismatch')
    tokens = [(i,s,e) for i, ((s,e), seq) in enumerate(zip(offsets,sequence_ids)) if seq==1]
    if not tokens: raise ValueError('window has no context tokens')
    if any(not 0 <= s < e <= len(context) for _,s,e in tokens):
        raise ValueError('invalid context token offsets')
    if any(a[1]>b[1] or a[2]>b[2] for a,b in zip(tokens,tokens[1:])):
        raise ValueError('nonmonotonic context offsets')
    results = []
    for gold in golds:
        start, end = gold['start'], gold['end']
        starts = [i for i,s,e in tokens if s==start]
        ends = [i for i,s,e in tokens if e==end]
        if starts and ends and starts[0] <= ends[-1]:
            results.append({'status':'exact','start_token':starts[0],'end_token':ends[-1]})
        elif end <= tokens[0][1] or start >= tokens[-1][2]:
            results.append({'status':'outside'})
        elif start >= tokens[0][1] and end <= tokens[-1][2]:
            results.append({'status':'unrepresentable_boundary'})
        else:
            results.append({'status':'partial'})
    return results


def tokenize_question(row, tokenizer):
    validate_row(row, gold=True)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    q_length = len(tokenizer.encode(row['question'], add_special_tokens=False).ids)
    capacity = MAX_LENGTH - tokenizer.num_special_tokens_to_add(True) - q_length
    if capacity <= STRIDE:
        raise ValueError(f'question leaves insufficient context capacity: {row["id"]}')
    whole = tokenizer.encode(row['context'], add_special_tokens=False)
    golds = [{'start':s,'end':s+len(t),'text':t}
             for t,s in zip(row['answers']['text'],row['answers']['answer_start'])]
    whole_audit = audit_window(row['context'],golds,whole.offsets,[1]*len(whole.ids))
    tokenizer.enable_truncation(max_length=MAX_LENGTH,stride=STRIDE,strategy='only_second')
    encoding = tokenizer.encode(row['question'],row['context'])
    inputs, audits = [], []
    for index, window in enumerate([encoding, *encoding.overflowing]):
        if len(window.ids)>MAX_LENGTH: raise ValueError('maximum length exceeded')
        offsets = [list(x) for x in window.offsets]
        inputs.append({'question_id':row['id'],'window_index':index,
                       'input_ids':window.ids,'attention_mask':window.attention_mask,
                       'token_type_ids':window.type_ids,'offsets':offsets,
                       'sequence_ids':window.sequence_ids})
        a = audit_window(row['context'],golds,offsets,window.sequence_ids)
        for g, result in zip(golds,a):
            if result['status']=='exact':
                s = offsets[result['start_token']][0]; e = offsets[result['end_token']][1]
                if row['context'][s:e] != g['text']: raise ValueError('reconstruction failure')
        audits.append({'window_index':index,'answers':a})
    coverage = []
    for i, full in enumerate(whole_audit):
        exact = sum(w['answers'][i]['status']=='exact' for w in audits)
        if exact:
            status = 'exact'
        elif full['status'] != 'exact':
            # Evidence: full untruncated context itself lacks the original boundaries.
            status = 'unrepresentable_boundary'
        else:
            length = full['end_token']-full['start_token']+1
            if length>capacity: status = 'answer_longer_than_window'
            elif length>STRIDE+1: status = 'crosses_window_boundaries'
            else: raise ValueError(f'unexplained alignment failure: {row["id"]}')
        coverage.append({'status':status,'exact_windows':exact,'full_context':full})
    return inputs, {'question_id':row['id'],'golds':golds,'windows':audits,
                    'coverage':coverage,'context_capacity':capacity}


def iter_rows(path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source = pq.ParquetFile(path)
    expected = pa.schema([('id',pa.string()),('title',pa.string()),
                          ('context',pa.string()),('question',pa.string()),
                          ('answers',pa.struct([('text',pa.list_(pa.string())),
                                               ('answer_start',pa.list_(pa.int32()))]))])
    if not source.schema_arrow.equals(expected,check_metadata=False):
        raise ValueError('pinned Parquet schema mismatch')
    for batch in source.iter_batches(batch_size=256,use_threads=False):
        yield from batch.to_pylist()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n')


def prepare(assets, output, limit=1000):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('pilot limit must be 1..1000')
    if output.exists(): raise FileExistsError(output)
    hashes = verified_assets(assets)
    manifest = build_manifest(iter_rows(assets/'train.parquet'))
    # A second independent streaming pass checks assignment reproducibility.
    repeat = build_manifest(iter_rows(assets/'train.parquet'))
    if manifest != repeat: raise ValueError('nonreproducible split')
    del repeat
    selected = select_pilot(manifest,limit)
    wanted = set(selected)
    sampled = {r['id']:r for r in iter_rows(assets/'train.parquet') if r['id'] in wanted}
    if set(sampled) != wanted: raise ValueError('pilot question coverage mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(assets/'tokenizer.json'))
    output.mkdir(parents=True)
    completed = []
    qid = None
    stage = 'output_setup'
    try:
        write_json(output/'split-manifest.json',manifest)
        write_json(output/'selection.json',{'question_ids':selected,'selection_uses_gold':False})
        contexts = {digest(r['context']):r['context'] for r in sampled.values()}
        write_json(output/'pilot-contexts.json',contexts)
        stats = Counter(); coverage = Counter(); roles = Counter(); exceptions = []
        stage = 'tokenization'
        with (output/'pilot-inputs.jsonl').open('x',encoding='utf-8') as inputs, \
             (output/'pilot-gold-audit.jsonl').open('x',encoding='utf-8') as audit:
            for qid in selected:
                row = sampled[qid]
                windows, record = tokenize_question(row,tokenizer)
                identity = manifest['questions'][qid]
                inputs.write(canonical({'question_id':qid,'question':row['question'],
                                        **identity,'windows':windows})+'\n')
                audit.write(canonical({**record,**identity})+'\n')
                roles[identity['role']] += 1
                for w in record['windows']:
                    stats.update(a['status'] for a in w['answers'])
                coverage.update(g['status'] for g in record['coverage'])
                if any(g['status']!='exact' for g in record['coverage']):
                    exceptions.append({'question_id':qid,'coverage':record['coverage']})
                completed.append(qid)
        if completed != selected: raise ValueError('processed ID ordering/coverage mismatch')
        qid = None
        stage = 'finalization'
        import importlib.metadata
        import platform
        provenance = {'dataset_id':DATASET,'dataset_revision':DATASET_REVISION,
                      'configuration':'plain_text','source_split':'train',
                      'source_path':'plain_text/train-00000-of-00001.parquet',
                      'tokenizer_id':TOKENIZER,'tokenizer_revision':TOKENIZER_REVISION,
                      'assets':hashes,'code_sha256':file_hash(Path(__file__)),
                      'python':platform.python_version(),
                      'packages':{k:importlib.metadata.version(k) for k in ('pyarrow','tokenizers')},
                      'max_length':MAX_LENGTH,'stride':STRIDE,'truncation':'only_second',
                      'pair_order':['question','context'],'offsets':'Unicode code points; half-open',
                      'padding':False,'model_weights_loaded':False,'scorer_executed':False}
        summary = {'status':'complete_with_alignment_exceptions' if exceptions else 'complete',
                   'processed_questions':len(completed),'sample_counts':dict(roles),
                   'window_gold_counts':dict(stats),'gold_coverage':dict(coverage),
                   'exceptions':exceptions,'split_reproduced':True,
                   'article_question_context_disjoint':True,
                   'max_article_component':max(manifest['component_sizes']),
                   'component_fraction_max':max(manifest['component_sizes'])/len(manifest['articles']),
                   'readiness':'preparation audit only; no trained checkpoint or QA accuracy'}
        write_json(output/'provenance.json',provenance)
        write_json(output/'audit-summary.json',summary)
        write_json(output/'artifact-hashes.json',{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
        return summary
    except Exception as exc:
        try:
            write_json(output/'failure.json', {'status':'blocked','error':str(exc),
                       'stage':stage,'completed_question_ids':completed,'failed_question_id':qid})
        except OSError:
            pass  # Preserve the original error even when the filesystem cannot record it.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=1000)
    args = parser.parse_args()
    import os
    import resource
    if len(os.sched_getaffinity(0)) != 1 or not 0 < resource.getrlimit(resource.RLIMIT_AS)[0] <= 2**30:
        parser.error('launch with one CPU affinity and at most 1 GiB RLIMIT_AS')
    controls = {'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                'RAYON_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':''}
    if any(os.environ.get(k)!=v for k,v in controls.items()):
        parser.error('required single-thread/CUDA-disabled environment missing')
    print(canonical(prepare(args.assets,args.output,args.limit)))


if __name__ == '__main__':
    main()
