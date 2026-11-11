"""Frozen all-candidate inference, then course evaluation and error analysis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from reranker.real_training import reload_predict
from reranker.team_candidates import adapt_export
from reranker.core import Reranker, questions_from_records
from scripts.squad_eval import evaluate

DEV_SHA = '46524095a11587a8b10195e6f9131eec81b92d343a683386d81a216d38ac1de9'
EXPECTED = {'bert': (65.403570, 79.287794), 'basic': (63.925090, 76.781898), 'full': (63.892235, 76.772235)}
ROOT = Path(__file__).resolve().parents[1]
TYPES = ['who','what','when','where','why','how','which','other']

def question_type(question):
    match = re.match(r'[A-Za-z]+', question.lstrip())
    word = match.group(0).lower() if match else ''
    return word if word in TYPES[:-1] else 'other'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def predict_all(candidates, models, pins, split):
    """No gold paths/labels are accepted by this function."""
    raw = Path(candidates).read_bytes()
    if split == 'dev' and hashlib.sha256(raw).hexdigest() != DEV_SHA:
        raise ValueError('Official-dev export differs from frozen branch hash')
    batch = adapt_export(raw, group='basic', allow_dev=(split == 'dev'))
    if batch['audit']['metadata']['split'] != split:
        raise ValueError('Wrong split')
    predictions = {'bert': {q['question_id']:q['candidates'][0]['answer'] for q in batch['questions']}}
    for group in ('basic','full'):
        predictions[group], _ = reload_predict(models, group, pins[group]['binding_sha256'], raw,
            allow_provisional_provenance=True, split=split)
    return predictions, batch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['internal_val','dev'], required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--models', type=Path, default=Path('frozen/linear'))
    p.add_argument('--pins', type=Path, default=Path('configs/submission-models.json'))
    p.add_argument('--validation-receipt', type=Path)
    p.add_argument('--original-baseline', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); start=time.monotonic(); pins=read(args.pins)
    if args.split == 'dev':
        if not args.validation_receipt:
            raise ValueError('Verify internal validation first and supply its receipt')
        receipt=read(args.validation_receipt)
        if not (receipt['split']=='internal_val' and receipt['checks_passed'] and receipt['model_pins']==pins):
            raise ValueError('Internal validation receipt does not match frozen models')
    args.output.mkdir(parents=True,exist_ok=False)
    predictions,batch=predict_all(args.candidates,args.models,pins,args.split)
    # Save EVERY prediction map before opening the gold file.
    for name, answers in predictions.items(): save(args.output/(name+'-predictions.json'),answers)
    gold=read(args.gold); articles=gold['data']
    ids=[q['id'] for a in articles for paragraph in a['paragraphs'] for q in paragraph['qas']]
    count=9131 if args.split=='internal_val' else 10570
    assert len(ids)==len(set(ids))==count
    for answers in predictions.values(): assert set(answers)==set(ids)
    source={q['id']:(a['title'],paragraph['context'],q) for a in articles for paragraph in a['paragraphs'] for q in paragraph['qas']}
    for q in batch['audit']['retained_origins']:
        title,context,qa=source[q['question_id']]
        assert (q['article_id'],q['context'],q['question'])==(title,context,qa['question'].strip())
    metrics={}; full_metrics={}; cli={}
    for name, answers in predictions.items():
        # Execute the original course CLI, then compare every question to checked-in wrapper.
        command=[sys.executable,'-X','utf8',str(ROOT/'course/evaluate-v2.0.py'),str(args.gold),str(args.output/(name+'-predictions.json'))]
        run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',check=True)
        cli[name]=json.loads(run.stdout); save(args.output/(name+'-course-cli.json'),cli[name])
        m=evaluate(articles,answers); full_metrics[name]=m
        assert all(abs(cli[name][k]-m[k])<1e-12 for k in ('exact','f1','total'))
        metrics[name]={k:m[k] for k in ('exact','f1','total')}
    spec=importlib.util.spec_from_file_location('course_scorer',ROOT/'course/evaluate-v2.0.py')
    course=importlib.util.module_from_spec(spec); spec.loader.exec_module(course)
    for name,answers in predictions.items():
        em,f1=course.get_raw_scores(articles,answers)
        assert em==full_metrics[name]['exact_per_question'] and f1==full_metrics[name]['f1_per_question']
    base=full_metrics['bert']['exact_per_question']
    for group in ('basic','full'):
        current=full_metrics[group]['exact_per_question']
        metrics[group].update(fixes=sum(current[i]>base[i] for i in ids),breaks=sum(current[i]<base[i] for i in ids),changed=sum(predictions[group][i]!=predictions['bert'][i] for i in ids))
    if args.split=='internal_val':
        for name,(em,f1) in EXPECTED.items():
            assert round(metrics[name]['exact'],6)==em and round(metrics[name]['f1'],6)==f1, metrics
        assert (metrics['basic']['fixes'],metrics['basic']['breaks'])==(339,474)
        assert (metrics['full']['fixes'],metrics['full']['breaks'])==(337,475)
        assert args.original_baseline and predictions['bert']==read(args.original_baseline)
        original=Reranker.load(ROOT/'configs/frozen-basic-model.json').predict(questions_from_records(batch['questions']))
        mismatch=[i for i in ids if original[i]!=predictions['basic'][i]]
        save(args.output/'original-basic-comparison.json',{'different_questions':mismatch,'count':len(mismatch),'full_original':'No original full model/prediction file recovered'})
        assert not mismatch
    else:
        assert round(metrics['bert']['exact'],6)==80.018921 and round(metrics['bert']['f1'],6)==87.553216
    # Question-type breakdown uses exactly the feature extractor's first ASCII word rule.
    rows=[]; examples={}; per_question=[]
    for typ in TYPES:
        subset=[i for i in ids if question_type(source[i][2]['question'])==typ]
        row={'type':typ,'count':len(subset)}
        for name in predictions:
            m=full_metrics[name]
            row[name]={'exact':100*sum(m['exact_per_question'][i] for i in subset)/len(subset) if subset else None,'f1':100*sum(m['f1_per_question'][i] for i in subset)/len(subset) if subset else None}
            if name!='bert': row[name].update(fixes=sum(m['exact_per_question'][i]>base[i] for i in subset),breaks=sum(m['exact_per_question'][i]<base[i] for i in subset))
        row['basic_full_answer_disagreements']=sum(predictions['basic'][i]!=predictions['full'][i] for i in subset)
        rows.append(row)
    for i in ids:
        per_question.append({'id':i,'question_type':question_type(source[i][2]['question']),**{name:{'exact':full_metrics[name]['exact_per_question'][i],'f1':full_metrics[name]['f1_per_question'][i]} for name in predictions}})
    for group in ('basic','full'):
        examples[group]={}
        for label,sign in [('fix',1),('break',-1)]:
            eligible=[i for i in ids if full_metrics[group]['exact_per_question'][i]-base[i]==sign]
            if eligible:
                i=min(eligible,key=lambda i:len(source[i][2]['question'])+sum(len(predictions[n][i]) for n in ['bert',group]))
                examples[group][label]={'id':i,'question':source[i][2]['question'],'references':[a['text'] for a in source[i][2]['answers']],'bert':predictions['bert'][i],'reranked':predictions[group][i]}
    save(args.output/'question-types.json',rows); save(args.output/'examples.json',examples); save(args.output/'per-question.json',per_question)
    report={'split':args.split,'systems':metrics,'model_pins':pins,'checks_passed':True,'candidate_sha256':sha(args.candidates),'gold_sha256':sha(args.gold),'course_scorer_sha256':sha(ROOT/'course/evaluate-v2.0.py'),'repository_scorer_sha256':sha(ROOT/'evaluate-v2.0.py'),'course_repository_per_question_parity':True,'prediction_sha256':{n:sha(args.output/(n+'-predictions.json')) for n in predictions},'wall_seconds':time.monotonic()-start,'command':sys.argv}
    save(args.output/'metrics.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
