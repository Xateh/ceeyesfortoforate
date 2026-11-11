"""Replay every table row from packaged inputs, checking predictions and scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.squad_eval import evaluate

def read(p): return json.loads(Path(p).read_bytes())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    root=args.output
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    def run(module,*argv):
        command=[sys.executable,'-X','utf8','-m',module,*map(str,argv)]; commands.append(command)
        proc=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (root/(str(len(commands))+'-'+module+'.log')).write_text(proc.stdout,encoding='utf-8')
        if proc.returncode: raise RuntimeError(proc.stdout)
    for split, gold, output in [('internal_val','data/splits/internal_val.json','internal-val'),('dev','data/dev-v1.1.json','official-dev')]:
        candidates='inputs/inference/internal_val_candidates.json' if split=='internal_val' else 'inputs/official-dev/dev_candidates.json'
        extra=['--original-baseline','inputs/inference/internal_val_predictions.json'] if split=='internal_val' else ['--validation-receipt',root/'internal-val/metrics.json']
        run('scripts.submission_compare','--split',split,'--candidates',candidates,'--gold',gold,'--output',root/output,*extra)
        original='results/internal-val-verified' if split=='internal_val' else 'results/official-dev'
        for name in ['bert','basic','full']:
            assert read(root/output/(name+'-predictions.json'))==read(Path(original)/(name+'-predictions.json'))
        print(output+' exact prediction parity',flush=True)
    run('scripts.reproduce_context_results','--exports','inputs/inference','--splits','data/splits','--scores','inputs/context-scores','--dev-candidates','inputs/official-dev/dev_candidates.json','--dev-gold','data/dev-v1.1.json','--output',root/'context')
    for name in ['gap_correction','gap_only','uncalibrated_span_mean']:
        assert read(root/'context'/(name+'-predictions.json'))==read(Path('results/context')/(name+'-predictions.json'))
    data=read('data/splits/internal_val.json')['data']
    baseline=read(root/'internal-val/bert-predictions.json')
    cls=read('originals/context/cls/epoch-5/internal_val_predictions.json')
    assert cls==baseline
    cls_metrics=evaluate(data,cls)
    span=read(root/'context/uncalibrated_span_mean-predictions.json')
    assert span==read('originals/context/span_mean/epoch-5/internal_val_predictions.json')
    calibrated=read(root/'context/gap_correction-predictions.json')
    base=evaluate(data,baseline); cal=evaluate(data,calibrated)
    sizes=[]; deltas=[]
    for article in data:
        ids=[q['id'] for p in article['paragraphs'] for q in p['qas']]
        sizes.append(len(ids)); deltas.append(sum(cal['f1_per_question'][i]-base['f1_per_question'][i] for i in ids))
    draws=np.random.default_rng(4248).integers(len(data),size=(20000,len(data)))
    ci=np.percentile(100*np.array(deltas)[draws].sum(1)/np.array(sizes)[draws].sum(1),[2.5,97.5]).tolist()
    run('scripts.span_length_ablation','--data','assets/data.zip','--inference','assets/inference.zip','--output',root/'span-length')
    for name in ['basic','without_length','with_list']:
        assert read(root/'span-length'/(name+'-predictions.json'))==read(Path('results/span-length')/(name+'-predictions.json'))
    run('scripts.candidate_oracle','--dataset','data/splits/internal_val.json','--candidates','inputs/inference/internal_val_candidates.json','--baseline','inputs/inference/internal_val_predictions.json','--output',root/'oracle.json')
    assert read(root/'oracle.json')['summary']==read('results/oracle.json')['summary']
    run('scripts.final_comparison','--data','assets/data.zip','--candidates','assets/dev_candidates.zip','--predictions-dir',root/'gated-top-two')
    assert read(root/'gated-top-two/frozen-predictions.json')==read('results/gated-top-two/frozen-predictions.json')
    checks={'all_passed':True,'all_main_prediction_maps_equal':True,'context_and_ablation_prediction_parity':True,'cls_saved_predictions_equal_bert':True,'cls_metrics':{k:cls_metrics[k] for k in ['exact','f1','total']},'calibration_article_bootstrap_95ci':ci,'bootstrap_seed':4248,'bootstrap_samples':20000,'commands':commands,'python':sys.version,'cwd':str(Path.cwd())}
    (root/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
