"""Run each suite in a fresh process, retaining logs and an aggregate receipt."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for suite in sorted(Path('tests').glob('test_*.py')):
        cmd=[sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-p',suite.name,'-v']
        run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (args.output/(suite.stem+'.log')).write_text(run.stdout,encoding='utf-8')
        count=re.search(r'Ran (\d+) tests?',run.stdout)
        skipped=re.search(r'OK \(skipped=(\d+)\)',run.stdout)
        results.append({'suite':suite.name,'exit_code':run.returncode,'tests':int(count.group(1)) if count else 0,'skipped':int(skipped.group(1)) if skipped else 0})
        print(suite.name,run.returncode,flush=True)
    report={'suites':results,'all_passed':all(r['exit_code']==0 for r in results),'total_tests':sum(r['tests'] for r in results),'skipped':sum(r['skipped'] for r in results),'python':sys.version}
    (args.output/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['all_passed']: raise SystemExit(1)

if __name__=='__main__': main()
