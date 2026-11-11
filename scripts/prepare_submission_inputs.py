"""Verify companion release archives and extract required inputs without overwrites."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def extract(archive, target, prefix=''):
    with ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir(): continue
            name=PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts or ':' in item.filename:
                raise ValueError('Unsafe archive path')
            destination=target/prefix/Path(*name.parts)
            raw=z.read(item)
            if destination.exists():
                if destination.read_bytes()!=raw: raise ValueError('Refusing to overwrite different input: '+str(destination))
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as f: f.write(raw)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',action='store_true',help='Also unpack assets/final.zip for BERT inference')
    args=p.parse_args(); manifest=json.loads(Path('provenance/assets.json').read_bytes())
    required=['data.zip','inference.zip','dev_candidates.zip']+(['final.zip'] if args.checkpoint else [])
    for name in required:
        path=Path('assets')/name
        assert sha(path)==manifest[name]['sha256'], 'Archive hash mismatch: '+name
    extract(Path('assets/data.zip'),Path('.'))
    extract(Path('assets/inference.zip'),Path('inputs'))
    extract(Path('assets/dev_candidates.zip'),Path('inputs/official-dev'))
    if args.checkpoint: extract(Path('assets/final.zip'),Path('checkpoints/bert-base-baseline'))
    print('Companion hashes verified; input files prepared. Existing identical inputs preserved.')

if __name__=='__main__': main()
