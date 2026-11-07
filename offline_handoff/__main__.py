"""Generate or validate a synthetic offline bundle. No model-related commands."""
import argparse
from pathlib import Path

from .contract import FILES, ContractError, encode, require, validate_bundle
from .fixtures import synthetic_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('fixture')
    create.add_argument('directory', type=Path)
    create.add_argument('--group', choices=('basic', 'full'), default='basic')
    check = commands.add_parser('validate')
    check.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'fixture':
            manifest, artifacts = synthetic_bundle(args.group)
            report = validate_bundle(manifest, artifacts)
            args.directory.mkdir(parents=False, exist_ok=False)
            for name, raw in {'manifest.json': manifest, **artifacts}.items():
                with (args.directory / name).open('xb') as stream:
                    stream.write(raw)
        else:
            require(not args.directory.is_symlink(), 'bundle directory must not be a symlink')
            paths = {p.name: p for p in args.directory.iterdir()}
            require(set(paths) == FILES | {'manifest.json'}, 'exact bundle file set required')
            require(all(p.is_file() and not p.is_symlink() for p in paths.values()),
                    'bundle artifacts must be regular files, not symlinks')
            manifest = paths['manifest.json'].read_bytes()
            artifacts = {name: paths[name].read_bytes() for name in FILES}
            report = validate_bundle(manifest, artifacts)
        print(encode(report).decode('utf-8'), end='')
    except (ContractError, OSError) as exc:
        parser.exit(2, f'offline contract rejected: {exc}\n')


if __name__ == '__main__':
    main()
