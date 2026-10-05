#!/usr/bin/env python3
"""Build a pinned upstream GGFT checkout and generate new Poisson/Gradient data."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

UPSTREAM_REVISION = '9cb7fc69ed5fb54ce03be6fa8556402e10451be6'
GRID_SIZE = 23
STRUCTURE_VALUES = 16 * 7
RECORD_VALUES = GRID_SIZE ** 3 + STRUCTURE_VALUES + 6 * GRID_SIZE ** 2


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Use a positive integer')
    return result


def check_format(path, samples):
    """Check header and complete record count, without reading GB-sized data into RAM."""
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(16)
    if len(header) != 16 or struct.unpack('<dd', header) != (23.0, 1.0):
        raise ValueError(f'{path.name}: expected float64 header N=23, block_width=1')
    expected = 16 + samples * RECORD_VALUES * 8
    if path.stat().st_size != expected:
        raise ValueError(f'{path.name}: expected {expected} bytes for {samples} complete records')
    return {'samples': samples, 'bytes': expected, 'grid_size': GRID_SIZE,
            'record_values': RECORD_VALUES, 'dtype': 'little-endian float64'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True,
                        help='DeepRWCap Git checkout at the documented GGFT revision')
    parser.add_argument('--kind', choices=['poisson', 'gradient', 'both'], default='both')
    parser.add_argument('--samples', type=positive_int, required=True, help='Samples per dataset')
    parser.add_argument('--threads', type=positive_int, default=12)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if sys.platform != 'linux' or sys.byteorder != 'little':
        parser.error('GGFT generation is supported on little-endian Linux')
    if args.threads > args.samples:
        parser.error('--threads must not exceed --samples')
    for name in ('git', 'make', 'g++'):
        if shutil.which(name) is None:
            parser.error(f'{name} is required')
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    if revision != UPSTREAM_REVISION:
        parser.error(f'Use upstream revision {UPSTREAM_REVISION}; found {revision}')
    changed = subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--', 'ggft'])
    if changed.returncode != 0:
        parser.error('The pinned GGFT source has local changes')
    ggft = upstream / 'ggft'
    if not (ggft / 'makefile').is_file():
        parser.error('The upstream checkout must contain ggft/makefile and its Eigen dependency')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    kinds = ['poisson', 'gradient'] if args.kind == 'both' else [args.kind]
    record = {'state': 'running', 'upstream_revision': revision, 'samples_per_dataset': args.samples,
              'threads': args.threads, 'kinds': kinds, 'files': {},
              'random_generation': 'Upstream mt19937 initialized with each OpenMP thread index.',
              'scope': 'New GGFT data; not an assertion of identity with the paper reference datasets.'}

    def save():
        (output / 'generation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

    save()
    try:
        print('Building pinned GGFT', flush=True)
        with (output / 'build.log').open('w') as log:
            # Rebuild even when an older executable exists in the checkout.
            subprocess.run(['make', '-B', '-C', str(ggft), '-j', str(args.threads)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for kind in kinds:
            print(f'Generating {args.samples} {kind} samples with {args.threads} threads', flush=True)
            command = [str(ggft / 'bin/ggft'), str(args.threads), str(args.samples), str(output), kind]
            with (output / f'{kind}.log').open('w') as log:
                subprocess.run(command, cwd=ggft, stdout=log, stderr=subprocess.STDOUT, check=True)
            record['files'][kind + '.bin'] = check_format(output / (kind + '.bin'), args.samples)
            save()
        record['state'] = 'completed'
    except Exception as exc:
        record.update(state='failed', error=str(exc))
        raise
    finally:
        save()
    print(f'Generated datasets: {output}', flush=True)


if __name__ == '__main__':
    main()
