#!/usr/bin/env python3
"""Evaluate the bundled upstream CPU baselines; Python standard library only."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

from common import ROOT, dump
from readout.parser import geometry, parse_output

METHODS = {'agf': 'FRW-AGF', 'microwalk': 'MicroWalk', 'fdm': 'FRW-FDM'}


def make_plan(method, profile='quick', workers=16):
    if method not in METHODS or profile not in ('quick', 'paper'):
        raise ValueError('Unsupported method or profile')
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    protocol = json.loads((ROOT/'configs/paper_protocol.json').read_text())
    cases = protocol['cases'] if profile == 'paper' else [c for c in protocol['cases'] if c['case'] == 'case7']
    seeds = protocol['solver_initial_seeds'] if profile == 'paper' else [2029]
    runs = []
    for seed in seeds:
        for case in cases:
            cfg = dict(case, seed=seed, workers=workers, p=protocol['p'], c=protocol['c'])
            runs.append({'directory': f"{case['case']}/seed{seed}", 'config': cfg})
    return {'method': method, 'name': METHODS[method], 'profile': profile,
            'binary': f'third_party/deeprwcap/baselines/rwcap_{method}',
            'workers': workers, 'seeds': seeds, 'cases': [c['case'] for c in cases], 'runs': runs,
            'endpoint': 'raw selected-master SelfCapErr; CER not applied',
            'seed_policy': 'Request upstream --seed; no guarantee of deterministic parallel trajectories.',
            'failure_policy': 'Stop and preserve outputs; no automatic retry or exclusion.'}


def command(plan, item, folder):
    cfg = item['config']
    return [str(ROOT/plan['binary']), '-f', str(ROOT/cfg['geometry']), '-n', str(cfg['workers']),
            '-p', str(cfg['p']), '-c', str(cfg['c']), '--c-ratio', str(cfg['c_ratio']),
            '--seed', str(cfg['seed']), '--out-file', str(folder/'result.out'),
            '--log-file', str(folder/'result.log')]


def baseline_environment():
    env = {k: v for k, v in os.environ.items()
           if k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH') and not k.startswith(('S29_', 'PAPER_SOLVER_'))}
    env.update(LC_ALL='C', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def parse_run(folder, cfg):
    master, _ = geometry(ROOT/cfg['geometry'])
    if master != cfg['master']:
        raise ValueError('Configured master does not match geometry')
    matrix, walks, hops, elapsed, cpu = parse_output(folder/'result.out')
    if set(matrix) != {master}:
        raise ValueError('Output master set does not match requested task')
    cap = matrix[master].get(master)
    reference = cfg['reference_F']
    if cap is None or not math.isfinite(cap) or cap <= 0 or not math.isfinite(reference) or reference <= 0:
        raise ValueError('Missing or invalid self capacitance/reference')
    if any(not math.isfinite(x) for row in matrix.values() for x in row.values()):
        raise ValueError('Nonfinite matrix')
    if walks <= 0 or hops <= 0 or elapsed < 0 or cpu < 0:
        raise ValueError('Invalid workload/timing')
    return {'case': cfg['case'], 'initial_seed_requested': cfg['seed'], 'workers': cfg['workers'],
            'master': master, 'reference_F': reference, 'raw_capacitance_F': cap,
            'raw_self_error_percent': 100*abs(cap/reference-1), 'walks': walks,
            'hops_per_walk': hops, 'total_steps_approx': round(walks*hops),
            'elapsed_seconds': elapsed, 'cpu_seconds': cpu}


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, plan):
    status = json.loads((output/'status.json').read_text())
    if status['state'] != 'completed':
        raise ValueError('Incomplete batch: no complete aggregate')
    rows = []
    for item in plan['runs']:
        folder = output/item['directory']
        recorded = json.loads((folder/'run.json').read_text())
        if recorded['configuration'] != item['config'] or recorded['method'] != plan['method']:
            raise ValueError('Recorded configuration disagrees with plan')
        # Derive the published metrics from solver outputs, not cached metrics.json.
        rows.append(parse_run(folder, item['config']))
    per_case = []
    for case in plan['cases']:
        group = [r for r in rows if r['case'] == case]
        values = [r['raw_self_error_percent'] for r in group]
        per_case.append({'case': case, 'repetitions': len(group),
                         'raw_mean_error_percent': statistics.mean(values),
                         'raw_repeat_sd': statistics.stdev(values) if len(values) > 1 else None,
                         'mean_elapsed_seconds': statistics.mean(r['elapsed_seconds'] for r in group),
                         'mean_cpu_seconds': statistics.mean(r['cpu_seconds'] for r in group)})
    means = [r['raw_mean_error_percent'] for r in per_case]
    report = {'complete': True, 'method': plan['name'], 'measured_solves': len(rows),
              'workers': plan['workers'], 'endpoint': plan['endpoint'],
              'raw_macro_self_error_percent': statistics.mean(means),
              'between_case_sd': statistics.stdev(means) if len(means) > 1 else None,
              'mean_elapsed_seconds_per_case': statistics.mean(r['elapsed_seconds'] for r in rows),
              'mean_cpu_seconds_per_case': statistics.mean(r['cpu_seconds'] for r in rows),
              'mean_total_walks_across_cases': sum(r['walks'] for r in rows)/len(plan['seeds']),
              'walk_weighted_hops_per_walk': sum(r['walks']*r['hops_per_walk'] for r in rows)/sum(r['walks'] for r in rows),
              'mean_total_steps_approx_across_cases': sum(r['total_steps_approx'] for r in rows)/len(plan['seeds']),
              'scope': 'This batch only. Raw readout; no CER, peak-memory measurement or automatic speedup claim.'}
    table(output/'per_run.csv', rows)
    table(output/'per_case.csv', per_case)
    dump(output/'summary.json', report)
    lines = [f"# {plan['name']} baseline", '', f"Completed solves: {len(rows)}; solver threads: {plan['workers']}.", '',
             '| Case | Repetitions | Raw SelfCapErr (%) | Repeat SD |', '|---|---:|---:|---:|']
    for r in per_case:
        sd = 'N/A' if r['raw_repeat_sd'] is None else f"{r['raw_repeat_sd']:.6f}"
        lines.append(f"| {r['case']} | {r['repetitions']} | {r['raw_mean_error_percent']:.6f} | {sd} |")
    lines += ['', f"Macro SelfCapErr: {report['raw_macro_self_error_percent']:.6f}%.", '', report['scope'], '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def execute(plan, output, runner=subprocess.run):
    binary = ROOT/plan['binary']
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not os.access(binary, os.X_OK):
        raise PermissionError(f'Run chmod +x {binary}')
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'plan.json', plan)
    status = {'state': 'running', 'completed': 0, 'planned': len(plan['runs']), 'failed': None}
    dump(output/'status.json', status)
    try:
        for item in plan['runs']:
            status['active'] = item['directory']
            dump(output/'status.json', status)
            folder = output/item['directory']
            folder.mkdir(parents=True)
            cmd = command(plan, item, folder)
            dump(folder/'run.json', {'method': plan['method'], 'configuration': item['config'], 'command': cmd,
                                    'seed_control': plan['seed_policy'], 'environment': {
                                        'LD_PRELOAD': None, 'CUDA_VISIBLE_DEVICES': '',
                                        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}})
            print(f"{status['completed']+1}/{status['planned']} {plan['name']} {item['directory']}", flush=True)
            with (folder/'console.log').open('w') as stream:
                runner(cmd, cwd=folder, env=baseline_environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
            dump(folder/'metrics.json', parse_run(folder, item['config']))
            status['completed'] += 1
        status.update(state='completed', active=None)
    except BaseException as exc:
        status.update(state='failed', failed={'directory': status.get('active'),
                      'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        dump(output/'status.json', status)
    return summarize(output, plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--profile', choices=['quick', 'paper'], default='quick')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be positive')
    output = args.output.resolve()
    if output.exists():
        ap.error('Choose a new output directory')
    plan = make_plan(args.method, args.profile, args.workers)
    if args.plan_only:
        output.mkdir(parents=True, exist_ok=False)
        dump(output/'plan.json', plan)
        print(f"Planned {len(plan['runs'])} {plan['name']} solves; no execution")
        return
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        ap.error('The bundled baseline binaries require Linux x86_64')
    try:
        report = execute(plan, output)
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'Baseline stopped: {exc}; any partial output is retained')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
