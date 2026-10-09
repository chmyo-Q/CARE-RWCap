#!/usr/bin/env python3
"""Recompute the archived CAPR training-seed supplement, without a solver/GPU."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Capacitance/reference must be finite and positive')
    return value


def close(a, b, context):
    if not math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=0):
        raise ValueError('Inconsistent ' + context)


def error(cap, ref):
    return 100 * abs(positive(cap) / positive(ref) - 1)


def summarize(data, protocol):
    data = Path(data)
    cases = {c['case']: c for c in protocol['cases']}
    seeds = protocol['solver_initial_seeds']
    study = json.loads((data / 'protocol.json').read_text(encoding='utf-8'))
    training_seeds = study['training_seeds']
    expected = {(c, s) for c in cases for s in seeds}
    baseline = {}
    for r in read_csv(data / 'baseline_runs.csv'):
        k = r['case'], int(r['seed'])
        if k not in expected or k in baseline:
            raise ValueError('Unexpected/duplicate baseline case-seed')
        if int(r['repeat']) != seeds.index(k[1]) + 1:
            raise ValueError('Baseline repeat/seed mismatch')
        close(r['reference_F'], cases[k[0]]['reference_F'], 'baseline reference')
        e = error(r['raw_F'], r['reference_F'])
        close(r['raw_err_percent'], e, 'baseline error')
        baseline[k] = r, e
    if set(baseline) != expected:
        raise ValueError('Incomplete baseline grid')
    measured = {}
    for r in read_csv(data / 'repeat_results.csv'):
        t = int(r['training_seed'])
        k = r['case'], int(r['frw_seed'])
        full_key = (t,) + k
        if t not in training_seeds or k not in expected or full_key in measured:
            raise ValueError('Unexpected/duplicate Full case-seed')
        if int(r['repeat']) != seeds.index(k[1]) + 1:
            raise ValueError('Full repeat/seed mismatch')
        b, be = baseline[k]
        close(r['reference_F'], b['reference_F'], 'Full reference')
        close(r['baseline_raw_F'], b['raw_F'], 'paired baseline capacitance')
        close(r['baseline_err_percent'], be, 'paired baseline error')
        close(r['raw_err_percent'], error(r['raw_F'], r['reference_F']), 'raw error')
        e = error(r['cer_F'], r['reference_F'])
        close(r['selfcaperr_percent'], e, 'CER error')
        close(r['paired_delta_pp'], e - be, 'paired delta')
        measured[full_key] = e
    if set(measured) != {(t,) + k for t in training_seeds for k in expected}:
        raise ValueError('Incomplete Full grid')
    base_case = {c: statistics.mean(baseline[c, s][1] for s in seeds) for c in cases}
    base_macro = statistics.mean(base_case.values())
    case_rows, seed_rows = [], []
    for t in training_seeds:
        group = []
        for c in cases:
            values = [measured[t, c, s] for s in seeds]
            mean = statistics.mean(values)
            row = dict(training_seed=t, case=c, mean_selfcaperr_percent=mean,
                       repeat_sample_sd_percent=statistics.stdev(values),
                       baseline_mean_percent=base_case[c], delta_pp=mean-base_case[c])
            case_rows.append(row)
            group.append(mean)
        macro = statistics.mean(group)
        seed_rows.append(dict(training_seed=t, macro_selfcaperr_percent=macro,
            baseline_macro_percent=base_macro, delta_pp=macro-base_macro,
            relative_change_percent=100*(macro/base_macro-1),
            wins=sum(a < base_case[c] for a, c in zip(group, cases)), cases=len(cases),
            measured_runs=len(expected)))
    macros = [r['macro_selfcaperr_percent'] for r in seed_rows]
    aggregate = dict(training_seed_count=len(macros),
        mean_macro_selfcaperr_percent=statistics.mean(macros),
        sample_sd_macro_selfcaperr_percent=statistics.stdev(macros), ddof=1,
        baseline_macro_percent=base_macro, baseline_reused=True,
        comparison='Historical protocol-matched DeepRWCap raw; not contemporaneous/common-path paired',
        uncertainty='Across three model-specific macro means; includes FRW evaluation noise')
    return case_rows, seed_rows, aggregate


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/training_seeds')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'configs/paper_protocol.json').read_text(encoding='utf-8'))
    cases, seeds, aggregate = summarize(args.input, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'case_summary.csv', cases)
    write_csv(args.output/'seed_summary.csv', seeds)
    (args.output/'aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n', encoding='utf-8')
    lines = ['# CAPR training-seed supplement', '',
        '| Training seed | Macro SelfCapErr (%) | Delta (pp) | Relative change (%) | Wins |',
        '|---|---:|---:|---:|---:|']
    for r in seeds:
        lines.append(f"| {r['training_seed']} | {r['macro_selfcaperr_percent']:.7f} | {r['delta_pp']:+.7f} | {r['relative_change_percent']:+.4f} | {r['wins']}/{r['cases']} |")
    lines += ['', f"Mean +/- sample SD (n=3): {aggregate['mean_macro_selfcaperr_percent']:.7f} +/- {aggregate['sample_sd_macro_selfcaperr_percent']:.7f}%.",
        '', aggregate['comparison']+'. '+aggregate['uncertainty']+'.',
        '', 'This command reaggregates released capacitance records; it does not rerun inference or reparse original solver logs.']
    (args.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
