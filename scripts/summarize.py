#!/usr/bin/env python3
"""Summarize all planned observations; never present an incomplete batch as a full result."""
import argparse
import json
import math
from pathlib import Path
import statistics
from common import ROOT, dump
from naming import ARM_LABELS, comparison_label
from input_checks import validate_plan
from statistics_report import attribution, workload
from evaluate_readout import evaluate


def describe(values):
    return {'n':len(values),'mean':statistics.mean(values),
            'repeat_sd':statistics.stdev(values) if len(values)>1 else None,
            'min':min(values),'max':max(values)}


def summarize(directory):
    directory=Path(directory)
    plan=json.loads((directory/'plan.json').read_text())
    validate_plan(plan)
    status_path=directory/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    valid=[]
    missing=[]
    invalid=[]
    for item in plan['runs']:
        path=directory/item['directory']/'metrics.json'
        if not path.exists():
            missing.append(item['directory'])
            continue
        try:
            row=json.loads(path.read_text())
            for k,expected in [('case',item['case']),('arm',item['arm']),('initial_seed',item['seed']),
                               ('master',item['config']['master']),('reference_F',item['config']['reference_F'])]:
                if row.get(k)!=expected:raise ValueError(f'{k} disagrees with plan')
            for k in ['raw_self_error_percent','s24_self_error_percent']:
                if not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<0:
                    raise ValueError('Invalid error metric: '+k)
            # Require solver configuration, not just an unlabeled numeric result.
            run=json.loads((path.parent/'run.json').read_text())
            if run['arm']!=item['arm'] or run['configuration']!=item['config']:
                raise ValueError('Run configuration disagrees with plan')
            # A stale/edited metrics file must not silently become a paper result.
            cfg=item['config']
            parsed=evaluate(ROOT/cfg['geometry'],path.parent/'result.out',cfg.get('reference_F'),cfg['master'])
            for k in ['raw_capacitance_F','s24_capacitance_F','raw_self_error_percent','s24_self_error_percent',
                      'walks','hops_per_walk','total_steps_approx','elapsed_seconds','cpu_seconds']:
                if k not in row or not math.isclose(row[k],parsed[k],rel_tol=1e-10,abs_tol=0):
                    raise ValueError('Metric disagrees with original solver output: '+k)
            if row.get('matrix_F')!=parsed['matrix_F'] or row.get('s24_action')!=parsed['s24_action']:
                raise ValueError('Matrix/readout disagrees with solver output')
            valid.append(row)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            invalid.append({'directory':item['directory'],'reason':str(exc)})
    warmup_missing=[item['directory'] for item in plan.get('warmups',[])
                    if not (directory/item['directory']/'metrics.json').exists()]
    complete=(len(valid)==len(plan['runs']) and not missing and not invalid
              and not warmup_missing and status.get('state')=='completed')
    per_case=[]
    for case in plan['cases']:
        for arm in plan['arms']:
            rows=[r for r in valid if r['case']==case and r['arm']==arm]
            if not rows:continue
            per_case.append({'case':case,'arm':arm,'planned_n':len(plan['seeds']),
                'raw_error_percent':describe([r['raw_self_error_percent'] for r in rows]),
                's24_error_percent':describe([r['s24_self_error_percent'] for r in rows])})
    macro={}
    differences={}
    module_comparisons={}
    resources={}
    if complete:
        module_comparisons=attribution(valid,plan)
        resources={arm:workload([r for r in valid if r['arm']==arm]) for arm in plan['arms']}
        for arm in plan['arms']:
            rows=[r for r in per_case if r['arm']==arm]
            raw=[r['raw_error_percent']['mean'] for r in rows]
            s24=[r['s24_error_percent']['mean'] for r in rows]
            endpoint=raw if plan['readout'][arm]=='raw' else s24
            macro[arm]={'readout':plan['readout'][arm], 'case_count':len(rows),
                'raw_macro_error_percent':statistics.mean(raw),'s24_macro_error_percent':statistics.mean(s24),
                'endpoint_macro_error_percent':statistics.mean(endpoint),
                'endpoint_between_case_sd':statistics.stdev(endpoint) if len(endpoint)>1 else None}
        if 'p0' in macro:
            baseline=macro['p0']['endpoint_macro_error_percent']
            for arm in plan['arms']:
                if arm=='p0':continue
                block_deltas=[]
                for seed in plan['seeds']:
                    a=[r for r in valid if r['initial_seed']==seed and r['arm']==arm]
                    b=[r for r in valid if r['initial_seed']==seed and r['arm']=='p0']
                    key='raw_self_error_percent' if plan['readout'][arm]=='raw' else 's24_self_error_percent'
                    block_deltas.append(statistics.mean(r[key] for r in a)-statistics.mean(r['raw_self_error_percent'] for r in b))
                delta=macro[arm]['endpoint_macro_error_percent']-baseline
                differences[arm]={'vs':'p0 raw','mean_difference_pp':delta,
                    'raw_difference_vs_p0_raw_pp':macro[arm]['raw_macro_error_percent']-macro['p0']['raw_macro_error_percent'],
                    's24_difference_vs_p0_s24_pp':macro[arm]['s24_macro_error_percent']-macro['p0']['s24_macro_error_percent'],
                    'relative_change_percent':100*delta/baseline if baseline else None,
                    'seed_block_difference_pp':describe(block_deltas),
                    'note':'Negative favors this arm. Blocks share initial seeds, not guaranteed common trajectories; no significance claim.'}
    reference_path=Path(__file__).resolve().parents[1]/'results/reference_public10.json'
    reference_comparison={}
    if complete and plan['cases']==[f'case{i}' for i in range(1,11)] and reference_path.exists():
        reference=json.loads(reference_path.read_text())
        for arm,entry in macro.items():
            if arm in reference['macro']:
                target=reference['macro'][arm]['endpoint_macro_error_percent']
                reference_comparison[arm]={'historical_reference_percent':target,
                    'new_minus_historical_pp':entry['endpoint_macro_error_percent']-target}
    return {'complete':complete,'profile':plan['profile'],'planned':len(plan['runs']),'valid':len(valid),
        'missing':missing,'invalid':invalid,'warmup_missing':warmup_missing,'batch_state':status.get('state','unknown'),
        'failed':status.get('failed'),'per_case':per_case,'macro':macro,'differences':differences,
        'historical_comparison':reference_comparison,
        'module_comparisons':module_comparisons,'resources':resources,
        'scope':'Fixed-benchmark, pointwise seed-block intervals; no multiplicity correction or accuracy pass/fail target. Shared seeds do not imply common trajectories.'}


def save_summary(directory,report):
    directory=Path(directory)
    dump(directory/'summary.json',report)
    lines=['# Benchmark summary','',f"Complete: {report['complete']}; valid/planned: {report['valid']}/{report['planned']}.",'']
    if report['complete']:
        lines += ['| Arm | Raw mean error (%) | CER mean error (%) | Selected endpoint (%) |',
                  '|---|---:|---:|---:|']
        for arm,row in report['macro'].items():
            lines.append(f"| {ARM_LABELS[arm]} | {row['raw_macro_error_percent']:.6f} | {row['s24_macro_error_percent']:.6f} | {row['endpoint_macro_error_percent']:.6f} |")
        lines += ['', '| Same-batch comparison | Difference (pp) | 95% seed-block interval |', '|---|---:|---|']
        for name,row in report['module_comparisons'].items():
            ci=row['ci95_pp'];label='Not estimated (one seed)' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
            lines.append(f"| {comparison_label(name)} | {row['mean_difference_pp']:+.6f} | {label} |")
        lines += ['', '| Arm | Elapsed (s/case) | CPU (s/case) | Total walks | Weighted hops/walk |', '|---|---:|---:|---:|---:|']
        for arm,r in report['resources'].items():
            if r:lines.append(f"| {ARM_LABELS[arm]} | {r['mean_elapsed_seconds_per_case']:.3f} | {r['mean_cpu_seconds_per_case']:.3f} | {r['mean_total_walks_across_cases']:.1f} | {r['walk_weighted_hops_per_walk']:.5f} |")
        lines += ['', 'Timings describe this batch. GPU/host memory and CER micro-latency are not measured here.']
    else:
        lines += ['Incomplete batch: aggregate performance comparisons are withheld. Inspect coverage and errors in summary.json.']
    lines += ['',report['scope'],'']
    (directory/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    a=ap.parse_args()
    result=summarize(a.input)
    save_summary(a.input,result)
    print(f"Complete={result['complete']}; valid/planned={result['valid']}/{result['planned']}")
    raise SystemExit(0 if result['complete'] else 2)
