#!/usr/bin/env python3
"""Run a predetermined public benchmark; default is a small functional check."""
import argparse
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
from common import ROOT, dump, check_files
from input_checks import validate_plan, validate_config
from naming import normalize_arm, CLI_ARMS, ARM_LABELS

ARMS=('p0','bpr','full')


def make_plan(protocol, profile='quick', arms=None):
    selected=[normalize_arm(a) for a in (arms or ['p0','bpr','full'])]
    if not selected or len(selected)!=len(set(selected)) or any(a not in ARMS for a in selected):
        raise ValueError('Specify distinct supported arms')
    if profile not in ('quick','paper'):
        raise ValueError('Unknown profile')
    cases=protocol['cases'] if profile=='paper' else [r for r in protocol['cases'] if r['case']=='case1']
    if not cases:
        raise ValueError('Protocol must include case1 for the quick profile')
    seeds=protocol['solver_initial_seeds'] if profile=='paper' else [2029,2030,2031]
    permutations=list(itertools.permutations(selected))
    rng=random.Random(protocol['order_seed'])
    runs=[]
    for repeat,seed in enumerate(seeds):
        indices=list(range(len(cases)))
        rng.shuffle(indices)
        for i in indices:
            case=cases[i]
            cfg={k:protocol[k] for k in ['workers','p','c','omp_threads','mkl_threads']}
            cfg.update(case)
            cfg['seed']=seed
            validate_config(cfg)
            for arm in permutations[(repeat+i)%len(permutations)]:
                runs.append({'case':case['case'],'arm':arm,'seed':seed,'config':cfg.copy(),
                    'directory':f"{case['case']}/seed{seed}/{arm}"})
    warmups=[]
    if profile=='paper':
        case=next(r for r in protocol['cases'] if r['case']=='case8')
        cfg={k:protocol[k] for k in ['workers','p','c','omp_threads','mkl_threads']}
        cfg.update(case)
        cfg['seed']=2040
        warmups=[{'case':'case8','arm':a,'seed':2040,'config':cfg.copy(),
                  'directory':f'warmup/{a}'} for a in selected]
    plan={'schema_version':2,'profile':profile,'arms':selected,
            'cases':[r['case'] for r in cases],'seeds':seeds,
            'readout':{a:'raw' if a=='p0' else 'strict-S24' for a in selected},
            'order_seed':protocol['order_seed'],'warmups':warmups,'runs':runs,
            'bootstrap':{'draws':20000,'seed':20260919,'unit':'initial-seed blocks across fixed cases'},
            'statistics_scope':'Fixed public benchmark; previously used during method development, not an unseen test.',
            'failure_policy':'Stop at first failed solve; preserve outputs; no automatic retry or result-based exclusion.'}
    validate_plan(plan)
    return plan


def execute(plan, output, runner=None):
    runner=Path(runner or ROOT/'scripts/run.py')
    output=Path(output).resolve()
    validate_plan(plan)
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'plan.json',plan)
    configs=output/'configs'
    configs.mkdir()
    status={'state':'running','completed':0,'failed':None}
    dump(output/'status.json',status)
    items=[(True,x) for x in plan['warmups']]+[(False,x) for x in plan['runs']]
    try:
        for index,(warmup,item) in enumerate(items):
            cfg=configs/f'{index:04d}.json'
            dump(cfg,item['config'])
            dest=output/item['directory']
            log=output/f'launch_{index:04d}.log'
            status['active']=item['directory']
            dump(output/'status.json',status)
            print(f"{index+1}/{len(items)} {item['case']}/seed{item['seed']} {ARM_LABELS[item['arm']]}",flush=True)
            try:
                with log.open('w') as stream:
                    subprocess.run([sys.executable,str(runner),'--config',str(cfg),'--arm',item['arm'],
                                    '--output',str(dest)],stdout=stream,stderr=subprocess.STDOUT,check=True)
            except BaseException as exc:
                status.update(state='failed',failed={'directory':item['directory'],
                    'warmup':warmup,'error':type(exc).__name__,'message':str(exc)})
                raise
            if not warmup:
                status['completed']+=1
            dump(output/'status.json',status)
        status.update(state='completed',active=None)
    finally:
        dump(output/'status.json',status)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=['quick','paper'],default='quick')
    ap.add_argument('--arms',nargs='+',type=normalize_arm,choices=ARMS,metavar=CLI_ARMS)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plan-only',action='store_true',help='Write the exact order without loading Torch or starting solves')
    args=ap.parse_args()
    if args.output.exists():
        ap.error('Output directory already exists; choose a new directory.')
    protocol=json.loads((ROOT/'configs/paper_protocol.json').read_text())
    plan=make_plan(protocol,args.profile,args.arms)
    if args.plan_only:
        args.output.mkdir(parents=True,exist_ok=False)
        dump(args.output/'plan.json',plan)
        print(f"Planned {len(plan['runs'])} measured solves and {len(plan['warmups'])} warmups")
        return
    check_files()
    error=None
    try:
        execute(plan,args.output)
    except (Exception, KeyboardInterrupt) as exc:
        error=exc
    from summarize import summarize, save_summary
    if (args.output/'plan.json').exists():
        report=summarize(args.output)
        save_summary(args.output,report)
    if error:
        raise SystemExit(f'Batch stopped: {error}. Partial outputs and coverage report were retained.')


if __name__=='__main__':main()
