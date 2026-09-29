#!/usr/bin/env python3
"""Evaluate frozen local transitions using existing external reference datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from common import ROOT,dump

# Dataset identity prevents newly generated labels being mistaken for the frozen split.
# No standalone checksum files or private server paths are needed.
DATA_IDENTITIES={
    'poisson':'2fb292e876ed69815bb27a584f25f38c3ec5ebad635aeb92f47cd6a9322cd778',
    'gradient':'2d24bb2514dd2931fd89f68a4a2284307945485be9cd6cfa64fddc0677f8c9f9'}


def preflight(data_dir,modules,config):
    required=['poisson','gradient'] if modules=='all' else ['poisson' if modules=='bpr' else 'gradient']
    info={}
    for name in required:
        p=data_dir/(name+'.bin')
        if not p.is_file():
            raise FileNotFoundError(f'{p} is required. Reference datasets are not bundled; see docs/LOCAL_VALIDATION.md.')
        if p.stat().st_size!=config['data_bytes'][name]:raise ValueError('Dataset size mismatch: '+name)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        if h.hexdigest()!=DATA_IDENTITIES[name]:raise ValueError('Not the frozen reference dataset: '+name)
        info[name]={'file':str(p),'bytes':p.stat().st_size,'frozen_identity_verified':True}
    ids=config['poisson_validation_indices']
    if len(ids)!=10000 or len(set(ids))!=10000 or not all(type(i)is int and 0<=i<100000 for i in ids):
        raise ValueError('Invalid Poisson validation split')
    return info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True,help='Directory containing the original poisson.bin/gradient.bin')
    ap.add_argument('--module',choices=['bpr','cpgr','all'],default='all')
    ap.add_argument('--output',type=Path,required=True,help='New output directory')
    ap.add_argument('--preflight-only',action='store_true',help='Verify datasets and split without importing GPU libraries')
    args=ap.parse_args()
    data=args.data_dir.resolve();out=args.output.resolve()
    if out.exists():ap.error('Output directory must be new')
    cfg=json.loads((ROOT/'configs/transition_validation.json').read_text())
    datasets=preflight(data,args.module,cfg)
    out.mkdir(parents=True,exist_ok=False)
    meta={'module':args.module,'datasets':datasets,'configuration':cfg,'training':False,'e2e':False,'label_generation':False}
    dump(out/'protocol.json',meta)
    if args.preflight_only:
        dump(out/'status.json',{'state':'preflight_complete','gpu_executed':False});return
    started=time.time();dump(out/'status.json',{'state':'running'})
    try:
        import transition_core as ev
        ev.DATA=data
        os.chdir(out)
        ev.torch.set_num_threads(4);ev.torch.set_grad_enabled(False)
        meta['environment']={'platform':platform.platform(),'torch':ev.torch.__version__,
            'torch_tensorrt':ev.torch_tensorrt.__version__,'tensorrt':ev.tensorrt.__version__,
            'gpu':ev.torch.cuda.get_device_name()}
        dump(out/'protocol.json',meta)
        result={}
        if args.module in ['bpr','all']:
            result['bpr']=ev.evaluate_bpr(ev.np.array(cfg['poisson_validation_indices']),out,cfg['batch_size'])
        if args.module in ['cpgr','all']:
            ids=ev.np.arange(cfg['gradient_start'],cfg['gradient_end'])
            result['cpgr'],result['weight'],result['checks']=ev.evaluate_gradient(ids,out,cfg['batch_size'])
        dump(out/'summary.json',result)
        lines=['# Frozen local transition evaluation','','Precision: '+cfg['precision']+'. No end-to-end solve.','']
        if 'bpr' in result:
            b=result['bpr'];lines+=['| BPR metric | P0 | BPR |','|---|---:|---:|']
            for k in ['kl','action_error']:lines.append(f"| Mean {k} | {b['raw_'+k]['mean']:.10g} | {b['bpr_'+k]['mean']:.10g} |")
            lines+=['',f"Mean TV(BPR,P0): {b['bpr_vs_p0_tv']['mean']:.10g}; this measures refinement magnitude.",'']
        if 'cpgr' in result:
            lines+=['| Head | Active / total | Raw NL2 | CPGR NL2 | Relative change (%) | Win fraction |','|---|---:|---:|---:|---:|---:|']
            for name,b in result['cpgr'].items():
                lines.append(f"| {name} | {b['activated_samples']}/{b['validation_samples']} | {b['raw_l2']['mean']:.10g} | {b['cpgr_l2']['mean']:.10g} | {b['l2_relative_change_percent']:.5f} | {b['l2_win_rate']:.5f} |")
            b=result['weight']['active'];lines+=['',f"Active six-face effective-mass NL1: {b['raw_error']['mean']:.10g} -> {b['cpgr_error']['mean']:.10g} ({b['samples']} configurations).",'']
        lines+=['Full parity, activation, tail statistics and numerical checks are in summary.json; all paired samples are retained in CSV.gz. Local improvements are not an E2E efficacy test.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        dump(out/'status.json',{'state':'completed','elapsed_seconds':time.time()-started})
    except BaseException as exc:
        dump(out/'status.json',{'state':'failed','error':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.time()-started})
        raise


if __name__=='__main__':main()
