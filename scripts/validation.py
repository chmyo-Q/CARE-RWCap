"""Reject incomplete or ambiguous experiment specifications before aggregation."""
import math
from pathlib import PurePosixPath


def validate_cpgr_counts(projection,selector):
    if projection['cuda_status']!=0 or selector['cuda_status']!=0:
        raise ValueError('CPGR CUDA error; inspect preserved counters')
    if projection['samples']!=sum(projection['masks']) or projection['samples']!=sum(selector['input_xyz_masks']):
        raise ValueError('CPGR selector/projection sample counts disagree')


def validate_config(cfg):
    for k in ['seed','workers','omp_threads','mkl_threads']:
        v=cfg[k]
        if type(v) is not int or v<=0:
            raise ValueError(k+' must be a positive integer')
    if cfg['seed']>2147483647:
        raise ValueError('seed exceeds supported range')
    for k in ['p','c','reference_F']:
        v=cfg.get(k)
        if k=='reference_F' and v is None:continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError(k+' must be finite and positive')
    if not 0<cfg['c_ratio']<=1:
        raise ValueError('c_ratio must be in (0,1]')
    for k in ['case','master','geometry']:
        if not isinstance(cfg.get(k),str) or not cfg[k].strip():
            raise ValueError(k+' must be nonempty text')


def validate_plan(plan):
    for k in ['cases','arms','seeds']:
        if not plan.get(k) or len(plan[k])!=len(set(plan[k])):
            raise ValueError('Empty or duplicate '+k)
    expected={(c,s,a) for c in plan['cases'] for s in plan['seeds'] for a in plan['arms']}
    seen=set();directories=set()
    for warmup,items in [(False,plan['runs']),(True,plan.get('warmups',[]))]:
        for item in items:
            key=(item['case'],item['seed'],item['arm'])
            if not warmup:
                if key in seen:raise ValueError('Duplicate planned observation')
                seen.add(key)
            p=PurePosixPath(item['directory'])
            if p.is_absolute() or '..' in p.parts or ':' in str(p) or '\\' in str(p) or str(p)=='.' or str(p) in directories:
                raise ValueError('Invalid or duplicate run directory')
            directories.add(str(p))
            cfg=item['config'];validate_config(cfg)
            if cfg['case']!=item['case'] or cfg['seed']!=item['seed']:
                raise ValueError('Plan/config identity mismatch')
    if seen!=expected:raise ValueError('Plan does not cover every case/seed/arm exactly once')
    if any(plan['readout'].get(a) not in ['raw','strict-S24'] for a in plan['arms']):
        raise ValueError('Unknown readout')
