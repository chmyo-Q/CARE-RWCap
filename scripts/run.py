#!/usr/bin/env python3
"""One isolated end-to-end run using the released paper models."""
import argparse
import json
from pathlib import Path
import subprocess
from common import ROOT, dump, model_paths, runtime_env, check_files
from evaluate import evaluate
from validation import validate_config, validate_cpgr_counts

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config', type=Path, default=ROOT/'configs/example.json')
ap.add_argument('--arm', choices=['p0', 'bpr', 'full'], default='full')
ap.add_argument('--output', type=Path, required=True, help='New directory; existing directories are never overwritten')
ap.add_argument('--seed', type=int, help='Override initial solver seed; does not fix asynchronous trajectories')
a = ap.parse_args()
check_files()
cfg = json.loads(a.config.read_text(encoding='utf-8'))
if a.seed is not None:
    cfg['seed'] = a.seed
validate_config(cfg)
geom = (ROOT/cfg['geometry']).resolve()
if not geom.is_file():
    raise FileNotFoundError(geom)
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
models = model_paths(a.arm in ['bpr','full'])
(out/'models.txt').write_text('\n'.join(map(str, models))+'\n')
env = runtime_env(cpgr=a.arm == 'full', seed=cfg['seed'])
env.update(OMP_NUM_THREADS=str(cfg['omp_threads']), MKL_NUM_THREADS=str(cfg['mkl_threads']))
cmd = [str(ROOT/'third_party/deeprwcap/runtime/deepRWCap'), '-f', str(geom),
       '-n', str(cfg['workers']), '-p', str(cfg['p']), '-c', str(cfg['c']),
       '--c-ratio', str(cfg['c_ratio']), '--out-file', str(out/'result.out'), '--log-file', str(out/'result.log')]
dump(out/'run.json', {'arm':a.arm,'configuration':cfg,'command':cmd,
    'models':[p.relative_to(ROOT).as_posix() for p in models],
    'environment':{k:env.get(k) for k in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','PAPER_SOLVER_SEED','S29_GRADIENT_PARITY_ENABLE','S29_GRADIENT_JOINT_ENABLE']}})
with (out/'console.log').open('w') as f:
    subprocess.run(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
seed = json.loads((out/'SEED_AUDIT.json').read_text())
if (seed['requested_seed'] != cfg['seed'] or not seed['torch_manual_seed_called']
        or seed['core_initial_seed_set'] != cfg['seed'] or seed['core_set_seed_calls'] <= 0):
    raise RuntimeError('Seed helper did not initialize requested seed')
if a.arm == 'full':
    projection=json.loads((out/'PROJECTION_COUNTS.json').read_text())
    selector=json.loads((out/'JOINT_COUNTS.json').read_text())
    validate_cpgr_counts(projection,selector)
result = evaluate(geom, out/'result.out', cfg.get('reference_F'), cfg['master'])
result.update(case=cfg['case'], arm=a.arm, initial_seed=cfg['seed'])
dump(out/'metrics.json', result)
print(json.dumps({k:v for k,v in result.items() if k!='matrix_F'}, indent=2))
