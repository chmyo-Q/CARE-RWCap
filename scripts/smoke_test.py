#!/usr/bin/env python3
"""Functional validation, not a statistical reproduction of paper tables."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, check_files, dump, model_paths, runtime_env

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,default=ROOT/'runs/smoke')
a=ap.parse_args()
out=a.output.resolve()
out.mkdir(parents=True,exist_ok=False)
check_files()

def run(cmd,name,env=None):
    cwd=out/name
    cwd.mkdir()
    print('Checking '+name,flush=True)
    with (cwd/'console.log').open('w') as log:
        subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return cwd

run([sys.executable,str(ROOT/'tests/test_readout.py')],'readout')
mc=run([sys.executable,str(ROOT/'tests/check_models.py')],'models')
kc=run([sys.executable,str(ROOT/'tests/check_kernels.py')],'kernels',runtime_env())
integ=out/'integration'
integ.mkdir()
(integ/'models.txt').write_text('\n'.join(map(str,model_paths(True)))+'\n')
print('Checking sampler integration',flush=True)
with (integ/'console.log').open('w') as log:
    subprocess.run([str(ROOT/'build/integration_qa')],cwd=integ,env=runtime_env(cpgr=True),
                   stdout=log,stderr=subprocess.STDOUT,check=True)
for arm in ['p0','full']:
    print('Checking end-to-end '+arm,flush=True)
    with (out/(arm+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'--arm',arm,'--output',str(out/arm)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics=json.loads((out/arm/'metrics.json').read_text())
    assert metrics['walks']>0 and metrics['s24_applied']
counts=json.loads((out/'full/PROJECTION_COUNTS.json').read_text())
joint=json.loads((out/'full/JOINT_COUNTS.json').read_text())
assert counts['cuda_status']==0 and counts['samples']>0 and sum(counts['masks'][1:])>0
assert joint['cuda_status']==0 and sum(joint['input_xyz_masks'][1:])>0
assert not (out/'p0/PROJECTION_COUNTS.json').exists()
report={'pass':True,'scope':'Functional acceptance only; two single-case runs do not establish accuracy gains.',
        'readout_unit_tests_passed':True,
        'model_check':json.loads((mc/'MODEL_CHECK.json').read_text()),
        'kernel_check':json.loads((kc/'KERNEL_CHECK.json').read_text()),
        'integration':json.loads((integ/'INTEGRATION_QA.json').read_text()),
        'e2e_arms':['p0','full'],'e2e_case':'case8','strict_s24_applied_both':True,
        'projection_counts':counts,'selector_counts':joint,
        'environment':json.loads((ROOT/'build/environment.json').read_text())}
dump(out/'summary.json',report)
print(json.dumps(report,indent=2))
