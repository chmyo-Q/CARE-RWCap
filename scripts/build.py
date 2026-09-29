#!/usr/bin/env python3
"""Build CPGR, initial-seed support, and the sampler integration check."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import torch
import torch_tensorrt
from common import ROOT, dump, check_files

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--cuda-home', type=Path, default=Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')))
ap.add_argument('--arch', default='89', help='CUDA SM architecture; frozen engines target RTX 4090')
args = ap.parse_args()
if platform.system() != 'Linux':
    raise SystemExit('The frozen solver requires Linux x86_64.')
check_files()
if not args.arch.isdigit():
    raise ValueError('--arch must contain digits only')
b = ROOT/'build'
b.mkdir(exist_ok=True)
t, tt, cu = Path(torch.__file__).parent, Path(torch_tensorrt.__file__).parent, args.cuda_home
abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
inc = ['-I'+str(p) for p in [t/'include', t/'include/torch/csrc/api/include', tt/'include',
                            ROOT/'third_party/deeprwcap/include', cu/'include']]
base = ['g++', '-O2', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI='+abi]
libs = ['-L'+str(t/'lib'), '-L'+str(cu/'lib64'), '-ltorch', '-ltorch_cpu', '-ltorch_cuda',
        '-lc10', '-lc10_cuda', '-lcudart', '-ldl', '-pthread']
commands = [
    [str(cu/'bin/nvcc'), '-O3', '-std=c++17', '-arch=sm_'+args.arch, '-Xcompiler', '-fPIC',
     '-c', str(ROOT/'cpp/cpgr/projection.cu'), '-o', str(b/'projection.o')],
    base + ['-shared', '-fPIC', str(ROOT/'cpp/cpgr/sampler.cpp'), str(b/'projection.o')] + inc + libs + ['-o', str(b/'cpgr.so')],
    base + ['-shared', '-fPIC', str(ROOT/'cpp/seed_control/seed_control.cpp')] + inc + libs + ['-o', str(b/'seed_control.so')],
    base + [str(ROOT/'tests/integration_qa.cpp')] + inc +
    ['-L'+str(ROOT/'third_party/deeprwcap/runtime'), '-L'+str(tt/'lib'), '-Wl,-rpath-link,'+str(tt/'lib'), '-ldnnsolver'] +
    libs + ['-o', str(b/'integration_qa')],
]
dump(b/'commands.json', commands)
for i, cmd in enumerate(commands):
    print(f'Building {i+1}/{len(commands)}: {Path(cmd[-1]).name}', flush=True)
    with (b/f'compile_{i}.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Build failed: inspect build/compile_{i}.log')
solver = ROOT/'third_party/deeprwcap/runtime/deepRWCap'
solver.chmod(solver.stat().st_mode | 0o111)
dump(b/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
    'torch_tensorrt': torch_tensorrt.__version__, 'cxx11_abi': int(abi), 'cuda_toolkit': str(cu),
    'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    'compute_capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
    'gpu_runtime_validated': False,
    'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0]})
print('Build complete')
