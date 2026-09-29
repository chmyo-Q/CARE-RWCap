#!/usr/bin/env python3
"""Check the supported frozen-engine environment before compiling or running."""
import platform
import subprocess
import sys
import argparse
import importlib
import os
import shutil
from pathlib import Path
from common import ROOT, check_files, runtime_env

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--build-only', action='store_true', help='Check build prerequisites without accepting the GPU runtime')
args = ap.parse_args()
modules = {}
for name in ['torch', 'torch_tensorrt', 'tensorrt']:
    try:
        modules[name] = importlib.import_module(name)
    except Exception as exc:
        print(f'{name}: {exc}. Install prerequisites following docs/INSTALL.md.', file=sys.stderr)
if len(modules) != 3:
    raise SystemExit(1)
torch, torch_tensorrt, tensorrt = (modules[n] for n in ['torch', 'torch_tensorrt', 'tensorrt'])
errors=[]
if sys.version_info[:2] != (3, 12):
    errors.append('Use Python 3.12 for the supplied environment.')
if platform.system()!='Linux' or platform.machine()!='x86_64':
    errors.append('The bundled solver requires Linux x86_64.')
if torch.__version__!='2.6.0+cu126':
    errors.append('Expected torch==2.6.0+cu126; found '+torch.__version__)
if torch_tensorrt.__version__!='2.6.0+cu126':
    errors.append('Expected torch_tensorrt==2.6.0+cu126; found '+torch_tensorrt.__version__)
if not tensorrt.__version__.startswith('10.7.'):
    errors.append('Expected TensorRT 10.7; found '+tensorrt.__version__)
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    errors.append('The upstream solver requires CXX11 ABI 1.')
if args.build_only:
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    if not (cuda/'bin/nvcc').is_file():
        errors.append('nvcc missing; install CUDA Toolkit 12.6 and set CUDA_HOME.')
    if not shutil.which('g++'):
        errors.append('g++ missing; install build-essential.')
elif not torch.cuda.is_available():
    errors.append('CUDA GPU unavailable.')
elif torch.cuda.get_device_name()!='NVIDIA GeForce RTX 4090':
    errors.append('Frozen engines are validated on NVIDIA GeForce RTX 4090 only; found '+torch.cuda.get_device_name())
check_files()
if platform.system()=='Linux':
    r=subprocess.run(['ldd',str(ROOT/'third_party/deeprwcap/runtime/deepRWCap')],
                     env=runtime_env(),text=True,capture_output=True)
    missing=[line.strip() for line in r.stdout.splitlines() if 'not found' in line]
    errors.extend(missing)
    if r.returncode:
        errors.append('ldd failed: '+r.stderr.strip())
if errors:
    print('\n'.join(errors),file=sys.stderr)
    raise SystemExit(1)
if args.build_only:
    print('Build prerequisites found. GPU runtime and frozen-engine loading remain unchecked.')
else:
    print('Supported environment found; run the smoke test to check engine loading and solver integration.')
