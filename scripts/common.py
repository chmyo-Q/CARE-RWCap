"""Repository-relative configuration and the frozen Linux runtime environment."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
NAMES = ['PoissonSelector', 'PoissonPredictor', 'GradientSelectorWeight',
         'Gradient2Predictor', 'Gradient1Predictor']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_files():
    paths = set(model_paths(False) + model_paths(True))
    paths.update(ROOT/'third_party/deeprwcap/runtime'/n for n in
                 ['deepRWCap', 'libdnnsolver.so', 'librwcap.so', 'librwcapall.so'])
    for p in paths:
        if not p.is_file() or not p.stat().st_size:
            raise RuntimeError('Required file missing: ' + str(p.relative_to(ROOT)))
    return len(paths)


def model_paths(bpr):
    return [ROOT / 'models' / ('capr' if bpr and n == 'PoissonPredictor' else 'deeprwcap') /
            (n + '_tensorrt_fp16.jit') for n in NAMES]


def runtime_env(cpgr=False, seed=None):
    import torch
    import torch_tensorrt
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('S29_', 'PAPER_SOLVER_')) and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    dirs = [ROOT/'third_party/deeprwcap/runtime', Path(torch.__file__).parent/'lib',
            Path(torch_tensorrt.__file__).parent/'lib', cuda/'lib64', Path(sys.prefix)/'lib']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, dirs))
    preload = []
    if seed is not None:
        if not isinstance(seed, int) or not 1 <= seed <= 2147483647:
            raise ValueError('seed must be an integer in [1, 2147483647]')
        preload.append(ROOT/'build/seed_control.so')
        env['PAPER_SOLVER_SEED'] = str(seed)
    if cpgr:
        preload.append(ROOT/'build/cpgr.so')
        env.update(S29_GRADIENT_PARITY_ENABLE='1', S29_GRADIENT_JOINT_ENABLE='1')
    for p in preload:
        if not p.is_file():
            raise RuntimeError(f'{p.name} missing; first run python scripts/build.py')
    if preload:
        env['LD_PRELOAD'] = ':'.join(map(str, preload))
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    return env
