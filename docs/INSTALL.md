# Installation

The supplied engines were deployed on **Ubuntu 24.04 x86_64, RTX 4090, CUDA 12.6, Python 3.12, Torch 2.6.0+cu126, Torch-TensorRT 2.6.0+cu126 and TensorRT 10.7**. Start from this combination. Changing the compilation architecture does not make these engines portable. The recorded driver and compiler versions are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For CPU-only FRW-AGF/MicroWalk/FRW-FDM use [CPU_BASELINES.md](CPU_BASELINES.md); the neural installation below is not required.

## Native Ubuntu installation

The current case7 workflow passed with a source rebuild in an existing matching environment. Earlier checks also used a fresh Python environment, but the most recent clean dependency installation stopped during download. See [validation coverage](VALIDATION.md); the commands below are not a claim that a fresh operating-system installation was tested.

Use an Ubuntu 24.04 x86_64 machine with a working NVIDIA driver (`nvidia-smi`). The commands below install operating-system dependencies with administrator privileges. They do not replace the driver. Run them only on a machine you administer; a configured research server may already have these packages.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl python3.12-venv python3.12-dev build-essential libprotobuf32t64
curl -fLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 \
  libnvinfer10=10.7.0.23-1+cuda12.6 \
  libnvinfer-plugin10=10.7.0.23-1+cuda12.6 \
  libnvonnxparsers10=10.7.0.23-1+cuda12.6
```

From this repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install -r requirements.txt
python scripts/check_environment.py --build-only
python scripts/build.py
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/smoke_test.sh --output runs/smoke
python scripts/benchmark.py --profile quick --output runs/quick
```

`--build-only` permits compilation without an attached GPU; it is not a runtime acceptance check. The final two commands need the supported GPU and create new directories. The smoke test has two case7 solves; the quick benchmark has nine case1 solves (three initial seeds per arm), covering the batch runner and repeated-result summary. See [the benchmark guide](../benchmarks/public10/README.md) for the example selection and full paper protocol.

## Optional data generation

The [GGFT workflow](DATA.md) uses Python, Git, Make and a C++17/OpenMP compiler on Linux. It does not require the neural environment. Generated datasets are not needed for public10 inference.

## Container recipe

The Dockerfile expresses the same system/Python dependencies and builds the extensions. A CUDA-enabled Docker host also needs NVIDIA Container Toolkit. From the repository root:

```bash
docker build -t care-rwcap:local .
docker run --rm --gpus all care-rwcap:local
mkdir -p runs
docker run --rm --gpus all \
  -v "$PWD/runs:/opt/care-rwcap/runs" \
  care-rwcap:local python scripts/benchmark.py --profile quick --output runs/container_quick
```

The container retains the RTX 4090 engine requirement. Its base image and package downloads require external network access. A clean build of this recipe has not yet been completed; see [validation status](VALIDATION.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing `libnvinfer.so.10`, plugin or parser library | Install the three matching system TensorRT packages above; Python packages alone do not supply the solver's full runtime. |
| Missing `libprotobuf.so.32` | Install Ubuntu 24.04 `libprotobuf32t64`. |
| Undefined LibTorch symbol / ABI error | Use the exact Torch and Torch-TensorRT versions, ABI 1, and rebuild. Avoid inherited `LD_PRELOAD`; the runner manages its libraries. |
| `nvcc` not found | Set `CUDA_HOME` to the CUDA Toolkit 12.6 directory. The driver-reported CUDA version is not the toolkit version. |
| GPU or TensorRT engine mismatch | Use the supported environment. The release has no validated fallback engine or other-GPU conversion path. |
| Existing output directory | Choose a new directory. Do not mix runs or overwrite a previous comparison. |
| Different numerical result with the same seed | Check inputs, models, configuration and readout first; then assess repeated-run distributions. Asynchronous execution is not bitwise deterministic. |

Dependency references: [CUDA 12.6 installation](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
