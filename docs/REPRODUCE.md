# Reproduce CARE-RWCap

The supplied engines were deployed on **Ubuntu 24.04 x86_64, RTX 4090, CUDA 12.6, Python 3.12, Torch 2.6.0+cu126, Torch-TensorRT 2.6.0+cu126 and TensorRT 10.7**. The table below lists the deployment versions.

## Deployment environment

| Component | Environment used for the supplied deployment |
|---|---|
| OS / architecture | Ubuntu 24.04 / Linux x86_64 |
| GPU | NVIDIA GeForce RTX 4090, SM 8.9 |
| NVIDIA driver | 570.124.04 |
| Python | 3.12.13 |
| CUDA toolkit | 12.6, nvcc 12.6.85 |
| PyTorch / Torch-TensorRT | 2.6.0+cu126 / 2.6.0+cu126 |
| TensorRT Python | 10.7.0.post1 |
| System TensorRT libraries | 10.7.0.23-1+cuda12.6 |
| System Protobuf | libprotobuf32t64 3.21.12-8.2ubuntu0.3 |
| G++ | 13.3.0; LibTorch CXX11 ABI = 1 |
| NumPy | 2.4.4 |

Install CUDA Toolkit 12.6 and the matching system libraries using the NVIDIA/Ubuntu package repositories. Required runtime packages include `libnvinfer10`, `libnvinfer-plugin10`, `libnvonnxparsers10`, and `libprotobuf32t64`. The Python requirements file does not install these operating-system dependencies or the GPU driver.

Use a Python 3.12 environment, install `requirements.txt`, and run `python scripts/build.py`. The build script discovers Torch headers and libraries from that active Python environment. It compiles for SM 89 by default. The supplied engines use SM 89 and the TensorRT version listed above.

The driver-reported CUDA compatibility level (12.8 on the tested machine) is not the installed toolkit version (12.6). Deployment engines are TensorRT FP16, whereas the supplied uncompiled TorchScript and state dictionaries are FP32. Use the supplied FP16 engines for the paper configuration. Other GPU/TensorRT combinations require separate engine compilation.

## Native Ubuntu installation

The case1 examples and case7 smoke workflow were tested in the deployment environment above. the smoke workflow below lists the build, runtime and dataset checks.

On Ubuntu 24.04 x86_64, check that `nvidia-smi` works, then install any missing system dependencies:

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
python scripts/smoke_test.py --output runs/smoke
python scripts/benchmark.py --profile quick --output runs/quick
```

`--build-only` checks the compiler and libraries without requiring an attached GPU. The final two commands need the supported GPU and create new directories. The smoke test has two case7 solves; the quick benchmark has nine case1 solves (three initial seeds per arm), covering the batch runner and repeated-result summary. See the public benchmark commands below for the example selection and full paper protocol.

## Optional data generation

The [GGFT workflow](#data-generation) uses Python, Git, Make and a C++17/OpenMP compiler on Linux. It does not require the neural environment. Generated datasets are not needed for public10 inference.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing `libnvinfer.so.10`, plugin or parser library | Install the three matching system TensorRT packages above; Python packages alone do not supply the solver's full runtime. |
| Missing `libprotobuf.so.32` | Install Ubuntu 24.04 `libprotobuf32t64`. |
| Undefined LibTorch symbol / ABI error | Use the exact Torch and Torch-TensorRT versions, ABI 1, and rebuild. Avoid inherited `LD_PRELOAD`; the runner manages its libraries. |
| `nvcc` not found | Set `CUDA_HOME` to the CUDA Toolkit 12.6 directory. The driver-reported CUDA version is not the toolkit version. |
| GPU or TensorRT engine mismatch | Match the GPU, Torch-TensorRT and TensorRT versions in the deployment table. |
| Existing output directory | Choose a new directory. Do not mix runs or overwrite a previous comparison. |
| Different numerical result with the same seed | Check inputs, models, configuration and readout first; then assess repeated-run distributions. Asynchronous execution is not bitwise deterministic. |

Dependency references: [CUDA 12.6 installation](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).


## Container setup

The Dockerfile uses Ubuntu 24.04 and the same CUDA, Python and TensorRT versions as the native setup. A host NVIDIA driver and NVIDIA Container Toolkit are required for GPU access.

```bash
docker build -t care-rwcap .
docker run --rm -it --gpus all -v "$PWD/runs:/opt/CARE-RWCap/runs" care-rwcap
```

Inside the container, run `python scripts/check_environment.py` and the smoke command below. On another GPU, rebuild the extension for its SM architecture and regenerate the engines as described below.

## Run and evaluate

```bash
python scripts/run.py --arm care-rwcap --output runs/full_case1
python scripts/smoke_test.py --output runs/smoke
python scripts/benchmark.py --profile quick --output runs/quick
python scripts/benchmark.py --profile paper --plan-only --output runs/paper_plan
python scripts/benchmark.py --profile paper --output runs/public10
python scripts/summarize.py --input runs/public10
```

The default example and quick comparison use case1. Smoke uses case7 to check CPGR activation and CER application. Quick has nine solves; the paper protocol has 300 measured solves and three excluded warmups. Use a new output directory for each command.

## Seed and repetition

The default case1 command uses initial seed **2029** and `-n 8 -p 0.01 -c 0.01 --c-ratio 0.95`. Case7 smoke uses `--c-ratio 0.3`. The seed-control extension initializes the Torch CPU/CUDA generators and solver-core seed before each process starts.

The paper profile uses seeds **2029–2038**, with the same initial seed for corresponding cases and methods. `configs/paper_protocol.json` contains the case settings and references. Execution follows a fixed shuffled order, preceded by one excluded case8 warmup per method at seed 2040. Training seed and solver seed are separate parameters.

The ten public layouts were used to evaluate module variants and inspect case-level behavior during development. CAPR epoch selection uses the fixed local Poisson validation score in [METHOD.md](METHOD.md#loss-and-model-selection), rather than public10 SelfCapErr. Public10 and the additional layouts have separate configurations.

## Metrics

- **SelfCapErr (%):** `100 * abs(C / C_ref - 1)`. Each solve retains raw and CER readouts. Reference capacitance is used only to compute error.
- **Macro error:** equal-weight mean of the per-case mean errors. Each public10 case has ten repeats. Tables distinguish within-case sample SD from the across-case SD of case means.
- **Workload:** total walks and walk-weighted hops/walk from solver output. Approximate steps are `round(walks * hops_per_walk)`; the reported hops value is already rounded.
- **Time:** solver-reported elapsed and CPU seconds. Python startup and model loading are outside these values.
- **Coupling NL1 (%):** `100 * sum(abs(abs(pred_j)-abs(ref_j))) / sum(abs(ref_j))`, over shared off-diagonal entries in the selected-master and reference rows. Average across repeats within each case, then equally across cases.

The main paper endpoints are DeepRWCap raw, CAPR+CER and CARE-RWCap CER. Default case1 uses master `APpolysub` and reference `1.591776e-14 F`; smoke uses case7 master `1`.

## Optional coupling-entry evaluation

Calculate coupling error from a saved case1 output:

```bash
python scripts/evaluate_readout.py --geometry benchmarks/public10/case1.cap3d \
  --output runs/full_case1/result.out --reference 1.591776e-14 \
  --coupling-reference benchmarks/public10/ref/case1.dspf \
  --json runs/full_case1/coupling_metrics.json
```

The result is `coupling_error.normalized_l1_percent`; the JSON also includes the shared-entry count and reference mass. DSPF/SPICE parsing follows the upstream benchmark format. The command requires a reference master with positive coupling mass on the shared entries.

## Smoke workflow

`scripts/smoke_test.py` checks model loading, CPGR kernels and sampler integration, then runs DeepRWCap and CARE-RWCap on case7. It checks CUDA status, CPGR activation and CER application, and writes a summary to the chosen output directory.

## Public10 batch evaluation

The [batch runner](../benchmarks/public10/README.md) runs all ten inputs in the prescribed three-method order. `--plan-only` writes the schedule without loading Torch or starting a solve. The summary rereads `result.out` and compares it with the saved configuration and metrics. Incomplete batches retain their outputs and coverage information; complete-table statistics require the full planned grid. A solver failure stops the batch for inspection.

## Paired comparisons

`benchmark_stats.py` averages error differences across the fixed cases within each initial-seed block, then computes percentile bootstrap intervals using 20,000 resamples and NumPy PCG64 seed 20260919. Intervals are pointwise, unadjusted across comparisons, and use initial-seed blocks as the sampling unit. A single block has no interval estimate. Shared initial seeds control initialization, not common trajectories.

Bootstrap stream indices are fixed by comparison: CAPR-CER vs DeepRWCap-raw uses 0, CARE vs DeepRWCap-raw uses 1, CARE vs CAPR-CER uses 2, and other contrasts use 3. Each stream block contains `draws × number_of_seeds` indices.

Raw and CER are readouts of the same solve. The three solver configurations provide CPGR comparisons conditional on CAPR; standalone DeepRWCap+CPGR requires a separate solver arm. The [results table](../results/README.md) lists the configurations and readout definitions.

## CAPR training-seed supplement

[Training-seed records](../results/training_seeds/README.md) contain 100 CARE solves for each of seeds 2029, 2039 and 2053. The DeepRWCap control is reused from the original experiment and matched by case and initial solver seed. The summary command reads capacitance records and generates CSV/JSON/Markdown tables. The across-model SD includes residual FRW variation.


## Train CAPR

Generate or supply a 100,000-record Poisson dataset in the GGFT format described below:

```bash
python scripts/train_capr.py --dataset data/ggft_100k/poisson.bin --output runs/train_capr
```

The defaults use training seed 2029, the fixed 90,000/10,000 split, AdamW and the risk-aware objective. The script saves the actual split and validation scores. New GGFT records support retraining; reproducing the paper's local-data measurements requires the original reference records described under local transition evaluation.

## Export and rebuild engines

Export a newly trained checkpoint:

```bash
python scripts/export_capr.py --checkpoint runs/train_capr/best.pt --output runs/export_capr
```

For CPU-only inspection, add `--torchscript-only`. To rebuild all six engines from the bundled FP32 sources on another GPU:

```bash
python scripts/export_capr.py --all-models --output runs/rebuilt_models
export CARE_MODEL_DIR="$PWD/runs/rebuilt_models"
python scripts/build.py --arch 89
python scripts/check_environment.py
python scripts/smoke_test.py --output runs/rebuilt_smoke
```

Replace `89` with the GPU's CUDA SM architecture. `CARE_MODEL_DIR` selects the rebuilt `capr/` and `deeprwcap/` subdirectories. TensorRT engine compatibility is checked by loading and running each model; changing the GPU name check alone does not make an engine portable.

## Data generation

Use [DeepRWCap revision 9cb7fc69ed5fb54ce03be6fa8556402e10451be6](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6/ggft). The source, Makefile and Eigen dependency are in its `ggft/` directory. The upstream MIT license and Eigen's own licenses apply. The workflow below uses this revision throughout.

On Linux, install Git, GNU Make and a C++17 compiler with OpenMP support. For Ubuntu:

```bash
sudo apt-get install -y git build-essential
```

From the CARE-RWCap repository root, prepare a separate upstream checkout:

```bash
git clone https://github.com/THU-numbda/deepRWCap.git ../deeprwcap-data
git -C ../deeprwcap-data checkout --detach 9cb7fc69ed5fb54ce03be6fa8556402e10451be6
```

The supplied Eigen dependency is used by the Makefile. This CPU workflow does not require PyTorch, TensorRT or a GPU.

## Generate data

First check the workflow with two samples of each kind:

```bash
python3 scripts/generate_data.py --upstream ../deeprwcap-data \
  --kind both --samples 2 --threads 1 --output data/ggft_example
```

To use the sample count and thread count in the upstream generation script:

```bash
python3 scripts/generate_data.py --upstream ../deeprwcap-data \
  --kind both --samples 100000 --threads 12 --output data/ggft_100k
```

Choose a new output directory. `--kind poisson` and `--kind gradient` generate only the selected dataset. The wrapper checks the upstream revision and unchanged tracked GGFT sources, rebuilds GGFT, runs the two datasets sequentially, and validates each file's header and record count. It preserves `build.log`, per-dataset logs and `generation.json` alongside the generated files.

Each 100,000-sample file is **12,362,400,016 bytes** (about 12.36 GB). The two final files total about 24.72 GB; allow additional space for the thread-local files during merging. Use the two-sample command to check the build and file format.

## Generation parameters and randomness

The pinned upstream code uses:

| Parameter | Value |
|---|---|
| Grid size | `N = 23` |
| Dielectric block width | `1` |
| FDM iterative tolerance | `1e-6` |
| Initial random blocks | `5`, followed by upstream nesting/padding/background rules |
| Structure slots | 15 block slots plus one background slot |
| Poisson mode | `GFT` |
| Gradient mode | `WVTZ` |
| Random generator | `std::mt19937`, seeded with the OpenMP thread index |

Thread count changes the random streams, workload partition and concatenated sample order. GGFT uses thread-index seeds, independently of the CAPR training and FRW solver seeds. Keep the generation thread count and source version fixed when comparing generated datasets; compiler/library differences can also affect numerical results.

## Binary layout

The documented Linux x86_64 output uses little-endian float64 values throughout. Each file begins with two float64 header values, `N` and `block_width`; these are not integer-encoded fields. Each subsequent record contains:

| Field | Values per record | Meaning |
|---|---:|---|
| Dielectric cube | `23^3 = 12,167` | Permittivity samples |
| Structure description | `16 * 7 = 112` | Center, radii and permittivity for each structure slot |
| Reference kernel | `6 * 23^2 = 3,174` | Values on the six cube faces |
| Total | `15,453` | `123,624` bytes per record |

The structure description is retained in the file but is not an input to the released CAPR network. The arrays below follow the upstream storage and face conventions.

A minimal inspection with NumPy is:

```python
import numpy as np

path = "data/ggft_example/poisson.bin"
data = np.memmap(path, dtype="<f8", mode="r")
assert tuple(data[:2]) == (23.0, 1.0)
records = data[2:].reshape(-1, 15453)
dielectric = records[:, :12167].reshape(-1, 1, 1, 23, 23, 23)
structure = records[:, 12167:12279].reshape(-1, 16, 7)
kernels = records[:, 12279:].reshape(-1, 6, 1, 23, 23)
```

## Connection to CARE-RWCap

For CAPR, the recorded training recipe normalizes each dielectric cube by its spatial maximum (using divisor 1 for a zero maximum). The face-zero target is `abs(kernel) + 1e-10`, normalized over its 23-by-23 face. The fixed split uses a NumPy PCG64 permutation with seed 20260805: 90,000 training samples and 10,000 validation samples. See [the model and loss](METHOD.md) and [training recipe](../configs/capr_training_recipe.json).

CPGR uses signed Gradient predictions and introduces no trainable checkpoint. Gradient data supply numerical reference kernels for local evaluation. CER acts on the solver's output capacitance row and does not use GGFT data.

The repository includes CAPR training and export scripts, architecture and loss. GGFT produces new reference samples for further training or analysis. The local evaluation below uses the original reference files and checks their identity against the saved configuration.

## Local transition evaluation

This optional entry evaluates the deployed CAPR and CPGR operators without training, new labels or end-to-end solves. Set up the deployment environment and build the extension first.

```bash
python scripts/evaluate_transitions.py --help
python scripts/evaluate_transitions.py --data-dir /path/to/original_data \
  --module all --preflight-only --output runs/local_preflight
python scripts/evaluate_transitions.py --data-dir /path/to/original_data \
  --module all --output runs/local_validation
```

The evaluator requires the original `poisson.bin` and `gradient.bin`, each with 100,000 records (12,362,400,016 bytes). These datasets are external to the repository and currently have no public download endpoint. Reproducing the local tables requires these original files; the GGFT commands generate new samples. Public10 inference uses the bundled layouts and models.

The fixed Poisson validation order is in `configs/transition_validation.json`: the last 10,000 indices of a NumPy PCG64 permutation of 100,000 records with seed 20260805. Gradient uses its separate fixed indices `[90000,100000)`. Gradient1 is evaluated on canonical face 1 and Gradient2 on canonical face 2; the two head evaluations of a record are counted separately. Activation is evaluated separately for each of the two Gradient heads. The evaluator uses frozen FP16 engines, FP32 production CUDA projection and FP64 metrics. It records its actual environment and preserves failed outputs. `--preflight-only` checks the files and split before GPU evaluation.

Use `--module capr` for Poisson-only evaluation or `--module cpgr` for Gradient evaluation.

Outputs include `protocol.json`, `status.json`, `summary.json`, `summary.md` and paired sample records in compressed CSV. CAPR reports KL (including mean, Q95 and Q99), six-probe Action MSE and TV(CAPR,DeepRWCap), using the normalized probabilities actually sampled from the FP16 outputs. TV is refinement magnitude, not error against the reference. CPGR reports Gradient1/2 activation counts, active-only normalized L2, parity violations, exact-label parity and improvement fractions. The weight check uses one measure: normalized L1 error of effective six-face absolute mass. The seventh selector output retains its original value.

The Markdown summary includes mean/Q95/Q99 KL, Action MSE, relative changes and TV for CAPR; strict-active per-head and pooled NL2, improvement percentages and reference/raw/refined parity for CPGR; and the single effective-face-mass NL1 measure for weight compensation. The pooled NL2 uses active-count weighting (473 and 476 in the paper reference), not an equal-weight average of the two head means. A negative relative change denotes a reduction; the paper's reduction column uses the opposite sign.

Local activation uses each head's canonical validation states as the denominator. Solver-visited activation uses Gradient calls collected during FRW solving.

## CPU baselines

Obtain `rwcap_agf`, `rwcap_microwalk` and `rwcap_fdm` from the pinned [DeepRWCap baseline directory](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6/executable/baselines). The GGFT upstream checkout above includes these files. They use Linux x86_64 CPU execution and need no neural environment.

```bash
chmod +x ../deeprwcap-data/executable/baselines/rwcap_*
python scripts/baselines.py --method agf --binary-dir ../deeprwcap-data/executable/baselines --output runs/agf
```

Use `--method microwalk` or `--method fdm` for the other methods, and `--profile paper` for ten cases with ten seeds. These methods use raw readout and 16 threads by default. The runner saves per-run and per-case CSVs and solver timing.

## Training-seed records

```bash
python scripts/summarize_training_seeds.py --output runs/training_seeds
```

This command summarizes the saved records under `results/training_seeds/`; it does not train models or launch solves.
