# CARE-RWCap

**Condition-Aware Refinement for Neural-Guided Floating Random Walk Capacitance Extraction**

[Installation](docs/INSTALL.md) · [Method](docs/METHOD.md) · [Evaluation](docs/EVALUATION.md) · [Data](docs/DATA.md)

CARE-RWCap refines local transitions and endpoint readout in the [DeepRWCap](https://github.com/THU-numbda/deepRWCap) floating random walk solver. This repository provides the refinement source, frozen deployment models, ten public benchmarks, the additional-layout panel and evaluation commands.

## Overview

[![CARE-RWCap framework](docs/figures/overview.png)](docs/figures/overview.pdf)

1. **CPGR — Conditional Parity Gradient Refinement:** at eligible first neural Gradient transitions, average opposite-face probabilities, project signed kernels and compensate contributions under strict input symmetry. Gradient network parameters stay frozen.
2. **CAPR — Condition-Aware Poisson Refinement:** refine subsequent Poisson in-face distributions through bounded residual reweighting, retaining the frozen anchor and Poisson face selector.
3. **CER — Conditional Endpoint Re-estimation:** obtain raw and conditional coupling-based self-capacitance readouts from the same solve.

The runner evaluates one selected master per invocation. [Method](docs/METHOD.md) describes the operators, input/fallback rules and manuscript-to-source mapping.

## Install and run

The frozen FP16 engines target **Ubuntu 24.04 / Linux x86_64, RTX 4090, Python 3.12, CUDA Toolkit 12.6, Torch/Torch-TensorRT 2.6.0+cu126 and TensorRT 10.7**. Other combinations are not validated. Follow [installation](docs/INSTALL.md) to prepare system libraries and Python dependencies first.

```bash
git clone https://github.com/chmyo-Q/CARE-RWCap.git
cd CARE-RWCap
python scripts/check_environment.py --build-only
python scripts/build.py
python scripts/check_environment.py
python scripts/run.py --arm care-rwcap --output runs/full_case1
```

The default case1 example uses initial seed 2029. Each invocation needs a new output directory and saves `result.out`, `metrics.json`, `run.json` and logs. Metrics include raw/CER estimates, SelfCapErr, walks, hops and solver time. CER requires no extra solve.

Check the CPU logic or run the case7 smoke workflow:

```bash
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/smoke_test.sh --output runs/smoke
```

CPU tests need NumPy but no GPU. Smoke uses `configs/smoke_case7.json` to exercise CPGR activation and CER application; it needs the supported GPU environment.

## Evaluate public10

| `--arm` | Poisson predictor | CPGR | Main readout |
|---|---|---|---|
| `deeprwcap` | DeepRWCap | Off | Raw |
| `capr` | CAPR | Off | CER |
| `care-rwcap` | CAPR | On | CER |

```bash
# Small comparison: case1, seeds 2029–2031, three arms (nine solves)
python scripts/benchmark.py --profile quick --output runs/quick

# Inspect the full plan without a GPU or any solves
python scripts/benchmark.py --profile paper --plan-only --output runs/paper_plan

# Ten cases × ten seeds × three arms, plus three excluded warmups
python scripts/benchmark.py --profile paper --output runs/public10
```

Both raw and CER readouts are retained for every arm. The full protocol uses initial seeds 2029–2038 and includes all ten cases. Quick is an illustrative example, not a ten-case performance estimate. See [benchmark commands](benchmarks/public10/README.md) for subsets and output definitions, and [evaluation](docs/EVALUATION.md) for statistics and available paper workflows.

## Models and reference results

The ten layouts/references and frozen settings are in `benchmarks/public10/` and `configs/paper_protocol.json`. [Models](models/README.md) describes all deployment engines and inspectable checkpoints. The bundled baseline uses DeepRWCap architectures with locally trained weights, not the official upstream pretrained files. CAPR replaces only the Poisson predictor; CPGR and CER add no trained checkpoint.

| Historical configuration | Macro SelfCapErr (%) |
|---|---:|
| DeepRWCap, raw | 0.9278 |
| CAPR + CER | 0.9001 |
| CARE-RWCap: CAPR + CPGR + CER | 0.8110 |

These are observations from the original 300-run cohort. [All observed raw/CER configurations](results/README.md) identify the measured arms. Fixed initial seeds do not fix asynchronous trajectories, and CPGR's incremental gain has varied across checked batches. New values and rankings can differ; the reference is not an accuracy acceptance target. [Validation status](docs/EVALUATION.md#validation-status) records actual checks, including the later CARE-only macro of 0.9053%.

## Optional workflows

| Task | Entry and requirements |
|---|---|
| Evaluate the five additional layouts | [Inputs, screening and frozen configurations](benchmarks/additional/README.md) |
| Generate Poisson/Gradient data | [Pinned upstream GGFT workflow](docs/DATA.md); CPU, Git and C++ compiler |
| Evaluate local CAPR/CPGR mechanisms | [Local transition evaluation](docs/DATA.md#local-transition-evaluation); original external reference datasets and GPU |
| Compare FRW-AGF, MicroWalk or FRW-FDM | [CPU baselines](docs/CPU_BASELINES.md); bundled upstream Linux x86_64 executables |
| Reaggregate training-seed results | [Saved records](results/training_seeds/README.md); CPU-only command below |

```bash
python scripts/summarize_training_seeds.py --output outputs/training_seed_summary
```

The three saved training-seed macro errors are 0.8112%, 0.8038%, 0.8414% (mean ± sample SD: 0.8188 ± 0.0199%). This study reused the historical DeepRWCap control; the command reaggregates observations rather than rerunning the extra models.

## Repository structure

```text
src/                       CAPR architecture/loss and CER parser/readout
cpp/                       CPGR and initial-seed extensions
models/                    Frozen engines and inspectable checkpoints
benchmarks/                Public10 and additional-layout inputs/references
configs/                   Frozen execution and local-evaluation settings
scripts/                   Build, inference and evaluation entries/helpers
third_party/deeprwcap/     Upstream runtime, CPU baselines and license
results/                   Historical reference and training-seed records
tests/                     CPU/GPU implementation checks
docs/                      Installation, method, evaluation and data guides
```

The upstream solver cores are binary dependencies. Full training data/driver, dedicated memory studies and paper-figure scripts are not bundled. See [experiment coverage](docs/EVALUATION.md#available-experiments-and-release-scope) for the exact scope.

## Citation and license

[CITATION.cff](CITATION.cff) contains the manuscript title and authors; publication details will be added when available. Please also cite DeepRWCap when using its components:

```bibtex
@article{rodriguez2026deeprwcap,
  title={DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design},
  author={Rodriguez, Hector R. and Huang, Jiechen and Yu, Wenjian},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40}, number={2}, pages={971--979}, year={2026},
  doi={10.1609/aaai.v40i2.37066}
}
```

MIT; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). We thank the DeepRWCap authors for the solver, architectures, benchmarks and baseline executables. For problems, open an issue with the command, environment and relevant log excerpt.
