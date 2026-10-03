# Release validation

Updated 2026-10-03. This page separates functional checks, replay of saved observations and new inference. Successful execution or replay alone does not establish a repeatable accuracy gain.

## Completed checks of the public source

The current acceptance run started from public revision `204ecf6`. The solver settings, method implementation and frozen models were not changed for the checks below.

| Check | Result and scope |
|---|---|
| CPU tests | 31 passed on the local checkout and server; parser/CER, scheduling, failure retention, saved-output consistency, seed statistics, local-data preflight, naming and baseline orchestration. |
| Build and model loading | Fresh extension build, six FP16 engine checks, CUDA projection/selector/compensation and actual sampler integration passed on the server described below. |
| Solver entry points | Case8 smoke, three-arm quick benchmark, CARE-RWCap example and standalone evaluation passed. These are functional checks. |
| Frozen assets and protocol | Neural engines/runtime, CPGR and seed-control source, public inputs and reference values were checked against the original batch. Case10 reference text differs only in line endings; the reference value is identical. All 300 planned case/seed/arm tuples match. |
| Original 300-run replay | Public evaluation code reparsed all 300 original outputs without numerical mismatches. Main macro errors round to 0.9278%, 0.9001% and 0.8110%. This is archived-output replay, not new inference. |
| Local transition evaluation | Frozen FP16 BPR/CPGR evaluation completed using the original external reference datasets and fixed splits. All 26 checked manuscript display values match; strict-active Gradient1/Gradient2 sample counts are 473/476. Those datasets are not bundled. |
| Checkpoints | Five DeepRWCap FP32 state dictionaries match their uncompiled TorchScript model parameters and buffers; profiling counters are excluded. BPR checkpoint/anchor consistency also passed. |
| Training-seed supplement | Public CPU reaggregation reproduces the saved per-seed summaries. This check does not rerun three models. |

## Hardware and pending end-to-end acceptance

The original paper batch recorded **RTX 4090, driver 570.124.04**. The 2026-10-03 server reports **RTX 4090 D, driver 595.71.05**. Both checks use the stated Python 3.12 / Torch and Torch-TensorRT 2.6.0+cu126 / TensorRT 10.7 / CUDA toolkit 12.6 stack. The new build and inference checks use an existing dependency environment, not a fresh operating-system installation.

The public strict environment checker currently accepts the original RTX 4090 name only and therefore rejects the 4090 D name. The build and inference checks were invoked separately; this rejection is retained in the acceptance record. A narrowly scoped checker update is under review. Successful loading on this particular server does not establish general TensorRT engine portability.

A new 300-solve public10 evaluation is in progress. No final rerun accuracy, ranking, replication tolerance or runtime improvement is reported here yet. The hardware difference must be retained when interpreting its eventual results; it is not an established explanation for any numerical difference.

## Earlier checks and optional workflows

- On 2026-09-25, unchanged method/model assets passed GPU smoke and Full case8 checks on RTX 4090. A fresh Python environment and isolated source rebuild reused installed system CUDA/TensorRT/Protobuf libraries.
- FRW-AGF and MicroWalk each completed case8 under the public wrapper on local WSL Linux. FRW-FDM help/startup passed, but its case8 inspection was stopped before completion. The current acceptance does not rerun these traditional baselines.
- GitHub Actions completed for earlier revision `9aa6555`; the current CPU workflow also includes training-seed reaggregation. Local tests do not establish the hosted CI status of another revision.

## Not established by these checks

A clean Docker build, complete retraining, general GPU portability, all traditional-baseline batches, additional-layout experiments, peak GPU memory and CER microtiming have not been validated by this acceptance. The Docker recipe's earlier attempt stopped during dependency download.

Fixed initial seeds do not fix asynchronous trajectories. The historical 300-run cohort contains three solver arms with two readouts each; it has no standalone DeepRWCap+CPGR arm. See [result interpretation](../results/README.md), [evaluation protocol](REPRODUCIBILITY.md) and [release scope](ARTIFACT_SCOPE.md).
