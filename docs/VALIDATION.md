# Release validation

Updated 2026-09-28. Functional acceptance and replay of archived observations do not establish a repeatable accuracy gain.

## Current public release

| Check | Result and scope |
|---|---|
| CPU tests | 22 passed: parser/readout, complete scheduling, failure retention, saved-output consistency, initial-seed statistics, data preflight, CPGR counters and CPU baseline orchestration. |
| Original 300-run reference | Reparsed the 300 preserved solver outputs with the current parser/summary. Raw/CER means and the primary bootstrap interval agree with the archived reference; no new neural solves. |
| Three-arm plan | 300 measured solves, three excluded warmups; original execution order and seed blocks preserved. |
| Frozen method/model assets | BPR/CPGR/CER source, neural engines, checkpoints, upstream neural runtime and public inputs unchanged. |
| BPR model consistency | Checkpoint, anchor and FP32 TorchScript consistency passed on checked CPU inputs (maximum difference 0). |
| CPU baseline provenance | All three executables match the specified upstream commit; upstream license retained. |
| FRW-AGF and MicroWalk | Each completed case8 through the new runner on local WSL Linux; outputs parsed and summaries generated. These are functional checks, not paper timing measurements. |
| FRW-FDM | Help and solver startup passed. The local case8 check was stopped at its 60-second inspection budget before completion; no completed FDM solve is claimed for this wrapper revision. The public runner itself has no such timeout. |
| GitHub CI | CPU checks and plan generation supplied and tested locally; no hosted Actions run is claimed. |

## Earlier GPU acceptance

On 2026-09-25 the unchanged method/model assets were built and checked on Ubuntu 24.04 / RTX 4090 / CUDA 12.6 with the stated Torch/TensorRT versions. Model loading, CUDA projection/selector/compensation, sampler integration, and P0/Full case8 solves passed. A fresh Python environment and isolated source rebuild also completed a Full case8 solve, reusing the server's system CUDA/TensorRT/Protobuf libraries.

These are earlier GPU checks, not fresh acceptance of every current Python-wrapper change. The later runtime-counter guard and repository-relative local-validation GPU entry have not received a fresh GPU run. Their CPU and archived-record checks are separate evidence.

## Not claimed

A clean Docker build, other-GPU deployment, retraining, a new complete paper accuracy batch, full CPU baseline batches, peak-memory studies and CER microtiming are not validated by this packaging revision. The Docker recipe's prior attempt was blocked during dependency download. Fixed initial seeds do not guarantee identical asynchronous trajectories, a fixed historical mean or a stable ranking.
