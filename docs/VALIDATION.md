# Release validation

Updated 2026-10-02. Functional acceptance and replay of archived observations do not establish a repeatable accuracy gain.

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

On 2026-09-25 the unchanged method/model assets were built and checked on Ubuntu 24.04 / RTX 4090 / CUDA 12.6 with the stated Torch/TensorRT versions. Model loading, CUDA projection/selector/compensation, sampler integration, and DeepRWCap/Full case8 solves passed. A fresh Python environment and isolated source rebuild also completed a Full case8 solve, reusing the server's system CUDA/TensorRT/Protobuf libraries.

These are earlier GPU checks, not fresh acceptance of every current Python-wrapper change. The later runtime-counter guard and repository-relative local-validation GPU entry have not received a fresh GPU run. Their CPU and archived-record checks are separate evidence.

## Not claimed

A clean Docker build, other-GPU deployment, retraining, a new complete paper accuracy batch, full CPU baseline batches, peak-memory studies and CER microtiming are not validated by this packaging revision. The Docker recipe's prior attempt was blocked during dependency download. Fixed initial seeds do not guarantee identical asynchronous trajectories, a fixed historical mean or a stable ranking.

## 2026-10-02 consistency and supplement audit

The [frozen implementation contract](IMPLEMENTATION_CONTRACT.md) was checked against the original aligned-batch configuration and archived training material. The five DeepRWCap engines, BPR engine, four upstream runtime files and twenty public geometry/reference files match the original frozen inventory. Released CPGR source matches the archived source after normalizing line endings.

All 300 original outputs gave identical results under the historical and released parsers. The 300 new training-seed Full outputs and 100 reused baseline outputs were independently checked against the published capacitance records. The CPU supplement script reproduces the accepted per-seed results to numerical precision. All 28 CPU unit tests passed, including rejection of missing/duplicate records, inconsistent references and altered capacitances. No new GPU solve or training was performed for this documentation/result-packaging audit.

The accepted three-model experiment is a separate result supplement, not a new execution of the release's three-arm benchmark command. Its comparison and model-release boundaries are documented with the data.

## Final manuscript and repository review

The TCAD_v0 manuscript's BPR optimizer/loss settings, conditional Gradient face symmetrization and CER logical-row boundary agree with the released implementation. Its code-availability statement describes the core implementation, frozen models, public benchmark inputs and evaluation configurations; the detailed release boundaries remain in [ARTIFACT_SCOPE.md](ARTIFACT_SCOPE.md).

The 28 CPU tests, 300-measured-solve/3-warmup plan and saved training-seed reaggregation were checked again. GitHub Actions completed successfully for the preceding repository revision `9aa6555`. This review did not execute new GPU solves or a clean Docker build. The CI workflow now also exercises the training-seed summary command.

The paper-name CLI aliases and report labels were checked with 31 CPU tests. New and legacy CLI names produce identical paper plans, and display formatting leaves saved statistical values and archived identifiers unchanged.
