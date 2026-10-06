# Release validation

Updated 2026-10-06. The case1 single-run/quick examples and separate case7 smoke workflow were checked on Ubuntu 24.04 with an RTX 4090, Python 3.12, Torch/Torch-TensorRT 2.6.0+cu126, TensorRT 10.7 and CUDA Toolkit 12.6. Checks used the existing matching dependency environment and rebuilt the extensions from source.

## Current checks

| Check | Result |
|---|---|
| CPU tests | 35 passed locally and on the server. Coverage includes parser/CER behavior, scheduling, separate example/smoke configurations, saved-output consistency, local-data preflight and GGFT file boundaries. |
| Build and environment | Extension compilation and the supported-GPU environment check passed. |
| Model loading | BPR source predictions matched its archived TorchScript in the checked inputs; the frozen DeepRWCap anchor matched the supplied checkpoint. |
| CPGR | Projection, selector, compensation and sampler-integration checks passed. The case7 solve recorded actual CPGR activation. |
| Default example and smoke | The default Python command and shell example completed case1 solves. The separate two-solve smoke test used `configs/smoke_case7.json` and confirmed case7, CPGR activation and CER application. These check execution, not a statistical performance claim. |
| Quick comparison | The fixed case1 batch completed all nine solves: three arms, initial seeds 2029–2031. All observations were summarized. Case1 was selected from prior improved examples; this small batch does not establish public10 performance. |
| Readout commands | New `evaluate_readout.py` and compatible `evaluate.py` produced identical results from the same solver output. |
| Saved-output compatibility | The renamed parser/readout reparsed 300 stored solver outputs without changes to the original numerical fields. New `cer_*` fields equal their retained `s24_*` aliases. |
| Paper protocol | The full 300-run plan and excluded warmups are unchanged. All ten benchmark cases remain included. No new full public10 batch was required for these interface changes. |
| Transition evaluation | The renamed entry completed on the original external datasets. All 207 numerical fields in its BPR, CPGR and weight summary matched the preceding frozen evaluation. |
| GGFT generation | The wrapper built the pinned upstream source and generated two Poisson and two Gradient records; headers and complete record counts passed. This validates the small-data workflow, not a new 100,000-sample training dataset. |

The frozen deployment engines, checkpoints, CPGR/seed-control numerical implementation and reference capacitances were not changed by this release update.

## Earlier results and remaining scope

The historical paper cohort and the separately saved BPR training-seed observations are described under [results](../results/README.md). Reaggregating those observations is not a new inference experiment. A separate public10 rerun completed on 2026-10-03 using RTX 4090 D; the incremental CPGR benefit did not retain the historical ranking. The strict public environment checker accepts RTX 4090; the 4090 D run is not a portability guarantee. See [the evaluation protocol](REPRODUCIBILITY.md).

Earlier FRW-AGF and MicroWalk case8 checks completed; FRW-FDM help/startup was checked, but its case8 solve was stopped before completion. The current change updates their default quick input to case7 and tests their orchestration with synthetic fixtures; it does not claim fresh traditional-solver case7 measurements.

A clean Docker build, complete retraining/export, all traditional-baseline batches, additional-layout experiments, peak GPU memory and CER microtiming are outside this verification. A fresh dependency installation was not completed in this check; the recorded runs used the existing matching environment.

Original local reference datasets remain external. The [GGFT workflow](DATA.md) generates new data and does not by itself reproduce those archived labels. Fixed solver initial seeds do not fix asynchronous trajectories or guarantee the historical error values. See [release scope](ARTIFACT_SCOPE.md) for the included workflows.
