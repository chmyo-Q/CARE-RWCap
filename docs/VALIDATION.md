# Release validation

Updated 2026-10-11 against manuscript v3. The checks use an independent checkout on Ubuntu 24.04 with an **RTX 4090**, Python 3.12.13, CUDA Toolkit 12.6, Torch/Torch-TensorRT 2.6.0+cu126 and TensorRT 10.7. Extensions were rebuilt from the released source in the existing matching neural environment.

## Completed checks

| Workflow | Verification |
|---|---|
| Build and environment | Python and Bash build entries, build prerequisites, runtime environment and installed dependency consistency passed. |
| CPU checks | All 37 tests passed. These cover parsing/CER, scheduling, summaries, naming compatibility, data boundaries and frozen local-data checks. |
| Model correspondence | The CAPR source matched its uncompiled TorchScript on the checked inputs, with maximum difference 0; its anchor matched the supplied DeepRWCap checkpoint. Parameter counts were 2,869 total and 1,466 trainable. |
| CPGR numerics and integration | All reflection masks, both face types, selector averaging, compensation, zero-projection behavior and sampler integration passed. Asymmetric inputs retained upstream behavior. |
| Single-case commands | The default Python command and Bash example completed case1 solves. |
| Smoke | The case7 workflow completed DeepRWCap and CARE-RWCap solves, with CPGR activation and CER application verified. |
| Quick | All nine case1 solves completed: three solver arms and initial seeds 2029–2031. Every planned observation was summarized. |
| Readout | `evaluate_readout.py` and the compatible `evaluate.py` produced identical results on the same output. |
| Paper plans | The three-arm plan contains 300 formal solves and three excluded warmups. CARE-only and CAPR/CARE subsets contain 100+1 and 200+2 solves. All ten public cases remain included. |
| Public10 CARE | All 100 formal solves and one excluded warmup completed; all outputs passed re-parsing and configuration checks. The new macro CER error was 0.9053%, versus 0.8110% in the original cohort. Raw was 0.8507%; CER increased this batch's error by 0.0546 pp (pointwise 95% seed-block interval: [0.0117, 0.0994] pp). No new DeepRWCap or CAPR-only cohort was run, so this check does not re-estimate their paired performance differences. |
| Saved results | The current reader reproduced all capacitances, errors, workload and solver times from the original 300 outputs. It also reparsed 100 additional-layout outputs and 300 training-seed outputs. These are record checks, not new inference. |
| Local FP16 evaluation | Inference completed on the original frozen external Poisson/Gradient files. All 203 numerical CAPR, CPGR and weight-summary fields matched the saved evaluation. Active-only Gradient counts were 473 and 476; pooled statistics weight these counts. |
| Saved local records | CPU reaggregation of 40,000 paired observations reproduced 198 numerical summary fields. |
| Training-seed supplement | The released capacitance records reaggregated to 0.8112%, 0.8038% and 0.8414%, with mean ± sample SD of 0.8188 ± 0.0199%. |
| GGFT | The wrapper rebuilt the pinned upstream source and generated two Poisson and two Gradient records. File headers and complete record counts passed. |
| CLI and documentation | All 13 CLI help entries, 38 Python files and 105 relative documentation links were checked. No broken file links were found. |

A separate Python 3.12.3 environment containing NumPy 2.4.4 and no Torch also passed all 37 CPU tests, the 300+3 paper plan and saved training-seed reaggregation. This checks the CPU workflow independently of the neural environment; it is not a fresh neural installation.

The release update changes documentation and citation metadata. The numerical source, 18 model files, reference capacitances and experiment settings remain unchanged.

## Interpretation and limits

Functional checks establish that the supplied implementation and documented inference workflows work in the supported environment. They do not establish a guaranteed historical error value, cross-batch ranking or universal accuracy tolerance. Fixed initial seeds do not fix asynchronous trajectories. CPGR's incremental end-to-end gain has varied across checked batches; see [the evaluation protocol](REPRODUCIBILITY.md).

The original 300-run table remains a specific historical cohort. Raw/CER results are dependent readouts of the same solve. Its three solver arms identify CPGR's effect conditional on CAPR; they do not provide a standalone DeepRWCap+CPGR arm or a complete factorial interaction.

The full traditional-baseline batches were not rerun in this check. Their CLI plans and orchestration are covered; earlier AGF/MicroWalk single-case checks completed, while an earlier FDM solve was stopped before completion. This release supplies these upstream executables for optional new measurements, not a freshly validated reproduction of every CPU table.

A fresh **neural** dependency installation, clean Docker build, complete retraining/export, other GPU targets and dedicated peak-memory studies were not completed. The compact release does not include the additional-layout dataset or original local reference datasets. The GGFT example generates new data; it does not recreate the archived labels. See [release scope](ARTIFACT_SCOPE.md) for the available workflows.
