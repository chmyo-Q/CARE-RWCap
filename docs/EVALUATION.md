# Evaluation

[Installation](INSTALL.md) · [Public10 commands](../benchmarks/public10/README.md) · [Paper results](../results/README.md)

## Commands and paths

Run commands from the repository root. Each invocation creates a new output directory containing its configuration, model list, logs and results. The runner sets library paths and method switches, preserves `CUDA_VISIBLE_DEVICES` when supplied, and otherwise uses GPU 0. OMP/MKL threads default to 1; the FRW solver uses eight workers.

For build errors, inspect `build/compile_*.log`. For runtime errors, start with the run's `console.log` and the [deployment environment](INSTALL.md#deployment-environment).

## Seed and repetition

The default case1 command uses initial seed **2029** and `-n 8 -p 0.01 -c 0.01 --c-ratio 0.95`. Case7 smoke uses `--c-ratio 0.3`. The seed-control extension initializes the Torch CPU/CUDA generators and solver-core seed before each process starts.

The paper profile uses seeds **2029–2038**, with the same initial seed for corresponding cases and methods. `configs/paper_protocol.json` contains the case settings and references. Execution follows a fixed shuffled order, preceded by one excluded case8 warmup per method at seed 2040. Training seed and solver seed are separate parameters.

Asynchronous thread scheduling and inference batching affect the assignment of random draws to trajectories. Repeated solves can therefore differ with the same initial seed. Use complete repeated batches with the same models, references, settings and readout when comparing methods. Public10 was used during development; it is a fixed benchmark rather than an unseen test set.

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

The original bootstrap stream indices are retained: CAPR-CER vs DeepRWCap-raw uses 0, CARE vs DeepRWCap-raw uses 1, CARE vs CAPR-CER uses 2, and other contrasts use 3. Each stream block contains `draws × number_of_seeds` indices.

Raw and CER are readouts of the same solve. The three solver configurations provide CPGR comparisons conditional on CAPR; standalone DeepRWCap+CPGR requires a separate solver arm. The [results table](../results/README.md) lists the configurations observed in the original batch.

## CAPR training-seed supplement

[Training-seed records](../results/training_seeds/README.md) contain 100 CARE solves for each of seeds 2029, 2039 and 2053. The DeepRWCap control is reused from the original experiment and matched by case and initial solver seed. The summary command reads capacitance records and generates CSV/JSON/Markdown tables. The across-model SD includes residual FRW variation.

## Tested workflows

Builds, environment checks, model loading, CPGR kernels, sampler integration, case1 examples, case7 smoke and the nine-run quick profile were tested on Ubuntu 24.04 / RTX 4090. Local FP16 evaluation used the original transition datasets. GGFT was tested with two records per kind. The current CPU suite has 40 tests; all 11 CLI help entries and the public10 plans also pass.

The original 300-run results and the later 100-run CARE-only check used the same model files:

| Batch | DeepRWCap raw (%) | CARE raw (%) | CARE CER (%) |
|---|---:|---:|---:|
| Original three-method experiment | 0.9278 | 0.8356 | 0.8110 |
| Later CARE-only check | — | 0.8507 | 0.9053 |

In the later check, CER increased macro error by 0.0546 pp, with a pointwise 95% seed-block interval of [0.0117, 0.0994] pp. Its DeepRWCap/CAPR control arms were not rerun. CPGR's incremental end-to-end gain has also varied across checked batches; the original ordering is not consistent across all batches.

These checks used the configured research server. A fresh neural dependency installation, other GPUs and full training/export have not been tested. Earlier AGF/MicroWalk single-case checks completed; an earlier FDM check stopped before completion. Dedicated memory measurements use a separate protocol from the runner's timing output.

## Experiment commands

| Experiment | Command or material | Inputs and outputs |
|---|---|---|
| Public10 SelfCapErr | `benchmark.py --profile paper` | Bundled layouts, references and models; raw/CER errors, workload and timing. |
| Local CPGR/CAPR | `evaluate_transitions.py` | Original external transition datasets; [data instructions](DATA.md#local-transition-evaluation). |
| Component comparisons | Three solver methods with raw/CER readouts | Six configurations listed in [results](../results/README.md). |
| Additional layouts | [Five layout configs and execution schedule](../benchmarks/additional/README.md) | Bundled inputs and references; both methods use CER. |
| Training-seed tables | `summarize_training_seeds.py` | Saved capacitance CSVs; extra seed-2039/2053 model files are external to this package. |
| Coupling NL1 | `evaluate_readout.py --coupling-reference` | Saved solver output and DSPF/SPICE reference. |
| CPU baselines | `baselines.py` | Bundled upstream executables; [instructions](CPU_BASELINES.md). |
| Gradient activation | CPGR counters in run outputs | Projection and face-selector counts. |
| τ sensitivity | `capr_training_recipe.json` | Five-bound post-hoc settings and checkpoint-selection epochs. |

Each inference command evaluates one master conductor. The code provides CAPR architecture/loss, CPGR, CER and their deployment workflows. The training recipe records the settings used for CAPR; the complete training/export driver and transition datasets are external. Solver execution uses the prebuilt DeepRWCap runtime. See [models](../models/README.md) for checkpoint files and Python interfaces.
