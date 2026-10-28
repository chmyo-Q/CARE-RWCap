# Evaluation

[Installation](INSTALL.md) · [Public10 commands](../benchmarks/public10/README.md) · [Historical results](../results/README.md)

## Commands and paths

Run commands from the repository root and choose a new output directory for each invocation. Input and model paths are repository-relative; each run creates its own `models.txt`. The runner manages library paths, clears inherited method switches and preserves an explicit `CUDA_VISIBLE_DEVICES` (otherwise device 0). OMP/MKL threads default to 1, independently of the solver's eight workers.

For build or loading errors, inspect `build/compile_*.log` or the run's `console.log` and check the [deployment environment](INSTALL.md#deployment-environment). CPGR runs save projection/selector counters; the smoke workflow checks CUDA status and activation.

## Seed and repetition

The default case1 example uses initial seed **2029** and `-n 8 -p 0.01 -c 0.01 --c-ratio 0.95`. Case7 smoke uses its own configuration with `--c-ratio 0.3`. The initial-seed extension initializes Torch CPU/CUDA generators and the exported solver-core seed before the main program, with a record of the actual initialization.

The paper protocol uses ten seed blocks, **2029–2038**, assigning the same initial seed to each case and method in a block. These are solver seeds, distinct from the CAPR training seed. `configs/paper_protocol.json` supplies all case settings, references and the predetermined order; `configs/example.json` supplies the single-run default. The full profile excludes three case8 warmups at seed 2040.

Fixed initial seeds do not fix asynchronous thread scheduling, inference batches or assignment of random draws to paths. Runs need not be bitwise equal, and no universal tolerance guarantees a historical aggregate or ranking. Compare complete batches under fixed models, references, readouts and settings, retaining all predetermined observations. Public10 was used during development and is not an unseen test set.

## Metrics

- SelfCapErr (%) = `100 * abs(C / C_ref - 1)`. Raw and CER are retained separately; references affect errors, not estimates or CER activation.
- The historical main comparison selects DeepRWCap raw, CAPR+CER and CARE-RWCap CER. Macro error equally averages the ten per-case mean errors, each from ten repeats. Within-case sample SD and between-case SD of case means are labeled separately.
- Walks and hops/walk come from solver output. Approximate steps are their product rounded to an integer; hops/walk is already rounded by the solver.
- Elapsed and CPU seconds are solver-reported values, excluding wrapper launch/model loading. Memory and CER micro-latency require separate measurement; this evaluator does not infer them from timing.
- Coupling-magnitude NL1 (%) = `100 * sum(abs(abs(pred_j)-abs(ref_j))) / sum(abs(ref_j))`, using shared off-diagonal entries of the selected-master and reference rows. Exclude self, average within each case, then equally across cases, as in manuscript v8 IV-D.

The default master is case1 `APpolysub`, with reference `1.591776e-14 F` from its DSPF. Smoke uses case7 master `1`. Parser fixtures in `tests/fixtures/` are synthetic readout inputs, not benchmark layouts.

## Optional coupling-entry evaluation

Reuse a saved case1 solve; no additional solver run is needed:

```bash
python scripts/evaluate_readout.py --geometry benchmarks/public10/case1.cap3d \
  --output runs/full_case1/result.out --reference 1.591776e-14 \
  --coupling-reference benchmarks/public10/ref/case1.dspf \
  --json runs/full_case1/coupling_metrics.json
```

Read `coupling_error.normalized_l1_percent`. JSON also records shared-entry count and reference mass. DSPF/SPICE parsing follows upstream benchmark conventions; missing masters and zero shared reference mass are errors. Omitting the flag preserves the default raw/CER readout.

## Smoke workflow

`scripts/smoke_test.py` checks readout, model loading, kernels and sampler integration, then runs DeepRWCap and CARE-RWCap with `configs/smoke_case7.json`. It requires actual CPGR activation and CER application, and saves its summary in the output directory. This is a functional check, not a paper-table or training reproduction.

## Public10 batch evaluation

The [batch runner](../benchmarks/public10/README.md) supplies all ten inputs and references and the original three-arm order. `--plan-only` needs no GPU. The summary reparses every `result.out`, checks saved metrics/configurations, rejects duplicate or incomplete grids and withholds complete-table comparisons if records are missing or inconsistent. There is no automatic retry or result-based exclusion. Zero Gradient visits remain valid for cases that do not use that branch.

## Component attribution and record checks

The summary reports same-readout contrasts alongside whole-method endpoints. Raw/CER are dependent readouts of one solve; the three-arm protocol measures CPGR conditional on CAPR, not standalone CPGR or the full CAPR-by-CPGR interaction. The [historical reference](../results/README.md) identifies its original cohort; cross-cohort percentage scaling cannot supply a missing observation.

For each initial seed, differences are averaged equally over fixed cases. `statistics_report.py` uses NumPy PCG64 seed 20260919 and 20,000 percentile bootstrap resamples of those blocks; one-seed intervals are omitted. Intervals are pointwise, with no multiple-comparison adjustment or automatic significance claim. Stream indices reproduce the original successive draws: CAPR-CER vs DeepRWCap-raw 0, CARE vs DeepRWCap-raw 1, CARE vs CAPR-CER 2, other contrasts 3. Each stream block contains `draws × number_of_seeds` indices. Matched initial seeds do not imply common random paths.

For FRW-AGF/MicroWalk/FRW-FDM, use the separate [CPU baseline instructions](CPU_BASELINES.md). Those programs do not load neural engines or apply CER.

## CAPR training-seed supplement

[Saved records](../results/training_seeds/README.md) cover training seeds 2029, 2039 and 2053, with 100 CARE solves per model and reused historical DeepRWCap records matched by case/initial seed. This is not a contemporaneous control or common-path study. The CPU-only command recomputes errors from capacitances and generates CSV/JSON/Markdown tables; derived tables are not duplicated in the repository. The extra two models are not bundled.

## Validation status

The numerical implementation was checked on 2026-10-11 in the supported Ubuntu 24.04 / RTX 4090 environment: source builds, environment checks, model/anchor equivalence, all CPGR masks and compensation, sampler integration, case1 examples, case7 smoke and nine case1 quick solves passed. The CARE-only batch completed 100 formal solves plus one excluded warmup; GGFT generated two records per kind. Local FP16 evaluation matched the frozen summary. These GPU records precede the interface/documentation cleanup; no new GPU solves were run for that cleanup.

The later CARE-only macro was **0.9053%** versus historical **0.8110%**. Its raw error was **0.8507%**; CER increased error by 0.0546 pp (pointwise 95% seed-block interval [0.0117, 0.0994] pp). No contemporaneous DeepRWCap/CAPR control was run, so this batch cannot re-estimate paired gains. CPGR's incremental end-to-end effect has varied between checked batches. Functional success does not establish stable accuracy gains.

After cleanup on 2026-10-28, all **40 CPU tests**, **11 CLI help entries**, source/checkpoint loading and the coupling command passed. The original main/additional execution orders were unchanged. Current readers reproduced the 300 main and 100 additional saved outputs; 40,000 local paired observations and the training-seed records reaggregated correctly. Coupling NL1 was 1.7500% for DeepRWCap, 1.7297% for CAPR and 1.6228% for CARE. The five-bound post-hoc archive supports the stated ≤1.26% neighborhood span; it does not establish the historical main hyperparameter selection process.

A fresh full neural installation, complete retraining/export, other GPUs and dedicated memory studies remain unvalidated. CPU tests also passed previously in a separate NumPy-only environment. Full traditional-baseline batches were not rerun for release checks; earlier AGF/MicroWalk single-case checks completed, while an earlier FDM check stopped before completion. No Docker workflow is supplied.

## Available experiments and release scope

| Manuscript experiment | Released workflow | Requirements or boundary |
|---|---|---|
| Table I: public10 | `benchmark.py --profile paper`; optional `baselines.py` | Ten inputs/references included; three neural arms. |
| Table II: CPGR; Table III: CAPR | `evaluate_transitions.py` | Original external reference datasets required; see [local evaluation](DATA.md#local-transition-evaluation). |
| Table IV: components | Three solver arms, raw/CER from each | Six observed readouts; no standalone DeepRWCap+CPGR or complete factorial interaction. |
| Table V: time/workload | `run.py`, `summarize.py` | Solver time, walks, weighted hops; no memory/latency study. |
| Table VI: additional layouts | [Six candidates, screening and five configs](../benchmarks/additional/README.md) | Ten seeds per arm, both use CER; original references included; no dedicated Table-VI bootstrap/figure entry. |
| Training seeds | `summarize_training_seeds.py` | Reaggregates saved observations; extra models/inference driver not bundled. |
| Coupling NL1 | `evaluate_readout.py --coupling-reference` | Saved output plus bundled DSPF/SPICE reference. |
| Visited activation | CPGR counters | No dedicated trajectory-aggregation entry. |
| τ sensitivity | `capr_training_recipe.json` | Settings only; extra training driver/data/checkpoints not bundled. |

The release includes core refinement source, 18 model files, inputs/references, configurations and tests. Solver cores and CPU baselines are upstream binary dependencies. Each invocation evaluates one master; the overview depicts repetition across masters. Full training data/driver, all research logs and paper-figure scripts are outside this release. GGFT generates new data rather than recreating archived labels. See [model interfaces](../models/README.md#file-compatibility) for migration from the removed legacy imports/commands.
