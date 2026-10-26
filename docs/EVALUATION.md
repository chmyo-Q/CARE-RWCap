# Evaluation

[Installation](INSTALL.md) · [Public10 commands](../benchmarks/public10/README.md) · [Historical results](../results/README.md)

## Commands and paths

Run commands from the repository root. Output directories are created fresh and existing directories are rejected. Input/model paths are resolved from the repository rather than an author-specific filesystem. `models.txt`, required by the upstream solver, is generated inside each run directory.

The runner controls library search paths and clears inherited method-switch variables. It preserves an explicitly set `CUDA_VISIBLE_DEVICES`; otherwise device 0 is used. Default `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are separate from the solver's eight-worker setting.

If a shared library fails to load, inspect `build/compile_*.log` or the run's `console.log`, then check the [deployment environment](INSTALL.md#deployment-environment). A successful CPGR-on run produces projection and selector counters; the smoke test checks that eligible constraints actually activated and that CUDA reported success.

## Seed and repetition

The default case1 example uses initial seed **2029**, with solver arguments `-n 8 -p 0.01 -c 0.01 --c-ratio 0.95`, matching its public10 configuration. The separate case7 smoke configuration uses `--c-ratio 0.3`. `cpp/seed_control/seed_control.cpp` initializes Torch CPU/CUDA generators and the exported solver-core seed before the main program. Each run records whether that initialization occurred.

The aligned 300-run public protocol uses ten seed blocks, **2029–2038**, with the same seed assigned to every case and method within a block. This is distinct from the training seed (2029) and from the single-run example default (2029). Actual run records and the frozen protocol agree on these ten seeds. Other public-case ratios and reference values are documented in `configs/paper_protocol.json`; the batch runner consumes this protocol and creates individual run configurations. The single-case runner uses `configs/example.json` by default.

Fixed initial states do not control asynchronous thread scheduling, inference-batch composition, or assignment of random draws to paths. Consequently, repeated end-to-end runs need not be bitwise equal. This release provides no empirically established tolerance within which a new batch is guaranteed to match a historical aggregate. Functional acceptance cannot establish the stability or statistical significance of a reported gain.

When comparing methods, keep model set, solver settings, inputs, reference values, and readout definition consistent. Use a predetermined repeat count and order; retain all runs in the comparison. Compare repeated observations rather than accepting or rejecting a result based on a single solve. Claim runtime improvement only with an appropriately controlled timing comparison.

## Metrics

- SelfCapErr (%) = `100 * abs(C / C_ref - 1)`; report raw and CER separately.
- The aligned three-arm table used DeepRWCap raw, CAPR + CER, and CAPR + CPGR + CER. Its mean is the unweighted mean of the ten per-case mean errors (each case has ten repetitions). Case-to-case standard deviation and within-case repeat variation are different statistics and must be labeled separately.
- Walks and hops/walk come from the solver's output. Approximate steps are their product, rounded to an integer; hops/walk is already rounded by the solver.
- Elapsed and CPU seconds are values reported by the solver. They are not measurements of Python launch, model loading, or the complete wrapper wall time.
- Coupling-magnitude NL1 (%) = `100 * sum(abs(abs(pred_j)-abs(ref_j))) / sum(abs(ref_j))`, over off-diagonal entries shared by the selected-master output and its reference row. Self is excluded. The optional readout flag below computes this metric without another solve. Average run values within each benchmark, then equally across benchmarks, as in v7 IV-D.
- The minimal evaluator does not measure GPU memory or CER microbenchmark latency; these require dedicated protocols and are not inferred from solver timing.

The default example reference is the frozen total self-capacitance for master `APpolysub` from public `case1.dspf` (`1.591776e-14 F`). The separate smoke configuration retains master `1` and its public case7 reference. References are used only for error computation. Parser fixtures under `tests/fixtures/` are synthetic readout inputs, not physical benchmark geometries.

## Optional coupling-entry evaluation

After a case1 solve, reuse its saved row with the same reference file:

```bash
python scripts/evaluate_readout.py --geometry benchmarks/public10/case1.cap3d \
  --output runs/full_case1/result.out --reference 1.591776e-14 \
  --coupling-reference benchmarks/public10/ref/case1.dspf \
  --json runs/full_case1/coupling_metrics.json
```

`coupling_error.normalized_l1_percent` uses v7's reference-covered off-diagonal magnitude metric. The JSON also records the number of shared entries and reference mass. This option does not change raw/CER estimates or applicability. DSPF/SPICE reference parsing follows the upstream benchmark conventions. Missing reference masters or zero shared reference mass are rejected rather than reported as zero error. Default readout behavior is unchanged when the flag is absent.

## Smoke workflow

`scripts/smoke_test.py` runs the unit/numerical checks, sampler integration, and one DeepRWCap and one CARE-RWCap solve using `configs/smoke_case7.json`. It explicitly checks case7, CPGR activation and CER application independently of the default single-run configuration. It saves a local summary under the requested output directory. These are functional checks in the stated GPU environment; they do not reproduce an entire paper table or validate retraining. Raw checkpoints are included for inspection and model loading, but full training data, a full training driver, and all-architecture export/compilation tooling are not provided.

## Public10 batch evaluation

The [batch runner](../benchmarks/public10/README.md) supplies all ten geometries and references. It reproduces the original three-arm planned execution order, with three excluded case8 warmups at seed 2040. `--plan-only` needs no GPU. Summaries include within-case sample standard deviation (`ddof=1`), between-case standard deviation of case means and seed-block differences; these quantities are labeled separately. There is no automatic significance test. Raw-to-raw and CER-to-CER differences are available alongside the historical whole-method endpoint comparison.

The public10 benchmark was used during method development. It is a fixed benchmark, not an unseen generalization set. Historical reference aggregates identify a specific batch, do not incorporate every exploratory batch and are not a guarantee that another batch has the same ordering. Changed model/configuration/seed protocols constitute a separate experiment.

## Component attribution and record checks

The current summary reparses each measured `result.out` and checks saved raw/CER values, errors, matrix, workload and times. It also rejects duplicate/incomplete planned case-seed-arm grids. A missing or inconsistent measured record prevents a complete-table aggregate. The solver runner checks requested seed initialization and CPGR CUDA/counter consistency; zero Gradient visits remain valid for cases that do not use that path.

For each initial seed, error differences are averaged equally over the fixed cases. `statistics_report.py` uses NumPy PCG64 seed 20260919 and 20,000 percentile bootstrap resamples of those blocks. With one seed, the interval is omitted. All intervals are pointwise and unadjusted for multiple comparisons. Results do not imply shared random paths or unseen-case generalization.

The original three comparisons consumed successive RNG blocks. To reproduce their interval endpoints without depending on display order, the implementation records explicit stream indices: CAPR-CER versus DeepRWCap-raw uses 0, CARE-RWCap versus DeepRWCap-raw uses 1, and CARE-RWCap versus CAPR-CER uses 2. Other component contrasts use block 3. Each block consists of `draws x number_of_seeds` sampled indices. Merely resetting seed 20260919 for every contrast gives slightly different Monte Carlo percentile endpoints; it is not a change in observed capacitance.

Raw/CER are dependent readouts of the same solve. The released three-arm protocol measures the CPGR effect conditional on CAPR, not its standalone effect or the complete CAPR-by-CPGR interaction. The [results reference](../results/README.md) belongs to the original 300-run cohort; keep new batches separate and do not transport percentage gains between cohorts to create missing observations.

For the traditional CPU programs, use the separate [baseline instructions](CPU_BASELINES.md). They do not load neural engines or use CER.

## CAPR training-seed supplement

[Archived records and reaggregation](../results/training_seeds/README.md) cover training seeds 2029, 2039 and 2053 with 100 new CARE-RWCap solves per model. They reuse 100 DeepRWCap solves from the original aligned batch, matched by case and initial solver seed. This is not a simultaneous control experiment or common-path pairing. The script verifies coverage, matching references and saved errors, and recomputes all summaries from capacitance values. It does not perform new inference. The main inference runner and checkpoints remain unchanged.

## Validation status

Checked on 2026-10-11 using an independent source checkout in the supported Ubuntu 24.04 / RTX 4090 neural environment. The documentation was subsequently consolidated; the solver, models, numerical code and settings were not changed.

| Workflow | Completed check |
|---|---|
| Build/environment | Python and Bash builds, prerequisite/runtime checks and installed dependency consistency passed. |
| Tests/models | 37 CPU tests passed in the neural environment and a separate NumPy-only environment. CAPR source/uncompiled TorchScript maximum difference was 0; the anchor matched the bundled baseline. |
| CPGR | All reflection masks, selector averaging, compensation, zero projection and sampler integration passed; asymmetric inputs retained upstream behavior. |
| Examples | Default case1 Python/Bash commands, case7 smoke and all nine case1 quick solves completed. Smoke verified CPGR activation and CER application. |
| Readout/plans | Current/compatible readout commands agreed; paper plans covered 300+3, CARE-only 100+1 and CAPR/CARE 200+2 solves. |
| CARE public10 | 100 formal solves and one excluded warmup completed; all result/configuration checks passed. |
| Historical records | Current readers reparsed the original 300, additional-layout 100 and training-seed 300 outputs. All original main capacitances, errors, workload and solver times matched. |
| Local FP16 | 203 numerical summary fields matched the frozen evaluation; Gradient1/2 active counts were 473/476. CPU reaggregation of 40,000 paired observations matched 198 summary fields. |
| Data generation | Pinned GGFT source rebuilt and generated two records per kind; headers and record counts passed. |

The new CARE-only macro error was **0.9053%**, compared with **0.8110%** in the historical cohort. In the new batch raw error was **0.8507%**; CER increased it by 0.0546 pp (pointwise 95% seed-block interval [0.0117, 0.0994] pp). No new DeepRWCap or CAPR-only control cohort was run, so this check cannot re-estimate paired method gains. CPGR's incremental end-to-end benefit has varied across checked batches. Functional acceptance does not guarantee the historical mean or ranking.

Full traditional-baseline batches were not rerun. Earlier AGF/MicroWalk single-case checks completed; an earlier FDM solve was stopped before completion. A fresh full neural installation, complete retraining/export, other GPUs and dedicated peak-memory studies remain unvalidated. The earlier unvalidated Docker recipe is no longer supplied. The separate NumPy-only environment checks CPU workflows, not a fresh neural installation.

## Available experiments and release scope

The package supports frozen-model inference and public10 evaluation. Architecture/loss, CPGR source, CER, all 18 model files, inputs/references, configs and tests are included; the upstream solver core and CPU baselines are binary dependencies. The figure shows repetition over masters, while each released invocation evaluates one selected master.

| Manuscript experiment | Released workflow | Additional requirements or boundary |
|---|---|---|
| Table I: public10 SelfCapErr | `benchmark.py --profile paper`; optional `baselines.py` | All ten inputs/references included. Historical compact results contain the three neural arms. |
| Table II: local CPGR; Table III: local CAPR | `evaluate_transitions.py` | Original external reference datasets required; see [local evaluation](DATA.md#local-transition-evaluation). |
| Table IV: component comparisons | Three neural arms with raw/CER from each solve | Six observed configurations; no standalone DeepRWCap+CPGR arm or complete factorial interaction. |
| Table V: time/workload | `run.py`, `summarize.py` | Solver time, walks and weighted hops; no peak-memory or CER-latency study. |
| Table VI: additional layouts | [Six candidates, screening and five frozen configs](../benchmarks/additional/README.md) | Ten seeds per arm; CER for both arms. Original two-tolerance references are included; historical full logs and a dedicated Table-VI aggregation entry are not. |
| Training-seed study | `summarize_training_seeds.py` | Saved observations included; the two extra models and multi-model inference driver are not. |
| Coupling-entry NL1 | `evaluate_readout.py --coupling-reference` | Reads a saved output and bundled DSPF/SPICE reference; no additional solver run. |
| Visited activation | CPGR counters retained | Specialized trajectory aggregation is not a separate release entry. |
| τ sensitivity | `capr_training_recipe.json` | Five-bound post-hoc settings are recorded; retraining driver/data and extra checkpoints are not bundled. |

Full training data/driver, all research logs and paper-figure scripts are outside this release. GGFT generates new data and does not recreate archived labels. The overview, inspectable checkpoints, CPU baselines, saved results, local evaluator and tests support optional research or inspection; only the runtime/deployment models, own extensions, layout and runner/config are needed for a single solve. Reference capacitance is needed to report error, not to estimate capacitance.

## Manuscript v4 release update

On 2026-10-16, the six additional-layout inputs, two-tolerance reference screening and five retained-layout execution configurations were recovered from the original study. The current reader reproduced all 100 archived outputs and the Table-VI means/SDs. The documented 100+2 command schedule was checked against the original run order. This update did not retrain models, regenerate references or execute new end-to-end solves.

The case7 smoke configuration now reads the public10 input and reference directly. Its redundant copies were removed after byte-for-byte identity checks; case, seed, master, reference capacitance, solver settings and both operators are unchanged.

## Manuscript v7 alignment

Updated 2026-10-26: method order and formula references now follow CPGR (6–14), CAPR (15–24), CER (25–27), and metrics (28–30); the author-supplied overview matches v7 Fig. 1. The reference-magnitude coupling reader was checked on all 300 original outputs: DeepRWCap 1.7500%, CAPR 1.7297%, CARE-RWCap 1.6228%. Reading these saved results is not a new solver experiment.

All 40 current CPU tests, 13 CLI help entries, source/checkpoint loading and the new coupling command passed. The earlier Linux/RTX 4090 neural checks remain the GPU verification record; this documentation and optional readout update did not run new GPU solves or retrain any model.

The five-bound post-hoc validation archive supports the stated 1.26% maximum neighborhood span. It does not retroactively establish how the original main hyperparameter was chosen. The three-arm archive still supplies no standalone DeepRWCap+CPGR arm; the released observed-configurations table does not substitute a cross-cohort estimate for that missing observation.
