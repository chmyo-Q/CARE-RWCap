# Environment and evaluation

For concrete installation commands, see [INSTALL.md](INSTALL.md). For the status of verification, see [VALIDATION.md](VALIDATION.md).

## Software environment

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

Use a Python 3.12 environment, install `requirements.txt`, and run `bash scripts/build.sh`. The build script discovers Torch headers and libraries from that active Python environment. It compiles for SM 89 by default. Changing `--arch` alone does **not** make the frozen TensorRT engines portable to another GPU or TensorRT version.

The driver-reported CUDA compatibility level (12.8 on the tested machine) is not the installed toolkit version (12.6). Deployment engines are TensorRT FP16, whereas the supplied uncompiled TorchScript and state dictionaries are FP32. Do not silently replace a failed engine with a different model. Full engine recompilation and other GPU targets are not validated in this release.

## Commands and paths

Run commands from the repository root. Output directories are created fresh and existing directories are rejected. Input/model paths are resolved from the repository rather than an author-specific filesystem. `models.txt`, required by the upstream solver, is generated inside each run directory.

The runner controls library search paths and clears inherited method-switch variables. It preserves an explicitly set `CUDA_VISIBLE_DEVICES`; otherwise device 0 is used. Default `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are separate from the solver's eight-worker setting.

If a shared library fails to load, inspect `build/compile_*.log` or the run's `console.log`, then check the environment above. A successful CPGR-on run produces projection and selector counters; the smoke test checks that eligible constraints actually activated and that CUDA reported success.

## Seed and repetition

The example uses initial seed **2029**, with solver arguments `-n 8 -p 0.01 -c 0.01 --c-ratio 0.3`. `cpp/seed_control/seed_control.cpp` initializes Torch CPU/CUDA generators and the exported solver-core seed before the main program. Each run records whether that initialization occurred.

The aligned 300-run public protocol uses ten seed blocks, **2029–2038**, with the same seed assigned to every case and method within a block. This is distinct from the training seed (2029) and from the single-run example default (2029). Actual run records and the frozen protocol agree on these ten seeds. Other public-case ratios and reference values are documented in `configs/paper_protocol.json`; the batch runner consumes this protocol and creates individual run configurations. The single-case runner uses `configs/example.json` by default.

Fixed initial states do not control asynchronous thread scheduling, inference-batch composition, or assignment of random draws to paths. Consequently, repeated end-to-end runs need not be bitwise equal. This release provides no empirically established tolerance within which a new batch is guaranteed to match a historical aggregate. Functional acceptance cannot establish the stability or statistical significance of a reported gain.

When comparing methods, keep model set, solver settings, inputs, reference values, and readout definition consistent. Use a predetermined repeat count and order; retain all runs in the comparison. Compare repeated observations rather than accepting or rejecting a result based on a single solve. Claim runtime improvement only with an appropriately controlled timing comparison.

## Metrics

- SelfCapErr (%) = `100 * abs(C / C_ref - 1)`; report raw and strict-S24 separately.
- The aligned three-arm table used P0 raw, BPR + strict-S24, and BPR + CPGR + strict-S24. Its mean is the unweighted mean of the ten per-case mean errors (each case has ten repetitions). Case-to-case standard deviation and within-case repeat variation are different statistics and must be labeled separately.
- Walks and hops/walk come from the solver's output. Approximate steps are their product, rounded to an integer; hops/walk is already rounded by the solver.
- Elapsed and CPU seconds are values reported by the solver. They are not measurements of Python launch, model loading, or the complete wrapper wall time.
- The minimal evaluator retains the selected-master capacitance row but does not compute coupling-row normalized L1 error.
- The minimal evaluator does not measure GPU memory or S24 microbenchmark latency; these require dedicated protocols and are not inferred from solver timing.

The bundled example reference is the frozen total self-capacitance for master `1` from public `case8.dspf`. It is used only for error computation. Parser fixtures under `tests/fixtures/` are synthetic readout inputs, not physical benchmark geometries.

## Scope of the checks

`scripts/smoke_test.py` runs the unit/numerical checks, sampler integration, and one P0 and one Full solve on case8. It saves a local summary under the requested output directory. These are functional checks in the stated GPU environment; they do not reproduce an entire paper table or validate retraining. Raw checkpoints are included for inspection and model loading, but full training data, a full training driver, and all-architecture export/compilation tooling are not provided.

## Public10 batch evaluation

The [batch runner](../benchmarks/public10/README.md) supplies all ten geometries and references. It reproduces the original three-arm planned execution order, with three excluded case8 warmups at seed 2040. `--plan-only` needs no GPU. Summaries include within-case sample standard deviation (`ddof=1`), between-case standard deviation of case means and seed-block differences; these quantities are labeled separately. There is no automatic significance test. Raw-to-raw and CER-to-CER differences are available alongside the historical whole-method endpoint comparison.

The public10 benchmark was used during method development. It is a fixed benchmark, not an unseen generalization set. Historical reference aggregates identify a specific batch, do not incorporate every exploratory batch and are not a guarantee that another batch has the same ordering. Changed model/configuration/seed protocols constitute a separate experiment.

## Component attribution and record checks

The current summary reparses each measured `result.out` and checks saved raw/CER values, errors, matrix, workload and times. It also rejects duplicate/incomplete planned case-seed-arm grids. A missing or inconsistent measured record prevents a complete-table aggregate. The solver runner checks requested seed initialization and CPGR CUDA/counter consistency; zero Gradient visits remain valid for cases that do not use that path.

For each initial seed, error differences are averaged equally over the fixed cases. `statistics_report.py` uses NumPy PCG64 seed 20260919 and 20,000 percentile bootstrap resamples of those blocks. With one seed, the interval is omitted. All intervals are pointwise and unadjusted for multiple comparisons. Results do not imply shared random paths or unseen-case generalization.

The original three comparisons consumed successive RNG blocks. To reproduce their interval endpoints without depending on display order, the implementation records explicit stream indices: BPR-CER versus P0-raw uses 0, Full-CER versus P0-raw uses 1, and Full-CER versus BPR-CER uses 2. Other component contrasts use block 3. Each block consists of `draws x number_of_seeds` sampled indices. Merely resetting seed 20260919 for every contrast gives slightly different Monte Carlo percentile endpoints; it is not a change in observed capacitance.

Raw/CER are dependent readouts of the same solve. The released three-arm protocol measures the CPGR effect conditional on BPR, not its standalone effect or the complete BPR-by-CPGR interaction. The [results reference](../results/README.md) belongs to the original 300-run cohort; keep new batches separate and do not transport percentage gains between cohorts to create missing observations.

For the traditional CPU programs, use the separate [baseline instructions](CPU_BASELINES.md). They do not load neural engines or use CER.
