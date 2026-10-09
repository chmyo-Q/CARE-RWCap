# Frozen local transition validation

This optional entry evaluates the deployed CAPR and CPGR operators without training, new labels or end-to-end solves. Use the same supported Linux / RTX 4090 environment and build the extension first.

```bash
python scripts/evaluate_transitions.py --help
python scripts/evaluate_transitions.py --data-dir /path/to/original_data \
  --module all --preflight-only --output runs/local_preflight
python scripts/evaluate_transitions.py --data-dir /path/to/original_data \
  --module all --output runs/local_validation
```

The data directory must contain the original `poisson.bin` and `gradient.bin`. Each contains 100,000 records and occupies 12,362,400,016 bytes. These reference datasets are **not included**, and this release currently supplies no public download endpoint for them. Consequently, a new reader can execute public10 end-to-end tests using the package alone, but cannot reproduce these local validation tables from the package alone. The optional [GGFT workflow](DATA.md) generates new data, not the frozen reference labels. The evaluator rejects a different dataset rather than labeling it a reproduction.

The exact archived Poisson validation order is in `configs/transition_validation.json`: the last 10,000 indices of a NumPy PCG64 permutation of 100,000 records with seed 20260805. Gradient uses its separate fixed indices `[90000,100000)`. Gradient1 is evaluated on canonical face 1 and Gradient2 on canonical face 2; the two head evaluations of a record are counted separately. These are not six independent validation samples per record. There is no split or threshold search. The evaluator uses frozen FP16 engines, FP32 production CUDA projection and FP64 metrics. It records its actual environment and preserves failed outputs. A preflight success only confirms the datasets and split; it does not validate GPU execution.

Use `--module capr` for Poisson-only evaluation; `--module bpr` remains a compatibility alias. New Poisson sample files are named `capr_samples.csv.gz`. To preserve existing result readers, `summary.json` and CSV column names retain the historical `bpr` / `bpr_*` metric keys for CAPR; these are storage names for the same component.

Outputs include `protocol.json`, `status.json`, `summary.json`, `summary.md` and paired sample records in compressed CSV. CAPR reports KL (including mean, Q95 and Q99), six-probe Action MSE and TV(CAPR,DeepRWCap), using the normalized probabilities actually sampled from the FP16 outputs. TV is refinement magnitude, not error against the reference. CPGR reports Gradient1/2 activation counts, active-only normalized L2, parity violations, exact-label parity and improvement fractions. The weight check uses one measure: normalized L1 error of effective six-face absolute mass. The seventh selector output is unchanged; this is not a claim that its neural prediction improves.

The Markdown summary includes mean/Q95/Q99 KL, Action MSE, relative changes and TV for CAPR; strict-active per-head and pooled NL2, improvement percentages and reference/raw/refined parity for CPGR; and the single effective-face-mass NL1 measure for weight compensation. The pooled NL2 uses active-count weighting (473 and 476 in the archived reference), not an equal-weight average of the two head means. A negative relative change denotes a reduction; the paper's reduction column uses the opposite sign.

The numerical functions preserve the frozen evaluator's formulas, label transformations and active sets. The renamed repository-relative entry passed CPU preflight/structural checks and GPU evaluation on the original reference files; see [VALIDATION.md](VALIDATION.md). Local metrics measure transition behavior and do not establish universal end-to-end improvement.

The strict-active denominator here is the fixed canonical validation set for each Gradient head. It is not the number of solver-visited Gradient calls. The local evaluator does not generate the paper's trajectory-activation or additional-layout statistics.
