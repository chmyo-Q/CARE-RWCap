# Frozen local transition validation

This optional entry evaluates the deployed BPR and CPGR operators without training, new labels or end-to-end solves. Use the same supported Linux / RTX 4090 environment and build the extension first.

```bash
python scripts/local_validation.py --help
python scripts/local_validation.py --data-dir /path/to/original_data \
  --module all --preflight-only --output runs/local_preflight
python scripts/local_validation.py --data-dir /path/to/original_data \
  --module all --output runs/local_validation
```

The data directory must contain the original `poisson.bin` and `gradient.bin`. Each contains 100,000 records and occupies 12,362,400,016 bytes. These reference datasets are **not included**, and this release currently supplies no public download endpoint for them. Consequently, a new reader can execute public10 end-to-end tests using the package alone, but cannot reproduce these local validation tables from the package alone. Upstream data generation produces new data, not the frozen reference labels. The evaluator rejects a different dataset rather than labeling it a reproduction.

The exact archived Poisson validation order is in `configs/transition_validation.json`; Gradient uses indices `[90000,100000)`. There is no split or threshold search. The evaluator uses frozen FP16 engines, FP32 production CUDA projection and FP64 metrics. It records its actual environment and preserves failed outputs. A preflight success only confirms the datasets and split; it does not validate GPU execution.

Outputs include `protocol.json`, `status.json`, `summary.json`, `summary.md` and paired sample records in compressed CSV. BPR reports KL (including mean, Q95 and Q99), six-probe Action MSE and TV(BPR,DeepRWCap), using the normalized probabilities actually sampled from the FP16 outputs. TV is refinement magnitude, not error against the reference. CPGR reports Gradient1/2 activation counts, active-only normalized L2, parity violations, exact-label parity and improvement fractions. The weight check uses one measure: normalized L1 error of effective six-face absolute mass. The seventh selector output is unchanged; this is not a claim that its neural prediction improves.

The numerical functions were ported from the verified frozen evaluator without changing metric formulas, label transformations or active sets. The new repository-relative entry has CPU preflight/structural checks; see [VALIDATION.md](VALIDATION.md) for whether this packaged GPU entry has been run. No local metric implies universal end-to-end improvement.

The strict-active denominator here is the fixed canonical validation set for each Gradient head. It is not the number of solver-visited Gradient calls. The local evaluator does not generate the paper's trajectory-activation or additional-layout statistics.
