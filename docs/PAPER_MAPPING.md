# Paper framework and released implementation

CARE-RWCap is **Condition-Aware Refinement for Neural-Guided Floating Random Walk Capacitance Extraction**. The [overview figure](figures/overview.png) shows the physical problem, transition refinements and same-solve readout. PNG and vector PDF versions are included.

## Follow the overview

| Stage in the figure | Released implementation | What happens |
|---|---|---|
| Physical input | [public10](../benchmarks/public10/README.md), [protocol](../configs/paper_protocol.json) | Read conductor/dielectric geometry, a selected master, convergence parameters and the local normalized dielectric tensor. |
| First neural Gradient transition: CPGR | [CUDA projection](../cpp/cpgr/projection.cu), [selector](../cpp/cpgr/selector.cuh), [sampler](../cpp/cpgr/sampler.cpp) | Check strict input reflection, average eligible opposite-face probabilities, project the signed in-face kernel and compensate its sampling contribution. |
| Subsequent neural Poisson transitions: BPR | [model](../src/bpr/model.py), [loss](../src/bpr/loss.py), [models](../models/README.md) | Keep the Poisson selector and frozen anchor; use bounded residual reweighting for the in-face conditional distribution. |
| Conductor hit and accumulation | Bundled upstream solver runtime | Accumulate signed terminal-conductor contributions within the existing solve. |
| Conditional self-capacitance readout: CER | [frozen parser/readout](../src/readout/frozen.py), [evaluation](../scripts/evaluate.py) | Check the logical output row and select the coupling-magnitude sum or the raw self-capacitance. |

The algorithmic diagram shows repetition for each master and a resulting capacitance matrix. The released Python evaluator handles **one configured master per invocation**. It retains that solver output row and reports raw/CER self estimates; it is not a multi-master matrix-assembly command.

## Scope of “unchanged”

BPR preserves the **Poisson** face selector. CPGR keeps the Gradient network parameters frozen but can change **Gradient face probabilities** by averaging eligible opposite-face pairs. It also projects the signed conditional kernel and applies the L1 norm ratio. The seventh selector output, the global weight, is preserved. These distinctions are part of the deployed method.

The upstream selection of neural/non-neural paths, scheduling policy and stopping configuration is retained. Changing a sampled transition can still change the realized trajectory, batch composition and stopping time.

## CER input and fallback contract

The released parser expects the upstream solver to provide one already-aggregated entry per logical conductor. It canonicalizes names such as `3__netA` to `netA`; it **does not add duplicate physical-fragment columns**. Repeated logical columns or master blocks are rejected.

For a valid row, CER requires the complete expected column set, at least one off-diagonal entry, nonpositive couplings and a positive coupling-magnitude sum. If these applicability checks fail, it retains the positive raw self-capacitance. Nonfinite output, a missing/nonpositive raw master self term or ambiguous duplicate records are rejected as invalid input, rather than converted into a valid fallback measurement. No reference capacitance is used to decide applicability.

The figure's “logical-conductor output formation” precedes CER and belongs to the existing solver-output pipeline. See [METHOD.md](METHOD.md) for the exact rule.

## Experiment coverage

| Experiment category | Available entry/material | Boundary |
|---|---|---|
| Public10 self-capacitance comparison | [three-arm runner](../benchmarks/public10/README.md), bundled layouts/references and [historical reference](../results/README.md) | 300 measured solves plus three warmups; not a fixed accuracy acceptance target. |
| Traditional FRW methods | [CPU baseline runner](CPU_BASELINES.md) | New raw SelfCapErr and solver workload/time measurements; no bundled historical CPU result table. |
| Local Poisson/Gradient validation | [optional evaluator](LOCAL_VALIDATION.md) | Requires original external reference datasets. Active-only Gradient statistics use a different denominator from solver-visited activation. |
| Module comparisons | Raw/CER readouts for `deeprwcap`, `bpr`, `care-rwcap` | Measures CER effects and CPGR conditional on BPR. No standalone DeepRWCap+CPGR arm or complete factorial interaction. |
| Runtime/workload | Solver elapsed/CPU time, walks, weighted hops and approximate steps | No peak-memory measurement or CER latency microbenchmark in this runner. |
| Coupling-row error and additional layouts | Solver output row is retained; a custom single-master config is accepted | Coupling-row normalized L1 reporting and the five-layout data/protocol are not supplied as ready-to-run paper workflows. |

## Terminology and training

Historical identifiers remain only where needed for file/interface compatibility: **S29 RF_RISK = BPR**, **Gradient-Joint = CPGR**, and **strict-S24 = CER**.

The frozen BPR recipe uses AdamW, learning rate 3e-4 and trust coefficient 2; validation selects epoch 25. Its full recipe and the distinction between training loss and validation score are in [METHOD.md](METHOD.md). The DeepRWCap is a locally trained DeepRWCap architecture, as documented in [BASELINE.md](BASELINE.md). CPGR and CER require no additional trained model.

## Frozen-version consistency

Use the [implementation contract](IMPLEMENTATION_CONTRACT.md) when describing the frozen method: distinguish unchanged Poisson probabilities from post-processed Gradient probabilities, and upstream logical-row output from CER's conditional readout. The [training recipe](../configs/bpr_training_recipe.json) records AdamW, 3e-4 and trust coefficient 2. The [training-seed supplement](../results/training_seeds/README.md) is an independent Full evaluation with a reused historical DeepRWCap baseline; it does not replace the original main table.
