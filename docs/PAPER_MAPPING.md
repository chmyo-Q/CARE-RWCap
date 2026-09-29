# Paper terms and implementation

The project name is **CARE-RWCap**. Historical experiment identifiers are retained only where needed to identify frozen code or model files.

| Paper term | Historical identifier | Implementation / evaluation role |
|---|---|---|
| Baseline-Anchored Poisson Refinement (BPR) | S29 RF_RISK | `src/bpr/model.py`, `loss.py`; replaces the Poisson predictor only. |
| Conditional Parity Gradient Refinement (CPGR) | Gradient-Joint, CPGR-Joint | `cpp/cpgr/`; eligible signed-kernel projection **and opposite-face probability averaging**, followed by sampling compensation. |
| Conditional Endpoint Re-estimation (CER) | strict-S24 | `src/readout/frozen.py`; applies the frozen strict endpoint rule after the solve. |
| Paper P0 | locally trained DeepRWCap baseline | `models/paper_p0/`; not the official upstream pretrained engine set. |

The `full` arm means BPR + CPGR in the solver. Its selected paper endpoint is CER/strict-S24. Every arm also retains the raw endpoint and its CER result, including P0. CPGR has no extra trained network.

## Important implementation details

**Gradient selector.** The gradient networks remain frozen. However, CPGR-Joint can change their face-selection probabilities at runtime: it averages eligible opposite-face pairs under exact input reflection symmetry. The seventh selector output (global weight) is preserved. “Unchanged network weights” must not be described as “unchanged selection probabilities.”

**Logical conductor rows.** The bundled solver must already output one entry per logical conductor. The parser normalizes names such as `3__netA` to `netA`; it does not combine separate physical-fragment estimates by summing them. Ambiguous repeated logical columns or repeated master blocks are rejected. Under this unique-logical-row input contract, CER uses the sum of absolute off-diagonal couplings when its completeness/sign conditions hold, and otherwise falls back to raw self capacitance. A general physical-fragment aggregation algorithm is outside the released implementation.

**Error comparisons.** The historical main table uses P0 raw versus BPR+CER versus BPR+CPGR+CER. This is a comparison of complete methods. The released protocol measures the conditional CPGR effect by comparing Full to BPR under the same readout. It does not identify CPGR's standalone effect on P0 or a full factorial interaction. A shared seed does not guarantee shared random-walk trajectories.

These correspondences describe the code being released. They do not establish statistical significance or a universal end-to-end gain from local parity constraints. See [METHOD.md](METHOD.md) for equations and conditions.

## Training recipe and evaluation coverage

The frozen BPR adapter uses AdamW, initial learning rate 3e-4 and trust coefficient 2. Its validation selection rule uses separate KL/action percentiles and is not the training mini-batch top-k expression. [METHOD.md](METHOD.md) is the implementation reference; upstream training defaults must not replace these final values.

The runner covers public10 extraction and same-batch attribution. Local tables require the original external datasets through [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md). Traditional CPU solver binaries and a separate [baseline runner](CPU_BASELINES.md) are included. Additional-layout inputs, peak memory and CER microtiming are outside the bundled reproduction. See the explicit [capability map](ARTIFACT_SCOPE.md).
