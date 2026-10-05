# Frozen implementation contract

The implementation below is the one associated with the original aligned public10 batch (2026-09-19). The audit used its archived configuration, deployed files and solver outputs, rather than selecting a recipe from a manuscript draft. Historical names are S29 RF_RISK = BPR, Gradient-Joint = CPGR, strict-S24 = CER.

## BPR training and checkpoint lineage

| Setting | Frozen recipe | Evidence type |
|---|---|---|
| Optimizer / initial learning rate | AdamW / 3e-4 | Archived training launcher and implementation |
| Weight decay / schedule | 1e-6 / cosine annealing over 30 epochs to 5e-6 | Archived training implementation |
| Action / trust / risk coefficient | 10 / 2 / 0.5 | Original training protocol and implementation |
| Tail fraction / maximum log dose | 0.25 / 0.125 | Original protocol |
| Batch size / gradient clipping | 16 / norm 1 | Launcher / implementation |
| Training seed / selected epoch | 2029 / 25 | Original protocol / completed RF_RISK run metadata |
| Split | 90,000 training + 10,000 validation; seed 20260805 | Fixed split manifest used by the training launcher |

The [machine-readable recipe](../configs/bpr_training_recipe.json) distinguishes the training tail objective from the validation selection score. Adam at 1e-3 and trust coefficient 0.25 do not describe this checkpoint. AdamW refers to [decoupled weight decay](https://arxiv.org/abs/1711.05101).

The source is the RF_RISK checkpoint selected in `train_seed2029_30e_20260909_180652`. Its released state dictionary and uncompiled TorchScript agree with the preserved training-seed artifact. The BPR FP16 engine, all five DeepRWCap engines, and CPGR source agree with the original aligned-batch inventory (one C++ file differs only in line endings). This identifies the deployed files; it is not a new training or engine-recompilation experiment. The original training protocol JSON records loss coefficients but does not itself record optimizer/lr. Those settings come from the archived launcher and implementation; an optimizer-state replay is not claimed.

## CPGR changes the Gradient face distribution

The Poisson selector is unchanged by BPR. The Gradient selector **network parameters** are also unchanged, but CPGR post-processes its first six outputs before sampling a face. For each strictly symmetric input axis, replace the corresponding opposite pair `(q_a, q_b)` by `((q_a+q_b)/2, (q_a+q_b)/2)`.

| Reflected input index after Gradient-axis rotation | Exact-equality mask bit | Zero-based face pair |
|---|---:|---|
| x (fastest index) | 1 | 4, 5 |
| y | 2 | 2, 3 |
| z | 4 | 0, 1 |

See [selector.cuh](../cpp/cpgr/selector.cuh) and the call to `joint_selector_launch` **before** `torch::multinomial(faceProbabilities,...)` in [sampler.cpp](../cpp/cpgr/sampler.cpp). Each pair sum and the sum of all six entries are preserved up to floating-point roundoff; inactive pairs and the seventh output (global weight) are copied unchanged. Eligible averaging can change individual face probabilities.

After the face is drawn from this corrected distribution, the conditional signed kernel is projected and its L1 norm ratio compensates in-face sampling as described in [METHOD.md](METHOD.md). The conditional expectation identity does not assert that the original joint face-and-position transition law is preserved. No original-to-corrected face-probability importance ratio is applied. Both `S29_GRADIENT_JOINT_ENABLE=1` and `S29_GRADIENT_PARITY_ENABLE=1` were enabled in the archived Full arm and are enabled by the public runner.

## CER consumes a logical row

The solver-to-parser interface supplies at most one entry for each logical conductor in the selected-master row. For example, archived case10 contains 48 physical blocks representing 8 logical names; its output row contains 8 unique logical columns. The parser canonicalizes a numeric `digits__` prefix for name matching but does **not** sum multiple physical-fragment columns.

The conceptual sum over physical blocks in a method derivation belongs to the upstream accumulation/output representation. It must not be described as a new summation performed by `src/readout/cer.py`. The released upstream core is a binary dependency: the interface is verified from archived outputs, but its internal fragment-accumulation implementation is not independently source-audited here.

CER checks the observed logical set against the layout's expected set. For a complete row with at least one off-diagonal term, nonpositive couplings and a positive sum of coupling magnitudes, it returns that sum. Otherwise it retains the valid raw self term. Reference capacitance is not an activation input. Missing geometry columns are not silently fabricated: archived cases1-3, for example, lack a `GROUND` output column and therefore use the raw fallback.

The public parser rejects duplicate logical columns/masters; the old parser overwrote duplicates. This is stricter input validation, not an added aggregation rule. Re-parsing all 300 archived original outputs with both implementations gave identical parsed data and raw/CER readouts. Malformed duplicate input is outside that equivalence claim. The evaluator additionally rejects nonfinite or nonpositive self-capacitance output.
