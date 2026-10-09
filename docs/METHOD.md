# Method

The [framework overview](figures/overview.png) follows the first Gradient transition (CPGR), subsequent Poisson transitions (CAPR), and same-solve endpoint readout (CER). The section order below groups the learned component before the runtime operators. See [paper-to-code mapping](PAPER_MAPPING.md) for stage order and the released single-master interface.

## CAPR: Condition-Aware Poisson Refinement

Implementation: `src/capr/model.py`; public class: `CAPRPredictor`; historical alias: `ResidualFactorizedPredictor`.

The input is a normalized dielectric tensor of shape `(B,1,1,23,23,23)`. The Poisson predictor returns a nonnegative `(B,23,23)` conditional face distribution. Normalize the dielectric tensor by its spatial maximum; use divisor 1 when that maximum is zero. The upstream solver performs this normalization for inference.

The frozen anchor consists of 2D positional coordinates, a 1×1 projection to 16 channels, depthwise-separable blocks with channels 16→16→8→4→2 and dilations 1,1,2,3, and a one-channel head. Each separable block has depthwise convolution, BatchNorm, GELU, pointwise convolution, BatchNorm, and GELU. Despite the historical `skip` member name, these blocks do not add a residual shortcut. ReLU plus `1e-10`, followed by normalization, yields `q0`.

The trainable adapter has a 23→24 pointwise convolution, two depthwise blocks (dilations 1 and 2), and a 24→12 pointwise projection; these stages use BatchNorm and GELU. Two heads determine the positional redistribution and scalar refinement strength. Using the manuscript's notation:

```
r          = tanh(residual_head(features))
r_centered = r - sum(q0 * r)
a_theta    = tau * sigmoid(gate(mean(features)))   # tau = 0.125
q_theta    = softmax(log(clamp(q0, min=1e-10)) + a_theta * r_centered)
```

In `CAPRPredictor`, `max_log_dose` stores `tau`, `components()` returns `(q0, r_centered, a_theta)`, and `forward()` returns `q_theta`. The checkpoint's `anchor`, `adapter.residual` and `adapter.gate` parameter keys retain their original names.

The anchor stays frozen, including BatchNorm statistics. The residual head starts at zero; initialization recovers the anchor distribution up to clamping and floating-point normalization. The final model has 2,869 parameters, of which 1,466 belong to the trainable adapter. Only the Poisson predictor is replaced in the solver.

### Loss and model selection

For each sample, define:

```
K = KL(target || q_theta)
A = mean_j <q_theta - target, probe_j>²
T = KL(q_theta || q0)
L = K + 10 A + 2 T
batch_loss = mean(L) + 0.5 * mean(largest ceil(0.25 * batch_size) values of L)
```

The six probes on the `[-1,1]²` grid are `x`, `y`, `xy`, `cos(πx)`, `cos(πy)`, and `cos(πx)cos(πy)`. Each is centered and normalized by its maximum absolute value. `src/capr/loss.py` implements these expressions, including the log clamps used during training.

The validation selection score is **different from the training tail loss**:

```
mean(K) + 10 mean(A) + 2 mean(T) + 0.5 * (q95(K) + 10 q95(A))
```

The final recipe uses 100,000 Poisson samples, face-zero targets `abs(kernel)+1e-10` normalized to unit sum, a 90,000/10,000 split using NumPy PCG64 permutation with seed 20260805, and training seed 2029. Adapter optimization uses AdamW (learning rate 3e-4, weight decay 1e-6), batch size 16, 30 epochs, gradient norm clipping at 1, and cosine decay to 5e-6. Epoch shuffling uses a Torch generator seeded with `2029 + epoch`; worker count is 0. Validation selects epoch 25. Full training data and the training driver are not bundled.

The fixed local evaluation uses the deployed engines. The archived CAPR training settings are recorded in [capr_training_recipe.json](../configs/capr_training_recipe.json). The source-FP32 to deployed-FP16 paths differ numerically for DeepRWCap and CAPR; deployment-level gains should not all be attributed to the residual adapter without a precision-controlled comparison.

## CPGR: Conditional Parity Gradient Refinement

Implementation: `cpp/cpgr/projection.cu`, `selector.cuh`, and `sampler.cpp`.

CPGR operates in the gradient branch. It changes both the conditional signed kernel and, under the corresponding exact input symmetry, opposite-face selection probabilities. The pretrained gradient networks are unchanged.

1. Rotate the input to the gradient-axis/face convention used by the upstream solver.
2. Detect reflection symmetries by exact equality of the discrete dielectric input. This is a structural condition, not an approximate numerical threshold on predicted kernels.
3. Average each eligible pair of opposite-face probabilities. Preserve the seventh selector output, the global weight.
4. Project the predicted signed kernel onto the eligible parity subspace. The horizontal reflection is even. The second reflection is even for faces 0/1 and odd for the other faces. Multiple applicable reflections are averaged jointly.
5. Sample a location proportionally to the absolute projected kernel and apply its sign together with `rho = ||Pg||_1 / ||g||_1` to the walk weight.

For an active, nonzero projection, this yields the conditional identity

```
E[ sign((Pg)_J) * rho * f(J) ] = sum_j (Pg)_j f(j) / ||g||_1,
    J ~ |Pg| / ||Pg||_1.
```

This identity explains the compensation factor relative to the original kernel normalization. It does not imply the projected prediction equals the exact physical kernel, nor guarantee a lower end-to-end error for every layout. If the projection is zero, the implementation retains the original sampling proposal and sets its compensating weight to zero. With no eligible symmetry, the kernel and correction are unchanged (`rho=1`).

The production extension interposes the pinned `DNNSolverGrad` methods using `LD_PRELOAD`. The legacy environment names `S29_GRADIENT_PARITY_ENABLE` and `S29_GRADIENT_JOINT_ENABLE` are preserved for compatibility; the public runner enables both for CPGR. CPGR is not enabled in the `deeprwcap` and `capr` arms. This mechanism requires the shipped upstream C++ ABI.

## CER: Conditional Endpoint Re-estimation

Implementation: `src/readout/cer.py`. Historical alias S24 is retained to avoid conflating the rule with an unvalidated variant.

The solver must provide an already aggregated logical-conductor row. Extract the expected logical conductor set and selected master from the layout. Names matching `digits__name` are mapped to `name`. For the selected matrix row:

- Require the observed logical conductor set to equal the expected set.
- Require at least one off-diagonal term, all off-diagonal values nonpositive, and a positive sum of their absolute values.
- If all conditions hold, read the self capacitance as that absolute coupling sum.
- Otherwise retain the raw self capacitance and report the fallback reason.

The evaluator rejects nonfinite matrices and missing/nonpositive master self capacitance. It does not use the reference capacitance to activate S24. The readout arithmetic is preserved for valid, unique logical rows. The parser now rejects duplicate logical columns and duplicate master blocks instead of silently overwriting them; it does not sum physical-fragment entries. Re-evaluation of all 300 archived valid outputs confirmed unchanged raw and strict-S24 capacitances. See [paper terminology and input contract](PAPER_MAPPING.md).

For a parsed row with a finite positive raw self term, the exact applicability rule can be written as:

```
S = expected_logical_columns - {master}
CER applies iff:
    observed_logical_columns == expected_logical_columns
    and S is nonempty
    and all(coupling[j] <= 0 for j in S)
    and sum(abs(coupling[j]) for j in S) > 0
```

The input-validation stage rejects duplicate logical records and nonfinite values before this gate. Thus an all-zero coupling row uses the raw fallback, whereas a malformed/nonfinite row is an error. These are different outcomes. The archived valid paper outputs do not require relaxing either check.

## Frozen recipe and interface evidence

The [implementation contract](IMPLEMENTATION_CONTRACT.md) gives the source of each training setting, the exact Gradient opposite-face pairs, and the solver/CER boundary. The [CAPR recipe JSON](../configs/capr_training_recipe.json) records the historical settings; it is metadata and does not change the deployed model or act as a training entry point.
