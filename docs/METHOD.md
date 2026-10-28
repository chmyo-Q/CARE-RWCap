# Method

The [overview](figures/overview.png), also available as a [vector PDF](figures/overview.pdf), is the author-supplied Fig. 1 of manuscript v8. It follows the first neural Gradient transition (CPGR), subsequent Poisson transitions (CAPR), conductor-hit accumulation, and endpoint readout (CER). These two formats document the same figure and are not runtime inputs. The released runner evaluates one configured master per invocation and retains its capacitance row. The upstream runtime supplies neural/non-neural path selection, scheduling and stopping behavior.

## CPGR: Conditional Parity Gradient Refinement

Implementation: `cpp/cpgr/projection.cu`, `selector.cuh`, and `sampler.cpp`.

CPGR operates in the gradient branch. It changes both the conditional signed kernel and, under the corresponding exact input symmetry, opposite-face selection probabilities. The pretrained gradient networks are unchanged.

1. Rotate the input to the gradient-axis/face convention used by the upstream solver.
2. Detect reflection symmetries by exact equality of the discrete dielectric input. This is a structural condition, not an approximate numerical threshold on predicted kernels.
3. Average each eligible pair of opposite-face probabilities. Preserve the seventh selector output, the global weight.
4. Project the predicted signed kernel onto the eligible parity subspace. The horizontal reflection is even. The second reflection is even for faces 0/1 and odd for the other faces. Multiple applicable reflections are averaged jointly.
5. Sample a location proportionally to the absolute projected kernel and apply its sign together with `rho = ||Pg||_1 / ||g||_1` to the walk weight.

The strict selector masks use the rotated Gradient input convention:

| Reflected input index | Mask bit | Zero-based opposite faces |
|---|---:|---|
| x (fastest index) | 1 | 4, 5 |
| y | 2 | 2, 3 |
| z | 4 | 0, 1 |

Each eligible pair receives its arithmetic mean before `torch::multinomial` draws the face. Pair sums are preserved up to roundoff; inactive pairs and the seventh output remain unchanged. No original-to-corrected face-probability importance ratio is applied. The identity below concerns conditional in-face sampling, not preservation of the original joint face-and-position law.

The paper's contribution coefficient is `W_f = eta_f * w`: `w` is selector output 7 and `eta_0=-1` (other faces use +1). The upstream `post_process_gradient_launcher` restores this face-0 sign before the sampler computes `sampleSign`. The returned path-weight column is `fpw[:,6:7] * sampleSign * parity_ratio`; center permittivity is returned separately. No additional face-probability importance ratio is introduced.

For an active, nonzero projection, this yields the conditional identity

```
E[ sign((Pg)_J) * rho * f(J) ] = sum_j (Pg)_j f(j) / ||g||_1,
    J ~ |Pg| / ||Pg||_1.
```

The identity gives the conditional expectation under projected-kernel sampling. End-to-end error also depends on the subsequent transitions and conductor endpoints. If the projection is zero, the implementation retains the original sampling proposal and sets its compensating weight to zero. With no eligible symmetry, the kernel and correction are unchanged (`rho=1`).

The production extension interposes the pinned `DNNSolverGrad` methods using `LD_PRELOAD`. The legacy environment names `S29_GRADIENT_PARITY_ENABLE` and `S29_GRADIENT_JOINT_ENABLE` are preserved for compatibility; the public runner enables both for CPGR. CPGR is not enabled in the `deeprwcap` and `capr` arms. This mechanism requires the shipped upstream C++ ABI.

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

Validation uses the following checkpoint-selection score:

```
mean(K) + 10 mean(A) + 2 mean(T) + 0.5 * (q95(K) + 10 q95(A))
```

The final recipe uses 100,000 Poisson samples, face-zero targets `abs(kernel)+1e-10` normalized to unit sum, a 90,000/10,000 split using NumPy PCG64 permutation with seed 20260805, and training seed 2029. Adapter optimization uses AdamW (learning rate 3e-4, weight decay 1e-6), batch size 16, 30 epochs, gradient norm clipping at 1, and cosine decay to 5e-6. Epoch shuffling uses a Torch generator seeded with `2029 + epoch`; worker count is 0. Validation selects epoch 25. The training settings are listed in the recipe below.

The fixed local evaluation uses the deployed engines. The archived CAPR training settings are recorded in [capr_training_recipe.json](../configs/capr_training_recipe.json). Local evaluation uses the FP16 deployment engines. FP32 checkpoint and FP16 engine outputs can differ numerically.

## CER: Conditional Endpoint Re-estimation

Implementation: `src/readout/cer.py`, with parsing in `src/readout/parser.py`.

The solver must provide an already aggregated logical-conductor row. Extract the expected logical conductor set and selected master from the layout. Names matching `digits__name` are mapped to `name`. For the selected matrix row:

- Require the observed logical conductor set to equal the expected set.
- Require at least one off-diagonal term, all off-diagonal values nonpositive, and a positive sum of their absolute values.
- If all conditions hold, read the self capacitance as that absolute coupling sum.
- Otherwise retain the raw self capacitance and report the fallback reason.

The parser requires finite values, a positive master self term and unique logical columns/master blocks. The solver provides the aggregated row; CER uses only that row and the layout conductor set. Reference capacitance is used when computing error.

For a parsed row with a finite positive raw self term, the exact applicability rule can be written as:

```
S = expected_logical_columns - {master}
CER applies iff:
    observed_logical_columns == expected_logical_columns
    and S is nonempty
    and all(coupling[j] <= 0 for j in S)
    and sum(abs(coupling[j]) for j in S) > 0
```

The input-validation stage rejects duplicate logical records and nonfinite values before this gate. Thus an all-zero coupling row uses the raw fallback, whereas a malformed/nonfinite row is an error.

## Checkpoint lineage

The DeepRWCap baseline uses the upstream architectures and locally trained seed-2029 weights. Its Poisson predictor was selected at epoch 177 of a completed 200-epoch run; the other four predictors/selectors come from the corresponding archived model set. See [models](../models/README.md) for the included weights and deployment files.

The final CAPR checkpoint is the RF_RISK adapter selected at epoch 25 in `train_seed2029_30e_20260909_180652`. Its training recipe comes from the archived launcher, implementation and run metadata. The released state dictionary, uncompiled TorchScript, FP16 engine and anchor were checked against the original aligned-batch files. The [recipe JSON](../configs/capr_training_recipe.json) records the optimizer, data split and selection rule. Its post-hoc sensitivity section records the five refinement bounds and validation-selected epochs. The sensitivity sweep was performed after the main model was fixed. See [model files](../models/README.md) for deployment order and inspectable artifacts.

## Manuscript-to-source map

| TCAD manuscript v8 definition | Source |
|---|---|
| CPGR projection, Eqs. (6)–(9) | `cpp/cpgr/projection.cu` |
| CPGR contribution, Eqs. (10)–(14) | `cpp/cpgr/projection.cu`, `sampler.cpp`; paper `beta_f` is code `ratio` |
| CAPR parameterization, Eqs. (15)–(21) | `src/capr/model.py`: `ResidualHead`, `CAPRPredictor.components`, `forward` |
| CAPR objective, Eqs. (22)–(24) | `src/capr/loss.py`: `components`, `capr_loss`; separate `validation_score` |
| CER, Eqs. (25)–(27) | `src/readout/parser.py`, `cer.py`; full applicability gate above |
| SelfCapErr, Eq. (28) | `scripts/evaluate_readout.py` |
| Local Gradient metrics, Eqs. (29)–(30) | `scripts/transition_metrics.py`: `kernel_rows`, `summarize_kernel` |

The solver-output pipeline forms logical-conductor rows before CER. For example, archived case10 has 48 physical blocks and eight unique logical output columns. The parser reads these logical-conductor columns; physical-block accumulation takes place in the upstream solver.

Historical names BPR/S29, Gradient-Joint and strict-S24 correspond to CAPR, CPGR and CER. Current imports and commands use the paper names; stored result keys and checkpoint class aliases are retained. See [file compatibility](../models/README.md#file-compatibility).
