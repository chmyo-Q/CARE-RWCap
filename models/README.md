# Models

## Deployment

`deeprwcap/` contains the five TensorRT FP16 baseline models, in this solver order:

1. `PoissonSelector_tensorrt_fp16.jit`
2. `PoissonPredictor_tensorrt_fp16.jit`
3. `GradientSelectorWeight_tensorrt_fp16.jit`
4. `Gradient2Predictor_tensorrt_fp16.jit`
5. `Gradient1Predictor_tensorrt_fp16.jit`

`capr/PoissonPredictor_tensorrt_fp16.jit` is the final CAPR predictor. CAPR uses this file and shares the other four engines with `deeprwcap/`. CPGR adds no learned checkpoint; it modifies the runtime gradient sampling path.

These files were frozen for the paper's aligned evaluation. They target RTX 4090 / Torch-TensorRT 2.6.0+cu126 / TensorRT 10.7.0. Use them with the deployment environment in [INSTALL.md](../docs/INSTALL.md).

## Provenance

The bundled DeepRWCap weights were trained locally with seed 2029 using the upstream architectures. The DeepRWCap Poisson predictor comes from the completed 200-epoch training run, with its best checkpoint selected at epoch 177; the other four models come from the corresponding verified seed-2029 model archive.

CAPR is the final S29 RF_RISK model: frozen DeepRWCap anchor, training seed 2029, 30-epoch schedule, validation-selected epoch 25. Its checkpoint, uncompiled TorchScript, and deployed engine are included. The architecture and loss are under `src/capr/`; the training recipe is documented in [METHOD.md](../docs/METHOD.md).

## Inspectable checkpoint files

`checkpoints/deeprwcap/` contains the original five `*_best.pt` state dictionaries and their uncompiled `*_best.jit` TorchScript files. `checkpoints/capr/` contains `best.pt` and `PoissonPredictor_best.jit`.

The `.pt` and uncompiled `.jit` files use FP32. They support model inspection and source-level loading; the inference runner loads the FP16 engines. `tests/check_models.py` demonstrates loading CAPR from source, checking its anchor against DeepRWCap, and comparing it with its uncompiled TorchScript. The remaining DeepRWCap architectures are available in the pinned upstream training source.

See [checkpoint lineage](../docs/METHOD.md#checkpoint-lineage) for training and checkpoint selection.

## Recorded CAPR recipe

The [recorded recipe](../configs/capr_training_recipe.json) and [method](../docs/METHOD.md) identify the frozen RF_RISK training settings and deployment lineage. The public default remains training seed 2029, selected epoch 25. The [three-seed supplement](../results/training_seeds/README.md) publishes capacitance records for the previously selected seed-2029/2039/2053 models (epochs 25/17/19); the extra two model files are not included in this compact result supplement.

## File compatibility

Current code uses `CAPRPredictor`, `DeepRWCapPredictor` and `load_deeprwcap` from `capr.model`, with `capr.loss.capr_loss` and `readout.cer.cer_value`. Earlier BPR/S29 and strict-S24 names refer to the same frozen components.

Use `src/capr/` and `src/readout/cer.py`, with `scripts/evaluate_readout.py` for saved solver outputs and `scripts/evaluate_transitions.py` for local mechanisms. Configuration and plan checks are in `input_checks.py`. Older code should use these current interfaces; checkpoint parameter keys remain unchanged.

Saved arm IDs `p0`, `bpr`, `full` correspond to CLI names `deeprwcap`, `capr`, `care-rwcap`; `--arm bpr` remains accepted. JSON retains equal-valued `s24_*` aliases alongside `cer_*`, and local result keys retain `bpr_*`. These storage names preserve existing readers and bootstrap streams; displayed tables use the current paper names.
