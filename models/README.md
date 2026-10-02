# Models

## Deployment

`paper_p0/` contains the five TensorRT FP16 models actually used as the paper baseline, in this solver order:

1. `PoissonSelector_tensorrt_fp16.jit`
2. `PoissonPredictor_tensorrt_fp16.jit`
3. `GradientSelectorWeight_tensorrt_fp16.jit`
4. `Gradient2Predictor_tensorrt_fp16.jit`
5. `Gradient1Predictor_tensorrt_fp16.jit`

`bpr/PoissonPredictor_tensorrt_fp16.jit` is the final BPR predictor. BPR uses this file and shares the other four engines with `paper_p0/`. CPGR adds no learned checkpoint; it modifies the runtime gradient sampling path.

These files were frozen for the paper's aligned evaluation. They target RTX 4090 / Torch-TensorRT 2.6.0+cu126 / TensorRT 10.7.0. The `.jit` extension does not mean a TensorRT engine is portable across arbitrary hardware or library versions.

## Provenance

The paper P0 models were trained locally with seed 2029 using the DeepRWCap architecture. They are **not** the five official pretrained deployment files from the upstream repository. The P0 Poisson predictor comes from the completed 200-epoch training run, with its best checkpoint selected at epoch 177; the other four models come from the corresponding verified seed-2029 model archive. Do not substitute the upstream pretrained models while labeling the baseline `paper_p0`.

BPR is the final S29 RF_RISK model: frozen P0 anchor, training seed 2029, 30-epoch schedule, validation-selected epoch 25. Its checkpoint, uncompiled TorchScript, and deployed engine are included. The architecture and loss are under `src/bpr/`; the training recipe is documented in [METHOD.md](../docs/METHOD.md).

## Inspectable checkpoint files

`checkpoints/paper_p0/` contains the original five `*_best.pt` state dictionaries and their uncompiled `*_best.jit` TorchScript files. `checkpoints/bpr/` contains `best.pt` and `PoissonPredictor_best.jit`.

The `.pt` and uncompiled `.jit` files use FP32. They are supplied for inspection and loading, not as silent substitutes for the frozen deployment engines. `tests/check_models.py` demonstrates loading BPR from source, checking its anchor against paper P0, and comparing it with its uncompiled TorchScript. The remaining P0 architectures are available in the pinned upstream training source.

See [baseline provenance and training-scope boundaries](../docs/BASELINE.md) and [paper terminology](../docs/PAPER_MAPPING.md).

## Audited BPR recipe

The [recorded recipe](../configs/bpr_training_recipe.json) and [implementation contract](../docs/IMPLEMENTATION_CONTRACT.md) identify the frozen RF_RISK training settings and deployment lineage. The public default remains training seed 2029, selected epoch 25. The [three-seed supplement](../results/training_seeds/README.md) publishes capacitance records for the previously selected seed2029/2039/2053 models (epochs 25/17/19); the extra two model files are not included in this compact result supplement.
