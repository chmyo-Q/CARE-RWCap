# Models

The solver loads five FP16 engines from `deeprwcap/`, in this order: PoissonSelector, PoissonPredictor, GradientSelectorWeight, Gradient2Predictor and Gradient1Predictor. CAPR replaces only PoissonPredictor with `capr/PoissonPredictor_tensorrt_fp16.jit`. CPGR and CER add no learned weights.

`checkpoints/deeprwcap/` contains FP32 state dictionaries and uncompiled TorchScript for all five baseline models. These weights were trained locally using the DeepRWCap architectures and training seed 2029. The Poisson predictor was selected at epoch 177 of a 200-epoch run. `checkpoints/capr/` contains CAPR `best.pt` and its FP32 TorchScript; the model uses seed 2029 and validation-selected epoch 25.

The bundled FP16 engines use the deployment environment described in [REPRODUCE.md](../docs/REPRODUCE.md). [Export instructions](../docs/REPRODUCE.md#export-and-rebuild-engines) rebuild engines for the current GPU from these FP32 sources. Set `CARE_MODEL_DIR` to use the exported directory without replacing bundled models.

The architecture and objective are in `src/capr/`. `tests/check_models.py` compares source predictions, anchor weights and FP32 TorchScript. See [METHOD.md](../docs/METHOD.md) for the training recipe.
