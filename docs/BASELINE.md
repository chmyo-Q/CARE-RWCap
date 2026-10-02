# Baseline and model provenance

The DeepRWCap baseline is the locally trained model set used in this project's aligned evaluation. It uses DeepRWCap neural architectures, locally trained with seed 2029. It is **not** the set of official pretrained engines in the upstream distribution. Reproducing the published upstream architecture and using our own trained weights are compatible statements; they are different from claiming byte-identical official models.

## Released artifacts

| Model | Verified origin | Released forms |
|---|---|---|
| PoissonPredictor | Local seed-2029 run completed at 200 epochs; best validation checkpoint at epoch 177 | FP32 state dict, uncompiled TorchScript, deployed FP16 engine |
| PoissonSelector | Corresponding archived local seed-2029 model set | Same three forms |
| GradientSelectorWeight | Corresponding archived local seed-2029 model set | Same three forms |
| Gradient2Predictor | Corresponding archived local seed-2029 model set | Same three forms |
| Gradient1Predictor | Corresponding archived local seed-2029 model set | Same three forms |
| BPR PoissonPredictor | Frozen DeepRWCap anchor plus final S29 RF_RISK adapter; validation epoch 25 of 30 | Same three forms |

The raw checkpoint archives, engine files and baseline anchor were reconciled during release preparation. This release supports evaluation with these supplied models. It does not claim full retraining reproducibility: the full local DeepRWCap training dataset and per-run training configuration are not bundled, and all historical per-model checkpoint-selection details have not been reconstructed.

## Upstream training reference, not a substitute for missing local run records

The pinned [DeepRWCap training entry point](https://github.com/THU-numbda/deepRWCap/blob/117c26612467239f912187e21fd0cf594fe29d1e/training_pytorch/src/main.py) defines batch size 16, a 200-epoch budget, learning rate 1e-3 and weight decay 1e-10. Its data split uses the first 90% for training and the last 10% for validation. The corresponding training loop uses AdamW and cosine warm restarts. These are upstream source defaults, not independently verified values for every locally trained checkpoint in this release. Do not present them as a fully recovered local DeepRWCap recipe.

The final BPR recipe, architecture and loss are documented separately in [METHOD.md](METHOD.md). BPR substitutes only the Poisson predictor. Both BPR and CPGR arms retain the same other four DeepRWCap engines.

For a like-for-like comparison use `models/paper_p0/`, the shipped solver, public10 inputs/references and the supplied protocol together. Using another baseline model set is a separate experiment and should be labeled accordingly. The dependency revision and license are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
