# Release scope

This is a fixed-model inference and public-benchmark release, not the full research archive or a guarantee that every paper experiment is independently reproducible from this package.

| Task | Included | Boundary |
|---|---|---|
| Inspect BPR, CPGR, CER | Architecture/loss, CUDA projection/selector/compensation, frozen readout | Full training driver and training data excluded |
| Run CARE-RWCap and DeepRWCap | Frozen engines, checkpoints, runtime and single-case runner | Supported Linux x86_64 / RTX 4090 deployment |
| Evaluate public10 | Ten layouts/references, 300-solve protocol, raw/CER summary | A fixed development benchmark; stochastic outcomes can differ |
| Compare traditional solvers | Upstream FRW-AGF, MicroWalk and FRW-FDM binaries; CPU-only runner | No source rebuild of these upstream solver cores |
| Check module effects | Same-readout DeepRWCap/BPR/Full differences and CER effects | Three solver arms do not identify CPGR's standalone effect or a full factorial interaction |
| Inspect the framework | Author-supplied overview PNG/PDF and stage-to-code map | Diagram shows multi-master repetition; the released evaluator handles one selected master per invocation |
| Review historical results | Original 300-run compact per-case and macro reference | Not all experimental cohorts or original logs |
| Inspect local transition metrics | Optional frozen evaluator and fixed split | Original external datasets required; not bundled |
| Evaluate a custom geometry | Neural single-run JSON configuration | One logical master task; reference needed for error; unsupported GPU engines require additional work |
| Reproduce every paper table | Not provided | Coupling-row normalized L1 summaries, additional-layout inputs, dedicated memory/microtiming studies and plotting archive excluded |
| Retrain or rebuild the complete solver | Not provided | BPR inspection and own CUDA extension builds are supported; upstream core remains a binary dependency |

The original paper 300-run cohort remains the main historical reference. A separate [training-seed supplement](../results/training_seeds/README.md) releases 300 Full and 100 reused DeepRWCap capacitance records with CPU reaggregation; it does not bundle the extra two models or a new multi-model inference runner. CPGR's incremental end-to-end benefit has not been consistent across checked batches; the reference does not establish a repeatable ranking or a universal tolerance. Numerical functional checks are distinct from validating an accuracy claim.

The release keeps the input/model/configuration and metric definitions needed to run its documented workflows. It does not require users to repeat every experiment. New outputs are retained and evaluated under their actual settings; reference values are not used to adjust predictions or select favorable runs.
