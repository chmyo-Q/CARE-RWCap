# Paper reference results

`reference_public10.json` contains the original experiment from 2026-09-19: ten cases, seeds 2029–2038 and ten repeats per case/method. The file was generated from the saved solver outputs.

| Arm | Readout used in the historical main comparison | Macro SelfCapErr (%) |
|---|---|---:|
| DeepRWCap | Raw | 0.9277505 |
| DeepRWCap + CAPR | CER | 0.9000874 |
| CARE-RWCap | CER | 0.8109996 |

The JSON includes per-case raw/CER means and repeat SDs. Macro means equally weight the ten cases. `scripts/summarize.py` compares a completed batch with this reference. Public10 was used during method development; it is a fixed benchmark.

## Component configurations

| Configuration | Macro SelfCapErr (%) | Error reduction vs DeepRWCap (%) |
|---|---:|---:|
| DeepRWCap | 0.9278 | 0.00 |
| DeepRWCap + CER | 0.9150 | 1.37 |
| CAPR | 0.8720 | 6.00 |
| CAPR + CER | 0.9001 | 2.98 |
| CAPR + CPGR | 0.8356 | 9.94 |
| CARE-RWCap | 0.8110 | 12.58 |

These six readouts come from three solver configurations: DeepRWCap, CAPR and CAPR+CPGR. The original batch has no standalone DeepRWCap+CPGR arm. Error reductions use unrounded means. CER changes CAPR's macro error from 0.8720% to 0.9001%, and CARE's from 0.8356% to 0.8110%.

## Repeated evaluations

The [training-seed study](training_seeds/README.md) contains a later evaluation using a reused DeepRWCap control. The [CARE-only check](../docs/EVALUATION.md#tested-workflows) gave 0.9053% CER error versus 0.8110% in the original batch. Asynchronous FRW trajectories vary between runs, and CPGR's incremental gain has varied across checked batches. Keep batch identities, initial seeds and readout definitions when comparing results.
