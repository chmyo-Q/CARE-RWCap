# Paper reference results

`reference_public10.json` contains the paper experiment: ten cases, seeds 2029–2038 and ten repeats per case/method. The file was generated from the saved solver outputs.

| Arm | Main comparison readout | Macro SelfCapErr (%) |
|---|---|---:|
| DeepRWCap | Raw | 0.9277505 |
| DeepRWCap + CAPR | CER | 0.9000874 |
| CARE-RWCap | CER | 0.8109996 |

The JSON includes per-case raw/CER means and repeat SDs. Macro means equally weight the ten cases. `scripts/summarize.py` compares a completed batch with this reference. See [evaluation protocol](../docs/REPRODUCE.md#seed-and-repetition).

## Component configurations

| Configuration | Macro SelfCapErr (%) | Error reduction vs DeepRWCap (%) | Source |
|---|---:|---:|---|
| DeepRWCap | 0.9278 | 0.00 | Bundled reference |
| DeepRWCap + CER | 0.9150 | 1.37 | Bundled reference |
| CPGR | 0.8901 | 4.06 | Paper (Table IV) |
| CAPR | 0.8720 | 6.00 | Bundled reference |
| DeepRWCap + CPGR + CER | 0.8947 | 3.56 | Paper (Table IV) |
| CAPR + CER | 0.9001 | 2.98 | Bundled reference |
| CAPR + CPGR | 0.8356 | 9.94 | Bundled reference |
| CARE-RWCap | 0.8110 | 12.58 | Bundled reference |

`reference_public10.json` contains the three solver configurations shown in the first table, each with raw and CER readouts. Error reductions use unrounded means. `Bundled reference` identifies entries stored in that JSON; `Paper (Table IV)` identifies entries quoted from the paper.

## Run records

The [training-seed records](training_seeds/README.md) identify their own evaluation batch and reused baseline.
