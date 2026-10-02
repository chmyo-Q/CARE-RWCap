# Author reference results

`reference_public10.json` summarizes the archived aligned batch dated 2026-09-19: ten public cases, initial solver seeds 2029–2038, and the released paper model set. Each case/arm has ten repetitions. It was recomputed from the preserved original outputs while preparing this release; it is **not** a new independent reproduction.

| Arm | Readout used in the historical main comparison | Macro SelfCapErr (%) |
|---|---|---:|
| P0 | Raw | 0.9277505 |
| BPR | CER / strict-S24 | 0.9000874 |
| Full: BPR + CPGR | CER / strict-S24 | 0.8109996 |

The JSON also supplies per-case raw and CER/S24 means, sample SD over repetitions, and both raw/S24 macro results. Macro means give equal weight to each case. Between-case variation is not the same as repeated-run variation.

This is an identified historical reference, not a target that every new batch must match. Fixed initial seeds do not fully fix asynchronous trajectories. No empirically validated universal replication tolerance is asserted. New results should be reported as a separate batch under the predetermined protocol, including unfavorable outcomes. These ten public cases were used during development and are not an unseen test.

`scripts/summarize.py` reports a new complete batch's difference from this reference without deciding success based on sign or proximity. Accuracy, statistical uncertainty, and runtime claims should be interpreted using their respective protocols. This release's batch script targets accuracy and solver-reported work/timing; it does not recreate the separate cold-process VRAM study or the S24 microbenchmark.

## Interpretation of module comparisons

The original cohort contains three solver arms with two readouts each: P0 raw/CER, BPR raw/CER, and BPR+CPGR raw/CER. Its BPR+CPGR raw mean is **0.8355514%**. It contains no P0+CPGR solve; three-arm results alone cannot identify CPGR's standalone effect or a BPR-by-CPGR factorial interaction.

The main reference above contains only this original cohort. A separate [BPR training-seed supplement](training_seeds/README.md) contains a later Full-only evaluation and explicitly identifies its reused P0 baseline. In other checked batches, CPGR's incremental end-to-end effect did not consistently retain the favorable direction seen here. Consequently, this historical table is not evidence of a stable cross-batch CPGR gain. The package includes neither all research cohorts nor their raw archives.

## All observed configurations in the original batch

| Configuration | Macro SelfCapErr (%) | Error reduction vs P0 (%) |
|---|---:|---:|
| DeepRWCap | 0.9278 | 0.00 |
| DeepRWCap + CER | 0.9150 | 1.37 |
| BPR | 0.8720 | 6.00 |
| BPR + CER | 0.9001 | 2.98 |
| BPR + CPGR | 0.8356 | 9.94 |
| CARE-RWCap | 0.8110 | 12.58 |

These six rows come from three solver arms, each with raw and CER readouts of the same solve. They are not six independent solver arms or a complete factorial experiment. There is no P0+CPGR observation in this cohort. Relative changes use unrounded means; the CER-only reduction is 1.37%. CER lowers the macro error for P0 and Full in this batch but increases it for the BPR-only arm (0.8720% to 0.9001%). All configurations are retained here.
