# Public ten-case benchmark

The ten layouts and reference files are bundled from the fixed DeepRWCap public benchmark used in the aligned experiment. No download or access to the authors' server is needed.

`configs/paper_protocol.json` selects one master per layout, its reference self-capacitance, solver settings, and seeds. DSPF cases use the declared `*|NET` total for that master. Case10 uses its frozen SPICE-derived reference. Reference values only enter error evaluation, not CER activation.

Quick repeated comparison (case1, seeds 2029–2031, three arms: nine solves):

```bash
python scripts/benchmark.py --output runs/quick
```

Case1 is an illustrative example selected for its lower CARE-RWCap mean error in earlier evaluations. The quick batch reports all three new runs per arm; stochastic values and rankings may differ. Use the full ten-case profile for a performance comparison. The default single-run example also uses case1; smoke uses its separate case7 configuration to exercise the conditional modules.

Paper accuracy protocol (10 cases × 10 seeds × 3 arms = 300 measured solves, plus 3 excluded warmups):

```bash
python scripts/benchmark.py --profile paper --plan-only --output runs/paper_plan
python scripts/benchmark.py --profile paper --output runs/paper
python scripts/summarize.py --input runs/paper
```

Use a new output directory for each invocation. `--plan-only` checks the planned order without needing Torch or a GPU. The public arm names are `deeprwcap`, `capr`, and `care-rwcap`, matching the three-arm comparison. Historical run IDs and directories remain unchanged; see [file compatibility](../../models/README.md#file-compatibility). The paper profile uses a fixed Python shuffle seed 20260919 and cycles the six arm permutations exactly as in that archived protocol. Warmup uses case8 and initial seed 2040 for each arm. A new batch remains a new set of stochastic observations.

The runner stops at the first failed solve and does not automatically retry. It records the plan before running, preserves all partial output, and writes coverage information. The summary command returns exit code 2 for an incomplete batch and withholds aggregate performance comparisons. Diagnose failures before explicitly starting a new batch; do not selectively replace unfavorable observations.

`summary.json` includes per-case raw and CER/S24 errors, within-case sample SD, macro means, between-case sample SD, and seed-block method differences. The paper endpoint compares DeepRWCap raw with CAPR/Full CER. Same-readout differences are also reported, so the effects of readout changes are not mistaken for CPGR's standalone effect. Sharing a seed block does not mean trajectories are identical. These statistics are descriptive; there is no automatic significance claim or accuracy pass threshold.

See `results/reference_public10.json` for the compact historical reference, and `results/README.md` for its interpretation. The reference file contains aggregate results only, not the full historical logs.
