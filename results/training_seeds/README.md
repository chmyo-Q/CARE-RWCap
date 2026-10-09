# CAPR training-seed robustness supplement

This supplement evaluates three **previously selected** CAPR models in full CARE-RWCap. Only the CAPR Poisson predictor was replaced; the CPGR, CER, solver, references and public10 parameters were fixed. Each model received ten fresh solves on each of ten cases (initial FRW seeds 2029-2038). No new training, tuning or best-seed selection was performed for this evaluation.

| CAPR training seed | Selected epoch | Macro SelfCapErr (%) | Change (pp) | Relative change (%) | Improved cases |
|---|---:|---:|---:|---:|---:|
| 2029 | 25 | 0.8112113 | -0.1165393 | -12.5615 | 7/10 |
| 2039 | 17 | 0.8037911 | -0.1239595 | -13.3613 | 7/10 |
| 2053 | 19 | 0.8413895 | -0.0863611 | -9.3087 | 7/10 |

Mean +/- sample SD across the three model-specific macro means: **0.8187973 +/- 0.0199141%** (`n=3`, `ddof=1`). Each macro is the unweighted mean of ten case means. Relative changes use the unrounded historical DeepRWCap raw mean **0.927750543349835%** (displayed as 0.9278%), not the rounded display value.

The 100 DeepRWCap records are reused from `paper_final_aligned_20260919`. Matching case and initial seed does not make this a contemporaneous control or a common-random-path experiment. The cross-model SD also includes residual FRW evaluation noise. These observations support lower macro error for the three tested models against this historical DeepRWCap reference; they do not establish universal seed robustness, an isolated CPGR effect, or a runtime improvement. Each model loses on three cases, which remain included in the per-case table.

The new seed2029 mean **0.8112113%** is separate from the old main-batch Full mean **0.8109996%**. The checkpoint is the same; the observations are fresh stochastic solves. Neither value replaces the other.

## Recompute the tables (CPU only)

From the repository root, using Python 3.10 or later:

```bash
python scripts/summarize_training_seeds.py --output outputs/training_seed_summary
```

The output directory must not already exist. The script uses only the standard library, checks the complete 3 x 10 x 10 measured grid and 10 x 10 reused baseline grid, verifies reference/seed matching and recomputes errors from capacitances. It writes per-case and per-model CSVs, aggregate JSON and a Markdown table.

## Contents and boundary

- `repeat_results.csv`: all 300 Full repeat-level capacitances/errors and matched baseline values, copied without rounding from the accepted experiment archive.
- `baseline_runs.csv`: the 100 reused DeepRWCap raw capacitance records.
- `protocol.json`: intervention, training/solver seeds, selected epochs and comparison boundaries.
- `case_summary.csv`, `seed_summary.csv`, `aggregate.json`, `SUMMARY.md`: generated from the released records by the command above.

The numerical records were checked against the original solver outputs before release. This compact supplement does not contain the original solver logs or the two additional model files. The command independently reaggregates the released capacitance records; it does not rerun all three models. The default inference workflow continues to use the bundled training-seed2029 model. See the [frozen CAPR recipe](../../configs/capr_training_recipe.json) and [release scope](../../docs/ARTIFACT_SCOPE.md).
