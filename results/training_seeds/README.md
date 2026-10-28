# CAPR training-seed evaluation

Three validation-selected CAPR models were evaluated in CARE-RWCap. CPGR, CER, solver settings and references were fixed; each model used ten runs on each public10 case, with initial FRW seeds 2029–2038. The selected checkpoints were used as recorded, with no tuning in this evaluation.

| CAPR training seed | Selected epoch | Macro SelfCapErr (%) | Change (pp) | Relative change (%) | Improved cases |
|---|---:|---:|---:|---:|---:|
| 2029 | 25 | 0.8112113 | -0.1165393 | -12.5615 | 7/10 |
| 2039 | 17 | 0.8037911 | -0.1239595 | -13.3613 | 7/10 |
| 2053 | 19 | 0.8413895 | -0.0863611 | -9.3087 | 7/10 |

Mean ± sample SD across the three model-specific macro means: **0.8187973 ± 0.0199141%** (`n=3`, `ddof=1`). Each macro equally weights the ten case means. Relative changes use the unrounded DeepRWCap mean, **0.927750543349835%**.

The DeepRWCap control reuses 100 records from the original experiment, matched by case and initial seed. Controls and CARE runs were measured in different batches; trajectories are not shared. The SD across models includes FRW evaluation noise as well as training variation. Each model improves seven of ten cases.

The seed-2029 value **0.8112113%** comes from fresh solves with the same checkpoint used in the original **0.8109996%** batch. The two values identify separate evaluations.

## Generate the tables

From the repository root, using Python 3.10 or later:

```bash
python scripts/summarize_training_seeds.py --output outputs/training_seed_summary
```

Choose a new output directory. The command uses the Python standard library, checks all 300 CARE records and 100 baseline records, and recomputes errors from capacitances. It generates per-case and per-model CSVs, aggregate JSON and a Markdown table.

## Files

- `repeat_results.csv`: the 300 CARE repeat-level capacitances, errors and matched baseline values.
- `baseline_runs.csv`: the 100 reused DeepRWCap raw capacitance records.
- `protocol.json`: training seeds, selected epochs, solver seeds and evaluation settings.

Generated outputs are `case_summary.csv`, `seed_summary.csv`, `aggregate.json` and `SUMMARY.md`. The model files in the main package use training seed 2029; this saved-result supplement contains the seed-2039/2053 measurements, but not those two extra model files. See the [training recipe](../../configs/capr_training_recipe.json) for the model settings.
