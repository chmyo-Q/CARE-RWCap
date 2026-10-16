# Additional layouts

Six candidate layouts were screened before end-to-end evaluation; five were retained. These are the layouts of Table VI and Fig. 4 of manuscript v4. `new004` remains included to document the reference-screening exclusion and has no end-to-end configuration.

## References and screening

[references.csv](references.csv) records FRW-FDM self-capacitances at stopping tolerances 0.005 and 0.0025. Both reference solves use 16 threads and `c_ratio=0.30`, with `p=c` set to the stated tolerance. The final reference is the raw target self term from the 0.0025 result. Discrepancy is `100 * abs(C_0.005 - C_0.0025) / C_0.0025`; retain a layout if this is at most 0.5%. New004 has discrepancy 0.7054% and was excluded before the end-to-end comparison. The records are copied from the preserved reference study, not regenerated. The two-tolerance discrepancy is a consistency check, not a confidence interval or proof of exact capacitance.

| Layout | Arrangement | Discrepancy (%) | Retained |
|---|---|---:|:---:|
| new001 | Elbow with offset neighbors | 0.4418 | Yes |
| new002 | Closed frame with inner islands | 0.3918 | Yes |
| new003 | Staircase with staggered neighbors | 0.2649 | Yes |
| new004 | Two connected interdigitated combs | 0.7054 | No |
| new005 | Segmented offset bus | 0.0930 | Yes |
| new006 | Cross with corner islands | 0.4888 | Yes |

## Single layout

Use the [supported neural environment](../../docs/INSTALL.md) and build the extensions. From the repository root:

```bash
python scripts/run.py --config configs/additional/new001.json \
  --arm care-rwcap --seed 9263701 --output runs/additional_new001
```

Replace `new001` with another retained layout; use `--arm deeprwcap` for the control. Both commands preserve raw and CER. The primary additional-layout comparison uses **CER for both arms**, unlike public10's DeepRWCap-raw versus CARE-CER main endpoint.

## Ten-repeat execution protocol

Initial seeds are 9263701–9263710, with eight solver workers, `p=c=0.01`, `c_ratio=0.30`, and OMP/MKL threads set to 1. The configuration files preserve these numerical settings. The original study interleaved both methods in layout order within each repeat and excluded one case8 warmup per method at seed 9263799. This Bash block uses the same ordering and requires a fresh `runs/additional` directory:

```bash
set -eu
test ! -e runs/additional
for arm in deeprwcap care-rwcap; do
  python scripts/run.py --config configs/additional/warmup.json \
    --arm "$arm" --output "runs/additional/warmup/$arm"
done
for repeat in {1..10}; do
  seed=$((9263700 + repeat))
  index=0
  for layout in new001 new002 new003 new005 new006; do
    arms="deeprwcap care-rwcap"
    if (( (repeat + index) % 2 == 0 )); then arms="care-rwcap deeprwcap"; fi
    for arm in $arms; do
      python scripts/run.py --config "configs/additional/$layout.json" \
        --arm "$arm" --seed "$seed" \
        --output "runs/additional/$layout/seed$seed/$arm"
    done
    index=$((index + 1))
  done
done
```

This runs 100 formal solves and two excluded warmups. It stops on a failed command; existing results are preserved. Keep all predetermined cases and seeds. Each run records its actual models, command, metrics and configuration.

For Table VI, compute each layout's mean and sample SD of `cer_self_error_percent` across its ten repeats; macro-average the five means and report their across-layout sample SD. The archived cohort gave DeepRWCap+CER 1.4334% and CARE-RWCap 1.3346%, with improvements in three of five layouts. Sharing initial seeds does not force identical asynchronous trajectories; new observations can differ. The compact package supplies inputs, references and execution configurations, not the complete historical log archive or a dedicated Table-VI bootstrap/figure entry. Do not feed this directory to the public10 summary command: it does not have that command's plan/status schema or endpoint definition.
