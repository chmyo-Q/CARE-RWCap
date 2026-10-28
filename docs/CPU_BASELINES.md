# Traditional CPU baselines

| Method | Unmodified upstream executable | Runtime |
|---|---|---|
| FRW-AGF | `third_party/deeprwcap/baselines/rwcap_agf` | Linux x86_64 CPU |
| MicroWalk | `third_party/deeprwcap/baselines/rwcap_microwalk` | Linux x86_64 CPU |
| FRW-FDM | `third_party/deeprwcap/baselines/rwcap_fdm` | Linux x86_64 CPU |

Source: [DeepRWCap executable/baselines](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6/executable/baselines), revision `9cb7fc69ed5fb54ce03be6fa8556402e10451be6`. The upstream MIT license is retained at `third_party/deeprwcap/LICENSE`. This directory contains the prebuilt Linux executables.

## Run

Use Linux x86_64 and Python 3.10 or newer. The baseline runner uses only the standard library; no neural model, Torch, GPU, or CARE extension build is required. The binaries are statically linked ELF executables. After copying files from Windows or extracting an archive, grant executable permission:

```bash
chmod +x third_party/deeprwcap/baselines/rwcap_*
python3 scripts/baselines.py --method agf --output runs/agf_example
python3 scripts/baselines.py --method microwalk --output runs/microwalk_example
python3 scripts/baselines.py --method fdm --output runs/fdm_example
```

The default `quick` profile runs case7 once with initial seed 2029 and the convergence settings listed below. FRW-FDM may take appreciably longer than the other methods. For a full selected-method batch:

```bash
python3 scripts/baselines.py --method agf --profile paper --plan-only --output runs/agf_plan
python3 scripts/baselines.py --method agf --profile paper --output runs/agf_public10
```

Replace `agf` with `microwalk` or `fdm` as needed. Every invocation requires a new output directory. `--plan-only` also works on Windows. `--workers` defaults to 16; changing it records a distinct setting. Failed runs retain their outputs for inspection.

## Evaluation settings and output

The paper profile uses the same ten public inputs and reference self-capacitances as `configs/paper_protocol.json`: `p=c=0.01`, `c_ratio=0.95` for case1–6 and `0.3` for case7–10. It requests seeds 2029–2038 via the upstream `--seed` argument, repeat-major in case-number order, with a fresh process for each solve. Parallel thread scheduling can change trajectories between runs. CPU baselines use 16 solver threads by default; the frozen neural protocol uses 8 workers. These configurations must be identified when comparing runtime.

Each run keeps `result.out`, `result.log`, `console.log`, its command/configuration and raw `metrics.json`. A completed batch writes `per_run.csv`, `per_case.csv`, `summary.json` and `summary.md`: raw SelfCapErr, sample SD over repeats, equal-case macro mean, walks, weighted hops/walk and solver elapsed/CPU seconds. Reference values only enter evaluation. **CER is not applied to traditional baselines.** Outputs contain capacitance errors, workload and solver timing.

For runtime comparisons, run the methods on the same machine and report their thread counts. See [validation status](EVALUATION.md#tested-workflows) for the tested workflows.
