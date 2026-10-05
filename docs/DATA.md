# Data sources and GGFT generation

CARE-RWCap uses two different types of input:

- **Public capacitance benchmarks:** the ten layouts and reference capacitances in `benchmarks/public10/`. These are sufficient to run the released frozen models. Reference capacitances are used only to compute errors.
- **Local transition data:** dielectric cubes and numerical Poisson/Gradient reference kernels. These support training and local mechanism evaluation; they are not the ten benchmark layouts.

The local-data generator is upstream DeepRWCap's **General Green's Function Table (GGFT)** solver. It uses finite differences to calculate reference kernels. CARE-RWCap does not introduce a new reference-label generator.

## Source and fixed version

Use [DeepRWCap revision 9cb7fc69ed5fb54ce03be6fa8556402e10451be6](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6/ggft). The source, Makefile and Eigen dependency are in its `ggft/` directory. The upstream MIT license and Eigen's own licenses apply. This revision fixes the optional generation workflow below; it is not proof that newly generated files equal the archived paper data.

On Linux, install Git, GNU Make and a C++17 compiler with OpenMP support. For Ubuntu:

```bash
sudo apt-get install -y git build-essential
```

From the CARE-RWCap repository root, prepare a separate upstream checkout:

```bash
git clone https://github.com/THU-numbda/deepRWCap.git ../deeprwcap-data
git -C ../deeprwcap-data checkout --detach 9cb7fc69ed5fb54ce03be6fa8556402e10451be6
```

The supplied Eigen dependency is used by the Makefile. This CPU workflow does not require PyTorch, TensorRT or a GPU.

## Generate data

First check the workflow with two samples of each kind:

```bash
python3 scripts/generate_data.py --upstream ../deeprwcap-data \
  --kind both --samples 2 --threads 1 --output data/ggft_example
```

To use the sample count and thread count in the upstream generation script:

```bash
python3 scripts/generate_data.py --upstream ../deeprwcap-data \
  --kind both --samples 100000 --threads 12 --output data/ggft_100k
```

Choose a new output directory. `--kind poisson` and `--kind gradient` generate only the selected dataset. The wrapper checks the upstream revision and unchanged tracked GGFT sources, rebuilds GGFT, runs the two datasets sequentially, and validates each file's header and record count. It preserves `build.log`, per-dataset logs and `generation.json` alongside the generated files.

Each 100,000-sample file is **12,362,400,016 bytes** (about 12.36 GB). The two final files total about 24.72 GB; allow additional space for the thread-local files during merging. The two-sample command exercises the pipeline but does not produce enough data for model training or paper evaluation.

## Generation parameters and randomness

The pinned upstream code uses:

| Parameter | Value |
|---|---|
| Grid size | `N = 23` |
| Dielectric block width | `1` |
| FDM iterative tolerance | `1e-6` |
| Initial random blocks | `5`, followed by upstream nesting/padding/background rules |
| Structure slots | 15 block slots plus one background slot |
| Poisson mode | `GFT` |
| Gradient mode | `WVTZ` |
| Random generator | `std::mt19937`, seeded with the OpenMP thread index |

Thread count changes the random streams, workload partition and concatenated sample order. The upstream generator does **not** take CARE's BPR training seed or FRW solver seed. The values 2029 and 2029–2038 in the other workflows are not GGFT generation seeds. Keep the generation thread count and source version fixed when comparing generated datasets; compiler/library differences can also affect numerical results.

## Binary layout

The documented Linux x86_64 output uses little-endian float64 values throughout. Each file begins with two float64 header values, `N` and `block_width`; these are not integer-encoded fields. Each subsequent record contains:

| Field | Values per record | Meaning |
|---|---:|---|
| Dielectric cube | `23^3 = 12,167` | Permittivity samples |
| Structure description | `16 * 7 = 112` | Center, radii and permittivity for each structure slot |
| Reference kernel | `6 * 23^2 = 3,174` | Values on the six cube faces |
| Total | `15,453` | `123,624` bytes per record |

The structure description is retained in the file but is not an input to the released BPR network. Preserve upstream storage and face conventions; do not add an axis permutation or reorder faces merely to match a visualization.

A minimal inspection with NumPy is:

```python
import numpy as np

path = "data/ggft_example/poisson.bin"
data = np.memmap(path, dtype="<f8", mode="r")
assert tuple(data[:2]) == (23.0, 1.0)
records = data[2:].reshape(-1, 15453)
dielectric = records[:, :12167].reshape(-1, 1, 1, 23, 23, 23)
structure = records[:, 12167:12279].reshape(-1, 16, 7)
kernels = records[:, 12279:].reshape(-1, 6, 1, 23, 23)
```

## Connection to CARE-RWCap

For BPR, the recorded training recipe normalizes each dielectric cube by its spatial maximum (using divisor 1 for a zero maximum). The face-zero target is `abs(kernel) + 1e-10`, normalized over its 23-by-23 face. The archived split uses a NumPy PCG64 permutation with seed 20260805: 90,000 training samples and 10,000 validation samples. See [the model and loss](METHOD.md) and [training recipe](../configs/bpr_training_recipe.json).

CPGR uses signed Gradient predictions and introduces no trainable checkpoint. Gradient data supply numerical reference kernels for local evaluation. CER acts on the solver's output capacitance row and does not use GGFT data.

**Generating data does not retrain or replace the bundled models.** This release provides BPR architecture/loss and the recorded recipe, but not a complete training-and-TensorRT-export entry. Newly generated files are inputs for further research; generating the same number of records does not establish identity with the archived training or validation samples.

The optional [transition evaluator](LOCAL_VALIDATION.md) requires the original frozen files and verifies their identity. It will not accept these newly generated files as a reproduction unless they actually match. The archived full datasets are not bundled and currently have no public download endpoint. Public10 inference is available without them.
