# Third-party materials

## DeepRWCap

Source: <https://github.com/THU-numbda/deepRWCap>

Pinned revision: [117c266](https://github.com/THU-numbda/deepRWCap/tree/117c26612467239f912187e21fd0cf594fe29d1e).

The files in `third_party/deeprwcap/runtime/` are the upstream `deepRWCap` executable and `libdnnsolver.so`, `librwcap.so`, and `librwcapall.so`. Headers under `third_party/deeprwcap/include/` preserve the upstream C++ interface. The public10 layouts and references come from the upstream cap3d benchmark; the single-case and smoke commands use these same inputs.

DeepRWCap is distributed under the MIT license; its license text is retained at `third_party/deeprwcap/LICENSE`. The neural architecture and sampler integration build on the corresponding upstream code. Attribution: Hector R. Rodriguez, Jiechen Huang, and Wenjian Yu, “DeepRWCap: Neural-guided random-walk capacitance solver for IC design,” AAAI, 2026.

CARE-RWCap uses the prebuilt DeepRWCap solver and libraries in `third_party/deeprwcap/runtime/`. The CARE refinement sources are in `src/` and `cpp/`.

## Optional GGFT data generation

`scripts/generate_data.py` uses an external DeepRWCap checkout at revision [9cb7fc6](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6). GGFT and its Eigen dependency are built from that checkout. Retain the upstream and Eigen notices when using or redistributing those sources. See [the generation instructions](docs/DATA.md).

## Traditional CPU baselines

`third_party/deeprwcap/baselines/rwcap_agf`, `rwcap_microwalk` and `rwcap_fdm` are copied unchanged from DeepRWCap revision [9cb7fc6](https://github.com/THU-numbda/deepRWCap/tree/9cb7fc69ed5fb54ce03be6fa8556402e10451be6), directory `executable/baselines/`. They retain the same upstream MIT license at `third_party/deeprwcap/LICENSE`. This baseline revision differs from the neural-runtime revision listed above.

## concurrentqueue

`third_party/deeprwcap/include/concurrentqueue.h` is by Cameron Desrochers, copyright 2013–2020, distributed under the Simplified BSD license stated in its header. That notice and disclaimer are preserved in full.

## Additional layouts

The six candidate layouts under `benchmarks/additional/` were constructed for CARE-RWCap. Their archived reference estimates and screening decisions accompany them; they are separate from the upstream public10 dataset.

## External dependencies

PyTorch, Torch-TensorRT, CUDA, TensorRT, and other installed dependencies retain their respective licenses. Install these dependencies using the versions listed in [INSTALL.md](docs/INSTALL.md).
