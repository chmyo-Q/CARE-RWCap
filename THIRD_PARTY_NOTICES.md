# Third-party materials

## DeepRWCap

Source: <https://github.com/THU-numbda/deepRWCap>

Pinned revision: `117c26612467239f912187e21fd0cf594fe29d1e`.

The files in `third_party/deeprwcap/runtime/` are the upstream `deepRWCap` executable and `libdnnsolver.so`, `librwcap.so`, and `librwcapall.so`. Headers under `third_party/deeprwcap/include/` preserve the upstream C++ interface. The public10 layouts and references, including the bundled examples, come from the upstream cap3d benchmark. The solver binaries match the fixed upstream distribution. The pinned revision and the development checkout used in the paper differ only in the upstream README, not these runtime files.

DeepRWCap is distributed under the MIT license; its license text is retained at `third_party/deeprwcap/LICENSE`. The neural architecture and sampler integration build on the corresponding upstream code. Attribution: Hector R. Rodriguez, Jiechen Huang, and Wenjian Yu, “DeepRWCap: Neural-guided random-walk capacitance solver for IC design,” AAAI, 2026.

The full random-walk solver core is not supplied as rebuildable source by this package. Bundling the upstream binary is not a claim of authorship over that core.

## Optional GGFT data generation

The data-generation entry uses an external DeepRWCap checkout pinned to revision `9cb7fc69ed5fb54ce03be6fa8556402e10451be6`. Its GGFT source and Eigen dependency are not copied into this package. Retain the upstream and Eigen notices when using or redistributing those sources. See [the generation instructions](docs/DATA.md).

## Traditional CPU baselines

`third_party/deeprwcap/baselines/rwcap_agf`, `rwcap_microwalk` and `rwcap_fdm` are copied unchanged from DeepRWCap revision `9cb7fc69ed5fb54ce03be6fa8556402e10451be6`, directory `executable/baselines/`. They retain the same upstream MIT license at `third_party/deeprwcap/LICENSE`. The baseline revision is recorded separately from the frozen neural-runtime revision above. They are binary dependencies, not newly authored implementations.

## concurrentqueue

`third_party/deeprwcap/include/concurrentqueue.h` is by Cameron Desrochers, copyright 2013–2020, distributed under the Simplified BSD license stated in its header. That notice and disclaimer are preserved in full.

## External dependencies

PyTorch, Torch-TensorRT, CUDA, TensorRT, and other installed dependencies retain their respective licenses. Their distributions are not bundled in this repository; the project license does not relicense those dependencies.
