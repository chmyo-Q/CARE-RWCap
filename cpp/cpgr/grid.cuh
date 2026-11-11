#pragma once

namespace care_cpgr {
// The frozen networks and upstream rotation kernels use a 23-point grid.
constexpr int N = 23;
constexpr int kLastIndex = N - 1;
constexpr int kFaceCells = N * N;
constexpr int kVolumeCells = N * N * N;
constexpr int kBlockThreads = 256;
} // namespace care_cpgr
