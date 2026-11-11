#include "grid.cuh"
#include <cuda_runtime.h>
#include <mutex>
static unsigned long long *counter_ptr = nullptr;
static std::once_flag initialize_counts;
static int allocation_error = 0;
__global__ void project(const float *x, const float *q, const unsigned char *faces, float *out,
                        float *ratio, int B, unsigned long long *counts) {
    int b = blockIdx.x;
    if (b >= B)
        return;
    bool side = faces[b] >= 2;
    __shared__ unsigned mask;
    __shared__ float p[care_cpgr::kFaceCells], r[care_cpgr::kFaceCells],
        s0[care_cpgr::kBlockThreads], s1[care_cpgr::kBlockThreads];
    if (threadIdx.x == 0)
        mask = 3;
    for (int k = threadIdx.x; k < care_cpgr::kFaceCells; k += blockDim.x)
        p[k] = q[b * care_cpgr::kFaceCells + k];
    __syncthreads();
    // Strict dielectric reflection eligibility, Eqs. (6)-(7).
    unsigned local = 3;
    for (int k = threadIdx.x; k < care_cpgr::kVolumeCells && local; k += blockDim.x) {
        int z = k / care_cpgr::kFaceCells, y = k / care_cpgr::N % care_cpgr::N,
            t = k % care_cpgr::N;
        float v = x[b * care_cpgr::kVolumeCells + k];
        if ((local & 1) && v != x[b * care_cpgr::kVolumeCells + z * care_cpgr::kFaceCells +
                                  y * care_cpgr::N + care_cpgr::kLastIndex - t])
            local &= ~1u;
        int reflected =
            side ? (care_cpgr::kLastIndex - z) * care_cpgr::kFaceCells + y * care_cpgr::N + t
                 : z * care_cpgr::kFaceCells + (care_cpgr::kLastIndex - y) * care_cpgr::N + t;
        if ((local & 2) && v != x[b * care_cpgr::kVolumeCells + reflected])
            local &= ~2u;
    }
    atomicAnd(&mask, local);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(&counts[mask], 1ULL);
    float a = 0, c = 0;
    for (int k = threadIdx.x; k < care_cpgr::kFaceCells; k += blockDim.x) {
        int y = k / care_cpgr::N, t = k % care_cpgr::N;
        float v = p[k], sign = side ? -1.f : 1.f;
        // Parity projection: average only the eligible reflection operators.
        // The second reflection is odd for side faces, even for faces 0/1.
        if (mask == 1)
            v = (p[k] + p[y * care_cpgr::N + care_cpgr::kLastIndex - t]) * .5f;
        else if (mask == 2)
            v = (p[k] + sign * p[(care_cpgr::kLastIndex - y) * care_cpgr::N + t]) * .5f;
        else if (mask == 3)
            v = ((p[k] + p[y * care_cpgr::N + care_cpgr::kLastIndex - t]) +
                 sign *
                     (p[(care_cpgr::kLastIndex - y) * care_cpgr::N + t] +
                      p[(care_cpgr::kLastIndex - y) * care_cpgr::N + care_cpgr::kLastIndex - t])) *
                .25f;
        r[k] = v;
        a += fabsf(p[k]);
        c += fabsf(v);
    }
    s0[threadIdx.x] = a;
    s1[threadIdx.x] = c;
    __syncthreads();
    for (int d = care_cpgr::kBlockThreads / 2; d; d /= 2) {
        if (threadIdx.x < d) {
            s0[threadIdx.x] += s0[threadIdx.x + d];
            s1[threadIdx.x] += s1[threadIdx.x + d];
        }
        __syncthreads();
    }
    // Eq. (10): L1 compensation; inactive inputs retain unit weight.
    if (threadIdx.x == 0)
        ratio[b] = mask ? (s0[0] > 0 ? s1[0] / s0[0] : 0.f) : 1.f;
    // A zero projection retains the original proposal and carries zero weight.
    for (int k = threadIdx.x; k < care_cpgr::kFaceCells; k += blockDim.x)
        out[b * care_cpgr::kFaceCells + k] = s1[0] > 0 ? r[k] : p[k];
}
extern "C" int parity_launch(const float *x, const float *q, const unsigned char *faces, float *out,
                             float *ratio, int B, void *stream) {
    std::call_once(initialize_counts, [] {
        allocation_error = (int)cudaMallocManaged(&counter_ptr, 4 * sizeof(unsigned long long));
        if (!allocation_error)
            for (int k = 0; k < 4; ++k)
                counter_ptr[k] = 0;
    });
    if (allocation_error)
        return allocation_error;
    if (B)
        project<<<B, care_cpgr::kBlockThreads, 0, (cudaStream_t)stream>>>(x, q, faces, out, ratio,
                                                                          B, counter_ptr);
    return (int)cudaGetLastError();
}
extern "C" int projection_counts(unsigned long long *out) {
    auto e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        return (int)e;
    if (!counter_ptr) {
        for (int k = 0; k < 4; ++k)
            out[k] = 0;
        return 0;
    }
    return (int)cudaMemcpy(out, counter_ptr, 4 * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost);
}

#include "selector.cuh"
