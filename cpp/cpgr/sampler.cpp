#include "cuda_kernels.h"
#include "dnn_solver.h"
#include <atomic>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fstream>
extern "C" int parity_launch(const float *, const float *, const uint8_t *, float *, float *, int,
                             void *);
extern "C" int projection_counts(unsigned long long *);
static std::atomic<unsigned long long> calls{0}, samples{0};
extern "C" int joint_selector_launch(const float *, const float *, float *, int, void *);
extern "C" int joint_counts(unsigned long long *);
static bool joint_enabled() {
    const char *p = getenv("CARE_CPGR_SELECTOR");
    return p && std::string(p) == "1";
}
// Diagnostics are opt-in and written to an explicit run directory. Ordinary
// library loading must not synchronize CUDA or create files in the caller's cwd.
static bool enabled() {
    const char *value = std::getenv("CARE_CPGR_PROJECTION");
    return value && std::string(value) == "1";
}
static thread_local at::Tensor parity_ratio;

extern "C" void care_cpgr_write_diagnostics() {
    const char *directory = std::getenv("CARE_DIAGNOSTICS_DIR");
    if (!directory || !*directory)
        return;
    unsigned long long projection[4] = {}, selector[8] = {};
    const int projection_status = samples.load() ? projection_counts(projection) : 0;
    const int selector_status = joint_enabled() ? joint_counts(selector) : 0;
    std::ofstream p(std::string(directory) + "/PROJECTION_COUNTS.json");
    p << "{\"calls\":" << calls.load() << ",\"samples\":" << samples.load()
      << ",\"cuda_status\":" << projection_status << ",\"masks\":[";
    for (int i = 0; i < 4; ++i) {
        if (i)
            p << ',';
        p << projection[i];
    }
    p << "]}\n";
    std::ofstream s(std::string(directory) + "/JOINT_COUNTS.json");
    s << "{\"cuda_status\":" << selector_status << ",\"input_xyz_masks\":[";
    for (int i = 0; i < 8; ++i) {
        if (i)
            s << ',';
        s << selector[i];
    }
    s << "]}\n";
}

torch::Tensor DNNSolverGrad::predictGradient(torch::Tensor &dielectricTensor,
                                             torch::Tensor &faceIds, torch::Tensor &axis) {
    torch::InferenceMode guard; // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();

    if (!enabled()) {
        using Fn = at::Tensor (*)(void *, at::Tensor &, at::Tensor &, at::Tensor &);
        static auto original =
            (Fn)dlsym(RTLD_NEXT, "_ZN13DNNSolverGrad15predictGradientERN2at6TensorES2_S2_");
        TORCH_CHECK(original, "original predictor missing");
        return original(this, dielectricTensor, faceIds, axis);
    }
    int batch_size = dielectricTensor.size(0);

    // Use pre-allocated gradient buffer instead of torch::zeros
    auto gradient = gradient_buffer_1_.slice(0, 0, batch_size).view({batch_size, N, N});
    auto rotated_tensor =
        rotate_faces_gradient_launcher(dielectric_buffer_2_, dielectricTensor, faceIds)
            .view({batch_size, 1, 1, N, N, N});

    auto face01_mask = (faceIds == 0) | (faceIds == 1);
    auto face01_indices = torch::nonzero(face01_mask).view(-1);
    auto face_other_indices = torch::nonzero(~face01_mask).view(-1);

    if (face01_indices.numel() > 0) {
        auto subset1 = rotated_tensor.index_select(0, face01_indices);
        auto g1 = gradientFace1Predictor.forward({subset1}).toTensor();
        gradient.index_copy_(0, face01_indices, g1);
    }
    if (face_other_indices.numel() > 0) {
        auto subset2 = rotated_tensor.index_select(0, face_other_indices);
        auto g2 = gradientFace2PredictorWithSign.forward({subset2}).toTensor();
        gradient.index_copy_(0, face_other_indices, g2);
    }

    calls.fetch_add(1);
    samples.fetch_add(batch_size);
    auto projected = at::empty_like(gradient);
    parity_ratio = at::empty({batch_size, 1}, gradient.options());
    TORCH_CHECK(rotated_tensor.is_contiguous() && gradient.is_contiguous(), "parity layout");
    int err = parity_launch(rotated_tensor.data_ptr<float>(), gradient.data_ptr<float>(),
                            faceIds.data_ptr<uint8_t>(), projected.data_ptr<float>(),
                            parity_ratio.data_ptr<float>(), batch_size,
                            (void *)c10::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(err == 0, "parity CUDA error: ", cudaGetErrorString((cudaError_t)err));
    gradient = projected;
    // Flatten the gradient for post-processing
    auto gradient_flat = gradient.view({batch_size, NN});

    // Use gradient_buffer_2_ for post-processing output
    auto processed_gradient =
        post_process_gradient_launcher(gradient_buffer_2_, gradient_flat, faceIds, axis);

    return processed_gradient;
}

torch::Tensor DNNSolverGrad::sampleGradient(torch::Tensor &dielectricTensor,
                                            torch::Tensor &max_vals, torch::Tensor &axis) {
    torch::InferenceMode guard; // Enable inference mode for this method
    // Stream is already set by calling function, but ensure it's set
    setCurrentStream();

    if (!enabled()) {
        using Fn = at::Tensor (*)(void *, at::Tensor &, at::Tensor &, at::Tensor &);
        static auto original =
            (Fn)dlsym(RTLD_NEXT, "_ZN13DNNSolverGrad14sampleGradientERN2at6TensorES2_S2_");
        TORCH_CHECK(original, "original sampler missing");
        return original(this, dielectricTensor, max_vals, axis);
    }
    // Rotate dielectric tensor based on the axis
    for (int ax = 0; ax < 2; ++ax) {
        auto mask = (axis == ax);
        auto nonz = mask.nonzero();
        if (nonz.numel() > 0) {
            auto idx = nonz.squeeze(1).to(torch::kLong);
            auto subset = dielectricTensor.index_select(0, idx);
            if (ax == 0)
                subset = subset.rot90(-1, /*dims=*/{-3, -1});
            else if (ax == 1)
                subset = subset.rot90(-1, /*dims=*/{-3, -2});
            dielectricTensor.index_copy_(0, idx, subset);
        }
    }

    auto fpw = gradientFaceSelectorWeightPredictor.forward({dielectricTensor}).toTensor();

    if (joint_enabled()) {
        auto constrained = at::empty_like(fpw);
        TORCH_CHECK(dielectricTensor.is_contiguous() && fpw.is_contiguous(), "joint layout");
        int error = joint_selector_launch(dielectricTensor.data_ptr<float>(), fpw.data_ptr<float>(),
                                          constrained.data_ptr<float>(), fpw.size(0),
                                          (void *)c10::cuda::getCurrentCUDAStream().stream());
        TORCH_CHECK(error == 0, "joint selector CUDA failed");
        fpw = constrained;
    }
    auto faceProbabilities = fpw.slice(1, 0, 6);
    auto selectedFaceTensor = torch::multinomial(faceProbabilities, 1, false);
    auto selectedFacesUint8 = selectedFaceTensor.view({-1}).to(torch::kUInt8);

    auto gradient = predictGradient(dielectricTensor, selectedFacesUint8, axis);
    // Eq. (13): sample in-face positions from |Pg|; predictGradient has
    // restored the upstream face orientation and sign convention.
    auto absGradient = torch::abs(gradient);
    auto sampleGradientTensor = torch::multinomial(absGradient, 1, false).view(-1);
    auto sampleGradientInt16 = sampleGradientTensor.to(torch::kInt16);

    auto positions = locate_index_cuda_launcher(sampleGradientInt16, selectedFacesUint8, axis);

    auto weights = fpw.slice(1, 6, 7);

    // Get sign of sampled gradient using gather (more efficient than batch indexing)
    auto sign = torch::sign(gradient);
    auto sampleSign = torch::gather(sign, 1, sampleGradientTensor.unsqueeze(1));

    auto centerDielectric = dielectricTensor
                                .select(1, 0)     // select channel 0
                                .select(1, 0)     // select first element of next dimension
                                .select(1, N / 2) // select center x
                                .select(1, N / 2) // select center y
                                .select(1, N / 2) // select center z
                                .view({-1, 1}) *
                            max_vals.view({-1, 1});

    // Single concatenation operation
    auto positionsAndWeightsAndCenter =
        torch::cat({positions,
                    // Signed sampling contribution, with the Eq. (10) L1 compensation.
                    weights * sampleSign * parity_ratio, centerDielectric},
                   1);

    parity_ratio = at::Tensor();
    return positionsAndWeightsAndCenter;
}