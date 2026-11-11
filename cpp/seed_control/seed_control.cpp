// Measurement support only: initialize RNGs once before main; forward core calls unchanged.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <torch/torch.h>

static std::atomic<unsigned long long> calls{0};
static long long first_seeds[64]{};
static unsigned long long requested_seed = 0;
static int core_before = 0;
static int (*original_main)(int, char **, char **) = nullptr;
static void save_audit() {
    const char *directory = std::getenv("CARE_DIAGNOSTICS_DIR");
    if (!directory || !*directory)
        return;
    std::string path = std::string(directory) + "/SEED_AUDIT.json";
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        return;
    fprintf(f,
            "{\"requested_seed\":%llu,\"torch_manual_seed_called\":true,\"core_initial_seed_"
            "before\":%d,\"core_initial_seed_set\":%llu,\"core_set_seed_calls\":%llu,\"first_core_"
            "seed_arguments\":[",
            requested_seed, core_before, requested_seed, calls.load());
    for (unsigned long long i = 0; i < calls.load() && i < 64; i++)
        fprintf(f, "%s%lld", i ? "," : "", first_seeds[i]);
    fprintf(f, "],\"scope\":\"Initial Torch CPU/CUDA and exported core seed initialized; thread "
               "scheduling and common random paths not guaranteed\"}\n");
    fclose(f);
}
extern "C" void record_core_seed(void *, long long) asm("_ZN5rwcap3Sea6Random8set_seedEx");
extern "C" void record_core_seed(void *self, long long value) {
    using F = void (*)(void *, long long);
    static F real = (F)dlsym(RTLD_NEXT, "_ZN5rwcap3Sea6Random8set_seedEx");
    if (!real) {
        fprintf(stderr, "missing core set_seed\n");
        abort();
    }
    auto i = calls.fetch_add(1);
    if (i < 64)
        first_seeds[i] = value;
    real(self, value);
}
static int seeded_main(int argc, char **argv, char **envp) {
    const char *s = getenv("CARE_SOLVER_SEED");
    if (!s || !*s) {
        fprintf(stderr, "CARE_SOLVER_SEED required\n");
        return 91;
    }
    char *end = nullptr;
    requested_seed = strtoull(s, &end, 10);
    if (*end || requested_seed == 0 || requested_seed > 2147483647ULL)
        return 92;
    int *core = (int *)dlsym(RTLD_DEFAULT, "_ZN5rwcap3Sea6Random4seedE");
    if (!core)
        return 93;
    core_before = *core;
    *core = (int)requested_seed;
    torch::manual_seed(requested_seed);
    const int status = original_main(argc, argv, envp);
    // Write diagnostics before shared-library teardown, only when requested.
    if (std::getenv("CARE_DIAGNOSTICS_DIR")) {
        using WriteDiagnostics = void (*)();
        auto write_cpgr =
            reinterpret_cast<WriteDiagnostics>(dlsym(RTLD_DEFAULT, "care_cpgr_write_diagnostics"));
        if (write_cpgr)
            write_cpgr();
        save_audit();
    }
    return status;
}
extern "C" int __libc_start_main(int (*mainfn)(int, char **, char **), int argc, char **argv,
                                 void (*init)(), void (*fini)(), void (*rtld_fini)(),
                                 void *stack_end) {
    using F = int (*)(int (*)(int, char **, char **), int, char **, void (*)(), void (*)(),
                      void (*)(), void *);
    auto real = (F)dlsym(RTLD_NEXT, "__libc_start_main");
    original_main = mainfn;
    return real(seeded_main, argc, argv, init, fini, rtld_fini, stack_end);
}
