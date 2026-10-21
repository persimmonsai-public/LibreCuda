#include <librecuda.h>

#include <iostream>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cstring>

inline void cudaCheck(libreCudaStatus_t error, const char *file, int line) {
    if (error != LIBRECUDA_SUCCESS) {
        const char *error_string;
        libreCuGetErrorString(error, &error_string);
        printf("[CUDA ERROR] at file %s:%d: %s\n", file, line, error_string);
        exit(EXIT_FAILURE);
    }
};
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

// Creates and destroys a context repeatedly in one process. Odd iterations leave the stream, memory and module
// alive, so destroying the context has to release them.
int main() {
    CUDA_CHECK(libreCuInit(0));

    LibreCUdevice device{};
    CUDA_CHECK(libreCuDeviceGet(&device, 0));

    std::vector<uint8_t> image{};
    {
        std::ifstream input("fill.cubin", std::ios::binary);
        image.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }

    constexpr int n = 1 << 16;
    constexpr int iterations = 6;
    std::vector<float> host(n);

    for (int iteration = 0; iteration < iterations; iteration++) {
        const bool leak = iteration % 2 == 1;

        LibreCUcontext ctx{};
        CUDA_CHECK(libreCuCtxCreate_v2(&ctx, CU_CTX_SCHED_YIELD, device));

        LibreCUmodule module{};
        CUDA_CHECK(libreCuModuleLoadData(&module, image.data(), image.size()));
        LibreCUFunction func{};
        CUDA_CHECK(libreCuModuleGetFunction(&func, module, "fill"));

        LibreCUstream stream{};
        CUDA_CHECK(libreCuStreamCreate(&stream, 0));

        void *a{}, *b{};
        CUDA_CHECK(libreCuMemAlloc(&a, n * sizeof(float)));
        CUDA_CHECK(libreCuMemAlloc(&b, n * sizeof(float)));

        float value = float(iteration) * 1000.0f;
        int count = n;
        void *params[] = {&a, &value, &count};
        CUDA_CHECK(libreCuLaunchKernel(func, n / 256, 1, 1, 256, 1, 1, 0, stream, params, 3, nullptr));

        // device to device (memcpy kernels of the context) and device to host
        CUDA_CHECK(libreCuMemCpy(b, a, n * sizeof(float), stream));
        memset(host.data(), 0, n * sizeof(float));
        CUDA_CHECK(libreCuMemCpy(host.data(), b, n * sizeof(float), stream));
        CUDA_CHECK(libreCuStreamAwait(stream));

        for (int i = 0; i < n; i++) {
            if (host[i] != value + float(i)) {
                printf("iteration %d: mismatch at %d: %f != %f\n", iteration, i, host[i], value + float(i));
                return EXIT_FAILURE;
            }
        }

        if (!leak) {
            CUDA_CHECK(libreCuMemFree(a));
            CUDA_CHECK(libreCuMemFree(b));
            CUDA_CHECK(libreCuStreamDestroy(stream));
            CUDA_CHECK(libreCuModuleUnload(module));
        }
        CUDA_CHECK(libreCuCtxDestroy(ctx));
        if (leak) {
            // the stream outlived its context; destroying it must still be safe
            CUDA_CHECK(libreCuStreamDestroy(stream));
        }
        std::cout << "iteration " << iteration << " ok" << std::endl;
    }
    std::cout << "ctx_recreate passed" << std::endl;
    return 0;
}
