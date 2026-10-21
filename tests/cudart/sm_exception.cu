// A kernel that hits an SM exception (here a misaligned store) must make the context report an error. LibreCUDA
// detects these exceptions in the background, so the error may surface one synchronization late; correct kernels must
// never report one.
#include "check.h"
#include <chrono>
#include <thread>

__global__ void fill(int *p, int n, int value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = value + i;
    }
}

__global__ void misalignedStore(int *p) {
    if (threadIdx.x == 3) {
        *reinterpret_cast<volatile int *>(reinterpret_cast<char *>(p) + 1) = 5;
    }
}

int main() {
    const int n = 1 << 20;
    int *buf;
    CHECK(cudaMalloc(&buf, n * sizeof(int)));

    // correct kernels for a while: no error may be reported
    const auto start = std::chrono::steady_clock::now();
    int rounds = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200)) {
        fill<<<64, 256>>>(buf, n, rounds++);
        CHECK(cudaDeviceSynchronize());
    }

    misalignedStore<<<1, 32>>>(buf);
    cudaError_t err = cudaSuccess;
    int syncs = 0;
    const auto fault_time = std::chrono::steady_clock::now();
    while (err == cudaSuccess && std::chrono::steady_clock::now() - fault_time < std::chrono::seconds(2)) {
        err = cudaDeviceSynchronize();
        syncs++;
        if (err == cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fault_time).count();
    printf("%d rounds of correct kernels; misaligned store reported as %s after %d synchronization(s), %.1f ms\n", rounds,
           cudaGetErrorName(err), syncs, ms);
    EXPECT(err == cudaErrorMisalignedAddress, "expected cudaErrorMisalignedAddress, got %s", cudaGetErrorName(err));
    // the error is sticky
    EXPECT(cudaDeviceSynchronize() == cudaErrorMisalignedAddress, "error is not sticky");
    PASS("sm_exception");
    return 0;
}
