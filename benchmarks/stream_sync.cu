// Waits between streams and between a stream's kernels and copies, on one device: two streams handing over to each
// other through events (a kernel, an event record, a cudaStreamWaitEvent on the other stream, per hop), a kernel and a
// 4 byte device to host copy on one stream back to back (the copy waits for the kernel on the GPU), and a kernel, the
// copy and a synchronize. Best of 6 rounds of 2000 (after 4000 untimed: the first few thousand operations of a stream
// also grow its command buffers). Plain CUDA runtime code: build with nvcc (-cudart shared) and run on any runtime
// (LibreCUDA or NVIDIA's).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t error_ = (call); if (error_ != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(error_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void increment(int *p) {
    *p += 1;
}

template<typename F> static double best(F body) {
    const int n = 2000;
    for (int i = 0; i < 2 * n; i++) {
        body();
    }
    CHECK(cudaDeviceSynchronize());
    double us = 1e30;
    for (int round = 0; round < 6; round++) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) {
            body();
        }
        CHECK(cudaDeviceSynchronize());
        us = std::min(us, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n);
    }
    return us;
}

int main() {
    int *device, *host;
    CHECK(cudaMalloc(&device, 2 * sizeof(int)));
    CHECK(cudaMallocHost(&host, sizeof(int)));
    cudaStream_t a, b;
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    cudaEvent_t ea, eb;
    CHECK(cudaEventCreateWithFlags(&ea, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&eb, cudaEventDisableTiming));

    printf("stream to stream through an event, per hop:   %6.2f us\n", best([&] {
        increment<<<1, 1, 0, a>>>(device);
        CHECK(cudaEventRecord(ea, a));
        CHECK(cudaStreamWaitEvent(b, ea, 0));
        increment<<<1, 1, 0, b>>>(device + 1);
        CHECK(cudaEventRecord(eb, b));
        CHECK(cudaStreamWaitEvent(a, eb, 0));
    }) / 2);
    printf("kernel + 4 B copy to the host, back to back: %6.2f us\n", best([&] {
        increment<<<1, 1, 0, a>>>(device);
        CHECK(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, a));
    }));
    printf("kernel + 4 B copy to the host + synchronize: %6.2f us\n", best([&] {
        increment<<<1, 1, 0, a>>>(device);
        CHECK(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, a));
        CHECK(cudaStreamSynchronize(a));
    }));
    return 0;
}
