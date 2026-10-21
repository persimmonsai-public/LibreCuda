// cudaDeviceReset destroys the device's context: memory, streams, events and modules are released, and the next API
// call creates a fresh context in which kernels, symbols and streams work again.
#include "check.h"
#include <vector>

__device__ int counter;

__global__ void addTo(int *p, size_t n, int value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        p[i] += value;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&counter, 1);
    }
}

int main() {
    size_t free_start = 0, total = 0;
    CHECK(cudaMemGetInfo(&free_start, &total));

    const size_t n = 1 << 20;
    std::vector<int> host(n);
    for (int cycle = 0; cycle < 4; cycle++) {
        // a large allocation that is never freed: the reset must release it
        void *leak;
        CHECK(cudaMalloc(&leak, size_t(total / 4)));

        int *dev;
        CHECK(cudaMalloc(&dev, n * sizeof(int)));
        CHECK(cudaMemset(dev, 0, n * sizeof(int)));
        int zero = 0;
        CHECK(cudaMemcpyToSymbol(counter, &zero, sizeof(zero)));

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        cudaEvent_t event;
        CHECK(cudaEventCreate(&event));
        for (int i = 1; i <= 3; i++) {
            addTo<<<64, 256, 0, stream>>>(dev, n, i);
        }
        CHECK(cudaEventRecord(event, stream));
        CHECK(cudaEventSynchronize(event));
        CHECK(cudaMemcpy(host.data(), dev, n * sizeof(int), cudaMemcpyDeviceToHost));
        int launches = 0;
        CHECK(cudaMemcpyFromSymbol(&launches, counter, sizeof(launches)));
        for (size_t i = 0; i < n; i++) {
            EXPECT(host[i] == 6, "cycle %d: element %zu is %d", cycle, i, host[i]);
        }
        EXPECT(launches == 3, "cycle %d: counter is %d", cycle, launches);

        // odd cycles leave work queued and the stream/event alive
        if (cycle % 2 == 1) {
            addTo<<<64, 256, 0, stream>>>(dev, n, 1);
        } else {
            CHECK(cudaStreamDestroy(stream));
        }
        CHECK(cudaDeviceReset());
        // an event from before the reset can still be destroyed
        CHECK(cudaEventDestroy(event));
    }

    size_t free_end = 0;
    CHECK(cudaMemGetInfo(&free_end, &total));
    printf("free memory at start / after 4 resets: %.1f / %.1f MiB\n", free_start / 1048576.0, free_end / 1048576.0);
    EXPECT(free_end + (64 << 20) >= free_start, "memory leaked across resets");
    PASS("device_reset");
    return 0;
}
