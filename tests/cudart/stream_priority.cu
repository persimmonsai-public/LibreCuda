// Stream priorities: the range has a higher priority than the default one, and a kernel launched on a high priority
// stream while a long grid (about 40 waves of 1 ms blocks) occupies the GPU runs before most of that grid, instead
// of after all of it. For comparison, the same kernel on a normal stream (not checked: the order of equal priority
// grids is the hardware's). Checked against NVIDIA's runtime too.
#include "check.h"
#include <chrono>
#include <thread>

__device__ unsigned long long globalTimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// each block spins for `ns`; the last block to finish records the time
__global__ void longGrid(unsigned long long ns, unsigned long long *end) {
    const unsigned long long start = globalTimer();
    while (globalTimer() - start < ns) {
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(end, globalTimer());
}

__global__ void stamp(unsigned long long *at) {
    *at = globalTimer();
}

// ns from the launch of the long grid until the probe kernel ran, and until the long grid ended
static void race(cudaStream_t longStream, cudaStream_t probeStream, int blocks, unsigned long long *times,
                 double *probeMs, double *longMs) {
    CHECK(cudaMemset(times, 0, 3 * sizeof(unsigned long long)));
    CHECK(cudaDeviceSynchronize());
    stamp<<<1, 1, 0, longStream>>>(times + 0);
    longGrid<<<blocks, 128, 0, longStream>>>(1000000ULL, times + 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stamp<<<1, 1, 0, probeStream>>>(times + 2);
    CHECK(cudaDeviceSynchronize());
    unsigned long long h[3];
    CHECK(cudaMemcpy(h, times, sizeof(h), cudaMemcpyDeviceToHost));
    *longMs = (h[1] - h[0]) / 1e6;
    *probeMs = (h[2] - h[0]) / 1e6;
}

int main() {
    int least = 0, greatest = 0;
    CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    EXPECT(greatest < least, "no priority range: %d .. %d", least, greatest);

    cudaStream_t normal, other, high;
    CHECK(cudaStreamCreateWithPriority(&normal, cudaStreamNonBlocking, least));
    CHECK(cudaStreamCreateWithPriority(&other, cudaStreamNonBlocking, least));
    CHECK(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, greatest));
    int priority = 0;
    CHECK(cudaStreamGetPriority(high, &priority));
    EXPECT(priority == greatest, "high priority stream reports %d", priority);
    CHECK(cudaStreamGetPriority(normal, &priority));
    EXPECT(priority == least, "normal stream reports %d", priority);

    int device = 0, sms = 0, perSm = 0;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, longGrid, 128, 0));
    const int blocks = 40 * sms * perSm;
    unsigned long long *times;
    CHECK(cudaMalloc(&times, 3 * sizeof(unsigned long long)));

    double probeMs, longMs;
    race(normal, other, blocks, times, &probeMs, &longMs);
    printf("normal priority probe ran after %.1f ms of a %.1f ms grid\n", probeMs, longMs);
    race(normal, high, blocks, times, &probeMs, &longMs);
    printf("high priority probe ran after %.1f ms of a %.1f ms grid\n", probeMs, longMs);
    EXPECT(longMs > 30, "the long grid took only %.1f ms", longMs);
    EXPECT(probeMs < longMs / 2, "the high priority kernel waited for the grid (%.1f of %.1f ms)", probeMs, longMs);

    CHECK(cudaFree(times));
    CHECK(cudaStreamDestroy(normal));
    CHECK(cudaStreamDestroy(other));
    CHECK(cudaStreamDestroy(high));
    PASS("high priority streams run ahead of pending grids");
    return 0;
}
