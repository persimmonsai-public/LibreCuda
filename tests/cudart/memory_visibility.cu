// Memory written by kernels is visible to what is ordered after them, many times over with fresh values: another
// stream's kernel after an event (all SMs write, the other kernel checks every word), the next kernel of the stream,
// the CPU after a synchronize (kernels writing page-locked host memory), and a copy to the host after the kernel. Each
// round writes a new value so stale data shows up as a mismatch.
#include "check.h"

#include <cstring>
#include <vector>

constexpr int WORDS = 1 << 18;  // 1 MiB
constexpr int ROUNDS = 3000;

__global__ void fill(unsigned *p, unsigned value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WORDS; i += gridDim.x * blockDim.x) {
        p[i] = value ^ i;
    }
}

__global__ void check(const unsigned *p, unsigned value, unsigned *mismatches) {
    unsigned bad = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < WORDS; i += gridDim.x * blockDim.x) {
        bad += p[i] != (value ^ i);
    }
    if (bad != 0) {
        atomicAdd(mismatches, bad);
    }
}

int main() {
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = 2 * sms;
    unsigned *device, *mismatches, *host, *copied;
    CHECK(cudaMalloc(&device, WORDS * sizeof(unsigned)));
    CHECK(cudaMalloc(&mismatches, sizeof(unsigned)));
    CHECK(cudaMemset(mismatches, 0, sizeof(unsigned)));
    CHECK(cudaMallocHost(&host, WORDS * sizeof(unsigned)));
    CHECK(cudaMallocHost(&copied, WORDS * sizeof(unsigned)));
    cudaStream_t a, b;
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    cudaEvent_t written, checked;
    CHECK(cudaEventCreateWithFlags(&written, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&checked, cudaEventDisableTiming));

    // another stream's kernel after an event, and the next kernel of the same stream
    for (int round = 0; round < ROUNDS; round++) {
        const unsigned value = 0x9e3779b9u * (round + 1);
        CHECK(cudaStreamWaitEvent(a, checked, 0));
        fill<<<blocks, 256, 0, a>>>(device, value);
        check<<<blocks, 256, 0, a>>>(device, value, mismatches);
        CHECK(cudaEventRecord(written, a));
        CHECK(cudaStreamWaitEvent(b, written, 0));
        check<<<blocks, 256, 0, b>>>(device, value, mismatches);
        CHECK(cudaEventRecord(checked, b));
    }
    CHECK(cudaDeviceSynchronize());
    unsigned bad = 0;
    CHECK(cudaMemcpy(&bad, mismatches, sizeof(unsigned), cudaMemcpyDeviceToHost));
    EXPECT(bad == 0, "%u words read stale after a kernel on the same or another stream", bad);

    // page-locked host memory written by a kernel, read by the CPU after a synchronize; device memory copied to the
    // host after the kernel
    long stale_host = 0, stale_copy = 0;
    for (int round = 0; round < ROUNDS / 10; round++) {
        const unsigned value = 0x85ebca6bu * (round + 1);
        fill<<<blocks, 256, 0, a>>>(host, value);
        fill<<<blocks, 256, 0, a>>>(device, value);
        CHECK(cudaMemcpyAsync(copied, device, WORDS * sizeof(unsigned), cudaMemcpyDeviceToHost, a));
        CHECK(cudaStreamSynchronize(a));
        for (int i = 0; i < WORDS; i++) {
            stale_host += host[i] != (value ^ i);
            stale_copy += copied[i] != (value ^ i);
        }
    }
    EXPECT(stale_host == 0, "%ld words of host memory written by kernels read stale after synchronizing", stale_host);
    EXPECT(stale_copy == 0, "%ld words copied to the host read stale", stale_copy);

    CHECK(cudaEventDestroy(written));
    CHECK(cudaEventDestroy(checked));
    CHECK(cudaStreamDestroy(a));
    CHECK(cudaStreamDestroy(b));
    CHECK(cudaFreeHost(host));
    CHECK(cudaFreeHost(copied));
    CHECK(cudaFree(device));
    CHECK(cudaFree(mismatches));
    PASS("memory_visibility");
    return 0;
}
