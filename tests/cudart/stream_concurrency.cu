// Kernels on different streams run concurrently: single-block kernels that spin for a fixed time on separate
// streams must finish in about the time of one, while kernels on the same stream still run one after another.
#include "check.h"
#include <chrono>
#include <vector>

__global__ void spin(long long cycles, int *order, int *next, int id) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
    order[id] = atomicAdd(next, 1);
}

static double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main() {
    int *order, *next;
    CHECK(cudaMalloc(&order, 64 * sizeof(int)));
    CHECK(cudaMalloc(&next, sizeof(int)));

    // calibrate: cycles for about 20 ms of spinning
    long long cycles = 20'000'000;
    {
        CHECK(cudaMemset(next, 0, sizeof(int)));
        spin<<<1, 1>>>(cycles, order, next, 0);
        CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();
        spin<<<1, 1>>>(cycles, order, next, 0);
        CHECK(cudaDeviceSynchronize());
        cycles = static_cast<long long>(cycles * 20.0 / elapsedMs(start));
    }
    auto runOne = [&] {
        auto start = std::chrono::steady_clock::now();
        spin<<<1, 1>>>(cycles, order, next, 0);
        CHECK(cudaDeviceSynchronize());
        return elapsedMs(start);
    };
    const double single = runOne();

    const int n_streams = 4;
    std::vector<cudaStream_t> streams(n_streams);
    for (auto &s: streams) {
        CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }

    // one kernel per stream: should overlap
    CHECK(cudaMemset(next, 0, sizeof(int)));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_streams; i++) {
        spin<<<1, 1, 0, streams[i]>>>(cycles, order, next, i);
    }
    CHECK(cudaDeviceSynchronize());
    const double parallel = elapsedMs(start);

    // two kernels per stream: must stay ordered within each stream
    CHECK(cudaMemset(next, 0, sizeof(int)));
    for (int i = 0; i < n_streams; i++) {
        spin<<<1, 1, 0, streams[i]>>>(cycles, order, next, i);
        spin<<<1, 1, 0, streams[i]>>>(cycles, order, next, n_streams + i);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int> host(2 * n_streams);
    CHECK(cudaMemcpy(host.data(), order, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_streams; i++) {
        EXPECT(host[i] < host[n_streams + i], "stream %d: second kernel finished before the first (%d, %d)", i,
               host[i], host[n_streams + i]);
    }

    // an event recorded on one stream only waits for that stream's work
    cudaEvent_t done;
    CHECK(cudaEventCreate(&done));
    spin<<<1, 1, 0, streams[0]>>>(cycles * 4, order, next, 0);
    start = std::chrono::steady_clock::now();
    spin<<<1, 1, 0, streams[1]>>>(cycles, order, next, 1);
    CHECK(cudaEventRecord(done, streams[1]));
    CHECK(cudaEventSynchronize(done));
    const double independent = elapsedMs(start);
    CHECK(cudaDeviceSynchronize());

    printf("one kernel: %.1f ms; %d streams x 1 kernel: %.1f ms (%.2fx); short stream next to a 4x longer one: %.1f ms\n",
           single, n_streams, parallel, n_streams * single / parallel, independent);
    EXPECT(parallel < 0.5 * n_streams * single, "kernels on different streams did not overlap");
    EXPECT(independent < 2.5 * single, "an event waited for another stream's work");

    for (auto &s: streams) {
        CHECK(cudaStreamDestroy(s));
    }
    CHECK(cudaEventDestroy(done));
    PASS("stream_concurrency");
    return 0;
}
