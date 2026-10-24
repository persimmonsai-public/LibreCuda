// Streams and events: pinned memory async copies overlapping with kernels on several streams,
// cross-stream dependencies with cudaStreamWaitEvent, legacy default stream implicit synchronization,
// event timing / queries, non-blocking streams, host callbacks, events recorded again before their previous record
// completed, and many events alive at once
#include "check.h"
#include <chrono>
#include <vector>
#include <atomic>

__global__ void scaleAdd(float *data, size_t n, float scale, float add, int spin) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        for (int k = 0; k < spin; k++) v = v * 1.0000001f + 1e-7f; // make the kernel take a while
        data[i] = v * scale + add - (spin > 0 ? (v - data[i]) : 0.0f);
    }
}

__global__ void addArrays(float *dst, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

__global__ void spinKernel(unsigned long long cycles, int *flag) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {}
    *flag = 1;
}

static void CUDART_CB hostCallback(void *userData) {
    static_cast<std::atomic<int> *>(userData)->fetch_add(1);
}

int main() {
    const int nstreams = 4;
    const size_t chunk = 4 << 20; // floats per stream
    float *h;
    CHECK(cudaMallocHost(&h, nstreams * chunk * sizeof(float)));
    for (size_t i = 0; i < nstreams * chunk; i++) h[i] = (float) (i % 1000);
    float *d;
    CHECK(cudaMalloc(&d, nstreams * chunk * sizeof(float)));

    cudaStream_t streams[nstreams];
    for (auto &s: streams) CHECK(cudaStreamCreate(&s));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // pipeline: H2D -> kernel -> D2H on each stream
    CHECK(cudaEventRecord(start));
    for (int s = 0; s < nstreams; s++) {
        float *hs = h + s * chunk, *ds = d + s * chunk;
        CHECK(cudaMemcpyAsync(ds, hs, chunk * sizeof(float), cudaMemcpyHostToDevice, streams[s]));
        scaleAdd<<<(chunk + 255) / 256, 256, 0, streams[s]>>>(ds, chunk, 2.0f, (float) s, 0);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpyAsync(hs, ds, chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop)); // legacy default stream: waits for the blocking streams
    for (int s = 0; s < nstreams; s++)
        for (size_t i = 0; i < chunk; i++) {
            float expect = (float) ((s * chunk + i) % 1000) * 2.0f + s;
            EXPECT(h[s * chunk + i] == expect, "stream %d element %zu = %f, expected %f", s, i, h[s * chunk + i], expect);
        }
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("4-stream pipeline of %zu MiB: %.3f ms (%.1f GB/s round trip)\n",
           nstreams * chunk * sizeof(float) >> 20, ms, 2.0 * nstreams * chunk * sizeof(float) / (ms * 1e6));
    EXPECT(ms > 0, "elapsed time %f", ms);

    // cross stream dependency: stream 1 consumes what stream 0 produces after a long kernel
    float *a, *b, *c;
    const size_t n = 1 << 20;
    CHECK(cudaMalloc(&a, n * 4)); CHECK(cudaMalloc(&b, n * 4)); CHECK(cudaMalloc(&c, n * 4));
    CHECK(cudaMemset(a, 0, n * 4));
    CHECK(cudaMemset(b, 0, n * 4));
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ones(n, 1.0f);
    cudaEvent_t produced;
    CHECK(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
    int *flag;
    CHECK(cudaMalloc(&flag, 4));
    CHECK(cudaMemset(flag, 0, 4));
    spinKernel<<<1, 1, 0, streams[0]>>>(200000000ull, flag); // ~0.1 s
    CHECK(cudaMemcpyAsync(a, ones.data(), n * 4, cudaMemcpyHostToDevice, streams[0])); // pageable: staged
    scaleAdd<<<(n + 255) / 256, 256, 0, streams[0]>>>(a, n, 3.0f, 0.0f, 0);
    CHECK(cudaEventRecord(produced, streams[0]));
    CHECK(cudaStreamWaitEvent(streams[1], produced, 0));
    addArrays<<<(n + 255) / 256, 256, 0, streams[1]>>>(c, a, a, n);
    std::vector<float> hc(n);
    CHECK(cudaMemcpyAsync(hc.data(), c, n * 4, cudaMemcpyDeviceToHost, streams[1]));
    CHECK(cudaStreamSynchronize(streams[1]));
    for (size_t i = 0; i < n; i++) EXPECT(hc[i] == 6.0f, "c[%zu] = %f", i, hc[i]);
    EXPECT(cudaEventQuery(produced) == cudaSuccess, "event should have completed");

    // legacy default stream ordering: a kernel on a blocking stream followed by cudaMemcpy on the default stream
    CHECK(cudaMemset(flag, 0, 4));
    spinKernel<<<1, 1, 0, streams[2]>>>(100000000ull, flag);
    int hflag = 0;
    CHECK(cudaMemcpy(&hflag, flag, 4, cudaMemcpyDeviceToHost));
    EXPECT(hflag == 1, "default stream did not wait for blocking stream");

    // non-blocking streams don't synchronize with the default stream, but are still stream ordered
    cudaStream_t nb;
    CHECK(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    CHECK(cudaMemsetAsync(flag, 0, 4, nb));
    spinKernel<<<1, 1, 0, nb>>>(50000000ull, flag);
    CHECK(cudaMemcpyAsync(&hflag, flag, 4, cudaMemcpyDeviceToHost, nb));
    CHECK(cudaStreamSynchronize(nb));
    EXPECT(hflag == 1, "non-blocking stream ordering");

    // query: a long running kernel is not ready
    spinKernel<<<1, 1, 0, streams[3]>>>(400000000ull, flag);
    EXPECT(cudaStreamQuery(streams[3]) == cudaErrorNotReady, "stream should be busy");
    CHECK(cudaStreamSynchronize(streams[3]));
    EXPECT(cudaStreamQuery(streams[3]) == cudaSuccess, "stream should be idle");

    // host functions run in stream order
    std::atomic<int> calls{0};
    CHECK(cudaLaunchHostFunc(streams[0], hostCallback, &calls));
    CHECK(cudaStreamSynchronize(streams[0]));
    EXPECT(calls.load() == 1, "host function calls %d", calls.load());

    // many streams & events
    std::vector<cudaStream_t> many(32);
    std::vector<cudaEvent_t> evs(32);
    for (int i = 0; i < 32; i++) { CHECK(cudaStreamCreate(&many[i])); CHECK(cudaEventCreate(&evs[i])); }
    for (int i = 0; i < 32; i++) {
        scaleAdd<<<(n + 255) / 256, 256, 0, many[i]>>>(a, n, 1.0f, 1.0f, 0);
        CHECK(cudaEventRecord(evs[i], many[i]));
        if (i + 1 < 32) CHECK(cudaStreamWaitEvent(many[i + 1], evs[i], 0));
    }
    CHECK(cudaEventSynchronize(evs[31]));
    std::vector<float> ha(n);
    CHECK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) EXPECT(ha[i] == 3.0f + 32.0f, "chained a[%zu] = %f", i, ha[i]);
    for (int i = 0; i < 32; i++) { CHECK(cudaEventDestroy(evs[i])); CHECK(cudaStreamDestroy(many[i])); }

    for (auto &s: streams) CHECK(cudaStreamDestroy(s));
    CHECK(cudaStreamDestroy(nb));
    CHECK(cudaFreeHost(h));
    // an event recorded on a busy stream, then again on an idle one: the second record completes first, and the first
    // one completing later must not undo it (waits for the event refer to its latest record)
    {
        float *slow;
        CHECK(cudaMalloc(&slow, 1024 * sizeof(float)));
        cudaStream_t busy, idle, waiter;
        CHECK(cudaStreamCreateWithFlags(&busy, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&idle, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&waiter, cudaStreamNonBlocking));
        cudaEvent_t e;
        CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (int round = 0; round < 20; round++) {
            scaleAdd<<<4, 256, 0, busy>>>(slow, 1024, 1.0f, 0.0f, 200000);
            CHECK(cudaEventRecord(e, busy));
            CHECK(cudaEventRecord(e, idle));
            CHECK(cudaStreamWaitEvent(waiter, e, 0));
            CHECK(cudaStreamSynchronize(idle));
            CHECK(cudaStreamSynchronize(busy));
            CHECK(cudaStreamSynchronize(waiter));
            EXPECT(cudaEventQuery(e) == cudaSuccess, "round %d: event re-recorded on an idle stream not complete", round);
            CHECK(cudaEventSynchronize(e));
        }
        CHECK(cudaEventDestroy(e));
        CHECK(cudaStreamDestroy(busy));
        CHECK(cudaStreamDestroy(idle));
        CHECK(cudaStreamDestroy(waiter));
        CHECK(cudaFree(slow));
    }

    // many events alive at once (PyTorch's NCCL backend keeps two per collective until its watchdog destroys them):
    // more than a context's first block of event signals
    {
        std::vector<cudaEvent_t> many(10000);
        for (auto &e : many) {
            CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            CHECK(cudaEventRecord(e, 0));
        }
        CHECK(cudaEventSynchronize(many.back()));
        for (auto &e : many) {
            EXPECT(cudaEventQuery(e) == cudaSuccess, "event not complete");
            CHECK(cudaEventDestroy(e));
        }
    }
    // destroyed while their records are still pending (behind a long kernel): each destruction costs about the same
    // (it once queried every earlier pending one: seconds for these)
    {
        float *slow;
        CHECK(cudaMalloc(&slow, 1024 * sizeof(float)));
        std::vector<cudaEvent_t> many(10000);
        for (auto &e : many) CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        scaleAdd<<<4, 256>>>(slow, 1024, 1.0f, 0.0f, 2000000);
        for (auto &e : many) CHECK(cudaEventRecord(e, 0));
        const auto start = std::chrono::steady_clock::now();
        for (auto &e : many) CHECK(cudaEventDestroy(e));
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        EXPECT(seconds < 0.5, "destroying 10000 pending events took %.2f s", seconds);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(slow));
        printf("    destroying 10000 pending events: %.2f us each\n", seconds * 1e6 / many.size());
    }

    PASS("streams_events");
    return 0;
}
