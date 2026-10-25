// Event records without a timestamp that need no work of the GPU: recorded on an idle stream (reached at once) and
// recorded again with no work in between (the same point of the stream), against records that need the GPU (work
// enqueued since, another stream still busy); and constant memory rewritten between launches (launches invalidate the
// shader caches only after modules were loaded, every grid its constant banks)
#include "check.h"

__constant__ int c_value;

__global__ void spinKernel(unsigned long long cycles, volatile int *flag) {
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {}
    *flag = 1;
}

__global__ void readConstant(int *out, int slot) {
    out[slot] = c_value;
}

__global__ void writeValue(int *out, int value) {
    *out = value;
}

int main() {
    int *flags;
    CHECK(cudaMallocHost(&flags, 4 * sizeof(int)));  // (read by the host while kernels run)
    cudaStream_t a, b;
    CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    cudaEvent_t e;
    CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    const unsigned long long long_spin = 400ull * 1000 * 1000;  // a few hundred ms

    // recorded on an idle stream: reached at once
    CHECK(cudaStreamSynchronize(a));
    CHECK(cudaEventRecord(e, a));
    EXPECT(cudaEventQuery(e) == cudaSuccess, "event recorded on an idle stream not reached");
    for (int i = 0; i < 10000; i++) {
        CHECK(cudaEventRecord(e, a));
        EXPECT(cudaEventQuery(e) == cudaSuccess, "event recorded on an idle stream not reached (record %d)", i);
    }

    // recorded behind a kernel: not reached until the kernel completed, also when recorded again with nothing in between
    flags[0] = 0;
    spinKernel<<<1, 1, 0, a>>>(long_spin, flags);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e, a));
    EXPECT(cudaEventQuery(e) == cudaErrorNotReady, "event recorded behind a running kernel reached");
    for (int i = 0; i < 1000; i++) {
        CHECK(cudaEventRecord(e, a));
    }
    EXPECT(cudaEventQuery(e) == cudaErrorNotReady, "event recorded again behind a running kernel reached");
    CHECK(cudaEventSynchronize(e));
    EXPECT(flags[0] == 1, "cudaEventSynchronize returned before the kernel completed");

    // reached, then recorded again behind new work: not reached until that work completed
    flags[0] = 0;
    spinKernel<<<1, 1, 0, a>>>(long_spin, flags);
    CHECK(cudaEventRecord(e, a));
    EXPECT(cudaEventQuery(e) == cudaErrorNotReady, "event recorded behind new work reached");
    // a stream waiting for it runs after the kernel
    writeValue<<<1, 1, 0, b>>>(flags + 1, 0);
    CHECK(cudaStreamWaitEvent(b, e, 0));
    readConstant<<<1, 1, 0, b>>>(flags, 2);  // (any work)
    writeValue<<<1, 1, 0, b>>>(flags + 1, 7);
    CHECK(cudaStreamSynchronize(b));
    EXPECT(flags[0] == 1 && flags[1] == 7, "work waiting for the event ran before the kernel (flag %d)", flags[0]);

    // recorded on a busy stream, then on an idle one: stands for the idle stream's point, reached while the first
    // stream still runs
    flags[0] = 0;
    spinKernel<<<1, 1, 0, a>>>(long_spin, flags);
    CHECK(cudaEventRecord(e, a));
    CHECK(cudaStreamSynchronize(b));
    CHECK(cudaEventRecord(e, b));
    CHECK(cudaEventSynchronize(e));
    EXPECT(flags[0] == 0, "event recorded on an idle stream waited for another stream's kernel");
    // and back on the busy stream: not reached until its kernel completed
    CHECK(cudaEventRecord(e, a));
    EXPECT(cudaEventQuery(e) == cudaErrorNotReady || flags[0] == 1, "event recorded on the busy stream again reached");
    CHECK(cudaEventSynchronize(e));
    EXPECT(flags[0] == 1, "cudaEventSynchronize returned before the kernel completed");

    // timing events are recorded by the GPU, also on an idle stream
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaStreamSynchronize(a));
    CHECK(cudaEventRecord(t0, a));
    spinKernel<<<1, 1, 0, a>>>(20ull * 1000 * 1000, flags);
    CHECK(cudaEventRecord(t1, a));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    EXPECT(ms > 0.5f, "elapsed time %f ms across a spinning kernel", ms);

    // constant memory rewritten between launches
    for (int i = 0; i < 100; i++) {
        CHECK(cudaMemcpyToSymbolAsync(c_value, &i, sizeof(int), 0, cudaMemcpyHostToDevice, a));
        readConstant<<<1, 1, 0, a>>>(flags, 3);
        CHECK(cudaStreamSynchronize(a));
        EXPECT(flags[3] == i, "kernel read %d from constant memory, %d written", flags[3], i);
    }

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaEventDestroy(e));
    CHECK(cudaStreamDestroy(a));
    CHECK(cudaStreamDestroy(b));
    CHECK(cudaFreeHost(flags));
    PASS("event_records");
    return 0;
}
