// NVCC_FLAGS: -arch=$ARCH -lcuda
// Stream memory operations (cuStreamWaitValue32/64, cuStreamWriteValue32/64, cuStreamBatchMemOp): a stream waits
// for a flag another stream writes after its kernel (device memory), for a flag the CPU writes (mapped host memory),
// the wait operations (GEQ cyclic, EQ, AND, NOR), and with two devices a stream of one device waiting for a flag the
// other device's stream writes into its memory (peer access). The writes are enqueued before the waits (while the
// writer's slow kernel still runs): NVIDIA's driver may deadlock when a stream waits for a write enqueued later in the
// same context. Checked against NVIDIA's driver too.
#include "check.h"
#include <cuda.h>
#include <chrono>
#include <thread>
#include <vector>

#define DRV(call) do { \
    CUresult drv_err = (call); \
    if (drv_err != CUDA_SUCCESS) { \
        const char *drv_name = nullptr; cuGetErrorName(drv_err, &drv_name); \
        fprintf(stderr, "driver error %s at %s:%d: %s\n", drv_name, __FILE__, __LINE__, #call); exit(1); \
    } \
} while (0)

__global__ void fillSlowly(int *p, int n, int base, long long cycles) {
    if (threadIdx.x == 0) {
        const long long start = clock64();
        while (clock64() - start < cycles) {
        }
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = base + i;
}

__global__ void copyTwice(const int *in, int *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = 2 * in[i];
}

static CUdeviceptr dptr(const void *p) {
    return reinterpret_cast<CUdeviceptr>(p);
}

static void expectDoubled(const int *out, int n, int base, const char *what) {
    std::vector<int> h(n);
    CHECK(cudaMemcpy(h.data(), out, n * sizeof(int), cudaMemcpyDefault));
    for (int i = 0; i < n; i++) EXPECT(h[i] == 2 * (base + i), "%s: [%d] = %d", what, i, h[i]);
}

// the wait / write order between streams through a flag in device memory
static void betweenStreams() {
    const int n = 1 << 20;
    int *data, *out;
    uint64_t *flag;
    CHECK(cudaMalloc(&data, n * sizeof(int)));
    CHECK(cudaMalloc(&out, n * sizeof(int)));
    CHECK(cudaMalloc(&flag, 2 * sizeof(uint64_t)));
    CHECK(cudaMemset(flag, 0, 2 * sizeof(uint64_t)));
    cudaStream_t producer, consumer;
    CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    for (int round = 1; round <= 3; round++) {
        // the wait holds the copy back until the slow kernel finished
        fillSlowly<<<64, 256, 0, producer>>>(data, n, round * 1000, 20000000LL);
        DRV(cuStreamWriteValue64(producer, dptr(flag), round, 0));
        DRV(cuStreamWaitValue64(consumer, dptr(flag), round, CU_STREAM_WAIT_VALUE_GEQ));
        copyTwice<<<64, 256, 0, consumer>>>(data, out, n);
        CHECK(cudaStreamSynchronize(consumer));
        expectDoubled(out, n, round * 1000, "64 bit flag");
    }
    // 32 bit flags in the same buffer: the neighbouring word is left alone
    uint32_t *flag32 = reinterpret_cast<uint32_t *>(flag + 1);
    fillSlowly<<<64, 256, 0, producer>>>(data, n, 77, 20000000LL);
    DRV(cuStreamWriteValue32(producer, dptr(flag32), 0x70000000u, 0));
    DRV(cuStreamWaitValue32(consumer, dptr(flag32), 0x70000000u, CU_STREAM_WAIT_VALUE_EQ));
    copyTwice<<<64, 256, 0, consumer>>>(data, out, n);
    CHECK(cudaStreamSynchronize(consumer));
    expectDoubled(out, n, 77, "32 bit flag");
    uint64_t h[2];
    CHECK(cudaMemcpy(h, flag, sizeof(h), cudaMemcpyDeviceToHost));
    EXPECT(h[0] == 3 && h[1] == 0x70000000u, "flags %llx %llx", (unsigned long long) h[0], (unsigned long long) h[1]);
    CHECK(cudaStreamDestroy(producer));
    CHECK(cudaStreamDestroy(consumer));
    CHECK(cudaFree(data));
    CHECK(cudaFree(out));
    CHECK(cudaFree(flag));
    PASS("streams ordered through a flag in device memory");
}

// waits for values the CPU writes into mapped host memory: GEQ (cyclic), EQ, AND, NOR
static void hostFlags() {
    volatile uint32_t *flags;
    CHECK(cudaHostAlloc(const_cast<uint32_t **>(&flags), 4 * sizeof(uint32_t), cudaHostAllocMapped));
    uint32_t *device;
    CHECK(cudaHostGetDevicePointer(&device, const_cast<uint32_t *>(flags), 0));
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    struct Case {
        unsigned int flags;
        uint32_t initial, wanted, notYet, finally;
        const char *name;
    } cases[] = {
        {CU_STREAM_WAIT_VALUE_GEQ, 0xfffffff0u, 0xfffffffau, 0xfffffff5u, 0x00000003u, "GEQ across the wrap"},
        {CU_STREAM_WAIT_VALUE_EQ, 0, 42, 41, 42, "EQ"},
        {CU_STREAM_WAIT_VALUE_AND, 0, 0x10, 0x0f, 0x30, "AND"},
        {CU_STREAM_WAIT_VALUE_NOR, 0xffffffffu, 0x0000ffffu, 0xffffffffu, 0xfffe0000u, "NOR"},
    };
    for (const Case &c: cases) {
        flags[0] = c.initial;
        DRV(cuStreamWaitValue32(s, dptr(device), c.wanted, c.flags));
        cudaEvent_t done;
        CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        CHECK(cudaEventRecord(done, s));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT(cudaEventQuery(done) == cudaErrorNotReady, "%s: passed before the CPU wrote", c.name);
        flags[0] = c.notYet;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT(cudaEventQuery(done) == cudaErrorNotReady, "%s: passed with 0x%x", c.name, c.notYet);
        flags[0] = c.finally;
        CHECK(cudaEventSynchronize(done));
        CHECK(cudaEventDestroy(done));
    }
    cudaGetLastError();
    // a batch: write two values, then wait for them
    CUstreamBatchMemOpParams ops[3] = {};
    ops[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[0].writeValue.address = dptr(device + 1);
    ops[0].writeValue.value = 11;
    ops[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[1].writeValue.address = dptr(device + 2);
    ops[1].writeValue.value = 12;
    ops[2].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    ops[2].waitValue.address = dptr(device + 2);
    ops[2].waitValue.value = 12;
    ops[2].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    DRV(cuStreamBatchMemOp(s, 3, ops, 0));
    CHECK(cudaStreamSynchronize(s));
    EXPECT(flags[1] == 11 && flags[2] == 12, "batch wrote %u %u", flags[1], flags[2]);
    CHECK(cudaStreamDestroy(s));
    CHECK(cudaFreeHost(const_cast<uint32_t *>(flags)));
    PASS("waits for values the CPU writes (GEQ, EQ, AND, NOR), batches");
}

// device 1's stream waits for a flag in its memory that device 0's stream writes after its kernel (peer access)
static void betweenDevices() {
    int count = 0, can01 = 0, can10 = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        PASS("between devices (skipped: one device)");
        return;
    }
    CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    if (!can01 || !can10) {
        PASS("between devices (skipped: no peer access)");
        return;
    }
    const int n = 1 << 20;
    int *data0, *out1;
    uint64_t *flag1;
    cudaStream_t s0, s1;
    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaMalloc(&data0, n * sizeof(int)));
    CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceEnablePeerAccess(0, 0));
    CHECK(cudaMalloc(&out1, n * sizeof(int)));
    CHECK(cudaMalloc(&flag1, sizeof(uint64_t)));
    CHECK(cudaMemset(flag1, 0, sizeof(uint64_t)));
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CHECK(cudaDeviceSynchronize());
    for (int round = 1; round <= 3; round++) {
        // device 1 reads device 0's memory once device 0's kernel wrote it
        CHECK(cudaSetDevice(0));
        fillSlowly<<<64, 256, 0, s0>>>(data0, n, round * 100, 20000000LL);
        DRV(cuStreamWriteValue64(s0, dptr(flag1), round, 0));
        CHECK(cudaSetDevice(1));
        DRV(cuStreamWaitValue64(s1, dptr(flag1), round, CU_STREAM_WAIT_VALUE_GEQ));
        copyTwice<<<64, 256, 0, s1>>>(data0, out1, n);
        CHECK(cudaStreamSynchronize(s1));
        expectDoubled(out1, n, round * 100, "between devices");
    }
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaFree(out1));
    CHECK(cudaFree(flag1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamDestroy(s0));
    CHECK(cudaFree(data0));
    PASS("a stream of one device waits for a flag the other device writes");
}

int main() {
    int attribute = 0;
    CHECK(cudaFree(nullptr));
    DRV(cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, 0));
    EXPECT(attribute == 1, "64 bit stream memory operations unsupported");
    DRV(cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, 0));
    EXPECT(attribute == 1, "NOR waits unsupported");
    betweenStreams();
    hostFlags();
    betweenDevices();
    return 0;
}
