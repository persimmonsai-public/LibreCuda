// Device side malloc / free: concurrent allocations from many threads, memory passed between kernels, exhaustion of
// the heap (cudaLimitMallocHeapSize) and reuse of freed memory.
#include "check.h"

// every thread repeatedly allocates, fills, checks and frees blocks of varying size
__global__ void churn(int rounds, unsigned *errors, unsigned *nulls) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < rounds; r++) {
        const unsigned n = 4 + ((tid + r) % 7) * 37;
        unsigned *p = static_cast<unsigned *>(malloc(n * sizeof(unsigned)));
        if (p == nullptr) {
            atomicAdd(nulls, 1u);
            continue;
        }
        if (reinterpret_cast<unsigned long long>(p) % 16 != 0) {
            atomicAdd(errors, 1000u);
        }
        for (unsigned i = 0; i < n; i++) {
            p[i] = tid * 131 + i + r;
        }
        for (unsigned i = 0; i < n; i++) {
            if (p[i] != tid * 131 + i + r) {
                atomicAdd(errors, 1u);
                break;
            }
        }
        free(p);
    }
}

// all threads use the same size class at once (the most contended free list)
__global__ void sameSize(int rounds, unsigned *errors) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < rounds; r++) {
        unsigned long long *p = static_cast<unsigned long long *>(malloc(16));
        if (p == nullptr) {
            atomicAdd(errors, 1u);
            continue;
        }
        p[0] = tid;
        p[1] = ~static_cast<unsigned long long>(tid);
        __syncwarp();
        if (p[0] != tid || p[1] != ~static_cast<unsigned long long>(tid)) {
            atomicAdd(errors, 1u);
        }
        free(p);
    }
}

__global__ void allocate(int **pointers, int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int *p = static_cast<int *>(malloc(count * sizeof(int)));
    if (p != nullptr) {
        for (int i = 0; i < count; i++) {
            p[i] = tid + i;
        }
    }
    pointers[tid] = p;
}

__global__ void checkAndFree(int **pointers, int count, unsigned *errors) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int *p = pointers[tid];
    if (p == nullptr) {
        atomicAdd(errors, 1u);
        return;
    }
    for (int i = 0; i < count; i++) {
        if (p[i] != tid + i) {
            atomicAdd(errors, 1u);
            break;
        }
    }
    free(p);
}

// allocates blocks of size bytes until malloc fails; returns the count and keeps the blocks in pointers
__global__ void exhaust(size_t size, void **pointers, int max, int *count) {
    int n = 0;
    while (n < max) {
        void *p = malloc(size);
        if (p == nullptr) {
            break;
        }
        pointers[n++] = p;
    }
    *count = n;
}

__global__ void freeAll(void **pointers, const int *count) {
    for (int i = 0; i < *count; i++) {
        free(pointers[i]);
    }
}

__global__ void edgeCases(unsigned *errors) {
    if (malloc(0) != nullptr) {
        atomicAdd(errors, 1u);
    }
    free(nullptr);
    // larger than the heap
    if (malloc(size_t(1) << 40) != nullptr) {
        atomicAdd(errors, 1u);
    }
}

int main() {
    size_t heap = 0;
    CHECK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
    EXPECT(heap == (8 << 20), "default heap size %zu", heap);
    const size_t heap_size = 16 << 20;
    // before the first kernel that calls malloc
    CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CHECK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
    EXPECT(heap == heap_size, "heap size %zu after setting %zu", heap, heap_size);

    unsigned *counters;
    CHECK(cudaMalloc(&counters, 4 * sizeof(unsigned)));
    CHECK(cudaMemset(counters, 0, 4 * sizeof(unsigned)));

    churn<<<64, 256>>>(20, counters, counters + 1);
    sameSize<<<64, 256>>>(50, counters + 2);
    edgeCases<<<1, 1>>>(counters + 3);
    unsigned host[4];
    CHECK(cudaMemcpy(host, counters, sizeof(host), cudaMemcpyDeviceToHost));
    EXPECT(host[0] == 0 && host[1] == 0, "churn: %u errors, %u failed allocations", host[0], host[1]);
    EXPECT(host[2] == 0, "same size class: %u errors", host[2]);
    EXPECT(host[3] == 0, "edge cases: %u errors", host[3]);
    // the heap can't change once used
    EXPECT(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size * 2) != cudaSuccess, "heap size changed after use");
    cudaGetLastError();

    // memory allocated in one kernel, used and freed in another
    {
        const int threads = 64 * 128, count = 33;
        int **pointers;
        CHECK(cudaMalloc(&pointers, threads * sizeof(int *)));
        CHECK(cudaMemset(counters, 0, sizeof(unsigned)));
        allocate<<<64, 128>>>(pointers, count);
        checkAndFree<<<64, 128>>>(pointers, count, counters);
        CHECK(cudaMemcpy(host, counters, sizeof(unsigned), cudaMemcpyDeviceToHost));
        EXPECT(host[0] == 0, "cross-kernel allocations: %u errors", host[0]);
        CHECK(cudaFree(pointers));
    }

    // exhaustion: 64 KiB blocks (64 KiB + header -> 128 KiB slots) until NULL, free them, and again
    {
        const int max = 1024;
        void **pointers;
        int *count;
        CHECK(cudaMalloc(&pointers, max * sizeof(void *)));
        CHECK(cudaMalloc(&count, sizeof(int)));
        int first = 0, second = 0;
        exhaust<<<1, 1>>>(64 << 10, pointers, max, count);
        CHECK(cudaMemcpy(&first, count, sizeof(int), cudaMemcpyDeviceToHost));
        freeAll<<<1, 1>>>(pointers, count);
        exhaust<<<1, 1>>>(64 << 10, pointers, max, count);
        CHECK(cudaMemcpy(&second, count, sizeof(int), cudaMemcpyDeviceToHost));
        freeAll<<<1, 1>>>(pointers, count);
        printf("heap of %zu MiB: %d blocks of 64 KiB before exhaustion, %d after freeing them\n", heap_size >> 20,
               first, second);
        EXPECT(first > 0 && first < max, "exhaustion after %d blocks", first);
        EXPECT(second >= first, "freed blocks not reused: %d then %d", first, second);
        CHECK(cudaFree(pointers));
        CHECK(cudaFree(count));
    }

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(counters));
    PASS("device_malloc");
    return 0;
}
