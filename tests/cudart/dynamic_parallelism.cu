// NVCC_FLAGS: -arch=$ARCH -rdc=true
// Dynamic parallelism: kernels launching kernels (nvcc links NVIDIA's libcudadevrt; LibreCUDA redirects the device
// runtime calls to its own). Children per block, nesting, stream order of siblings, tail launches, work after the
// parent on the host stream, device side cudaMalloc / cudaGetLastError and the pending launch limit.
#include "check.h"
#include <vector>

__global__ void child(int *out, int base, int value) {
    out[base + threadIdx.x] = value + threadIdx.x;
}

__global__ void parentPerBlock(int *out) {
    if (threadIdx.x == 0) {
        child<<<1, 32>>>(out, blockIdx.x * 32, blockIdx.x * 1000);
        if (cudaGetLastError() != cudaSuccess) {
            out[blockIdx.x * 32] = -1;
        }
    }
}

// level n launches level n + 1 until depth; every level records itself
__global__ void nested(int *levels, int level, int depth) {
    if (threadIdx.x == 0) {
        levels[level] = level + 100;
        if (level + 1 < depth) {
            nested<<<1, 1>>>(levels, level + 1, depth);
        }
    }
}

__global__ void addOne(int *p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] += 1;
    }
}

__global__ void doubleIt(int *p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] *= 2;
    }
}

__global__ void sum(const int *p, int n, int *result) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += p[i];
    }
    *result = s;
}

// siblings in one stream run in order; the tail launch runs after them
__global__ void ordered(int *p, int n, int *result) {
    if (threadIdx.x == 0) {
        sum<<<1, 1, 0, cudaStreamTailLaunch>>>(p, n, result);
        addOne<<<(n + 127) / 128, 128>>>(p, n);
        doubleIt<<<(n + 127) / 128, 128>>>(p, n);
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        addOne<<<(n + 127) / 128, 128, 0, s>>>(p, n);
        cudaStreamDestroy(s);
    }
}

// device side cudaMalloc / cudaFree and dynamic shared memory of a child
__global__ void sharedChild(int *out, int value) {
    extern __shared__ int buffer[];
    buffer[threadIdx.x] = value * threadIdx.x;
    __syncthreads();
    out[threadIdx.x] = buffer[blockDim.x - 1 - threadIdx.x];
}

__global__ void scratchParent(int *out) {
    if (threadIdx.x == 0) {
        int *scratch = nullptr;
        if (cudaMalloc(reinterpret_cast<void **>(&scratch), 64 * sizeof(int)) != cudaSuccess) {
            out[0] = -1;
            return;
        }
        sharedChild<<<1, 64, 64 * sizeof(int)>>>(out, 3);
        cudaFree(scratch);
        int device = -1;
        cudaGetDevice(&device);
        out[64] = device;
    }
}

// every thread launches a child: more than the pending launch limit
__global__ void counter(int *count) {
    atomicAdd(count, 1);
}

__global__ void flood(int *count, int *failures) {
    counter<<<1, 1>>>(count);
    if (cudaGetLastError() == cudaErrorLaunchPendingCountExceeded) {
        atomicAdd(failures, 1);
    }
}

int main() {
    // children per block, then a host kernel on the stream sees their results
    {
        const int blocks = 8;
        int *out;
        CHECK(cudaMalloc(&out, blocks * 32 * sizeof(int)));
        CHECK(cudaMemset(out, 0, blocks * 32 * sizeof(int)));
        parentPerBlock<<<blocks, 32>>>(out);
        CHECK(cudaGetLastError());
        addOne<<<blocks, 32>>>(out, blocks * 32);
        std::vector<int> host(blocks * 32);
        CHECK(cudaMemcpy(host.data(), out, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < blocks; b++) {
            for (int t = 0; t < 32; t++) {
                const int expected = b * 1000 + t + 1;
                EXPECT(host[b * 32 + t] == expected, "child [%d][%d] = %d, expected %d", b, t, host[b * 32 + t],
                       expected);
            }
        }
        CHECK(cudaFree(out));
    }

    // nesting
    {
        const int depth = 6;
        int *levels;
        CHECK(cudaMalloc(&levels, depth * sizeof(int)));
        CHECK(cudaMemset(levels, 0, depth * sizeof(int)));
        nested<<<1, 32>>>(levels, 0, depth);
        int host[depth];
        CHECK(cudaMemcpy(host, levels, sizeof(host), cudaMemcpyDeviceToHost));
        for (int i = 0; i < depth; i++) {
            EXPECT(host[i] == i + 100, "level %d = %d", i, host[i]);
        }
        CHECK(cudaFree(levels));
    }

    // order of siblings and the tail launch: ((1 + 1) * 2 + 1) per element
    {
        const int n = 1000;
        int *p, *result;
        CHECK(cudaMalloc(&p, n * sizeof(int)));
        CHECK(cudaMalloc(&result, sizeof(int)));
        std::vector<int> ones(n, 1);
        CHECK(cudaMemcpy(p, ones.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        ordered<<<1, 32>>>(p, n, result);
        int host_result = 0;
        CHECK(cudaMemcpy(&host_result, result, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT(host_result == 5 * n, "tail launch sum %d, expected %d", host_result, 5 * n);
        CHECK(cudaFree(p));
        CHECK(cudaFree(result));
    }

    // device side cudaMalloc, dynamic shared memory, cudaGetDevice
    {
        int *out;
        CHECK(cudaMalloc(&out, 65 * sizeof(int)));
        scratchParent<<<1, 1>>>(out);
        int host[65];
        CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
        for (int t = 0; t < 64; t++) {
            EXPECT(host[t] == 3 * (63 - t), "shared child [%d] = %d", t, host[t]);
        }
        EXPECT(host[64] == 0, "device side cudaGetDevice = %d", host[64]);
        CHECK(cudaFree(out));
    }

    // the pending launch limit (default 2048)
    {
        size_t limit = 0;
        CHECK(cudaDeviceGetLimit(&limit, cudaLimitDevRuntimePendingLaunchCount));
        EXPECT(limit == 2048, "pending launch limit %zu", limit);
        int *counts;
        CHECK(cudaMalloc(&counts, 2 * sizeof(int)));
        CHECK(cudaMemset(counts, 0, 2 * sizeof(int)));
        flood<<<16, 256>>>(counts, counts + 1);
        int host[2];
        CHECK(cudaMemcpy(host, counts, sizeof(host), cudaMemcpyDeviceToHost));
        // the device side last error is shared by all threads in LibreCUDA, so only some failures may be seen
        EXPECT(host[0] == 2048 && host[1] >= 1, "%d children ran, %d failed launches seen", host[0], host[1]);
        CHECK(cudaFree(counts));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("dynamic_parallelism");
    return 0;
}
