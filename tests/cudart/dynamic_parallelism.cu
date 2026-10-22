// NVCC_FLAGS: -arch=$ARCH -rdc=true
// Dynamic parallelism: kernels launching kernels (nvcc links NVIDIA's libcudadevrt; LibreCUDA redirects the device
// runtime calls to its own). Children per block, nesting, stream order of siblings, tail launches, work after the
// parent on the host stream, device side cudaMalloc / cudaGetLastError, the pending launch limit, device side
// cudaMemcpyAsync / cudaMemsetAsync (1D, 2D, 3D) and events, ordered with the launches.
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

// fills, copies and checks in device side stream order: src = 7s, then dst (pitched) = src, then a child checks
__global__ void checkCopies(const unsigned char *dst, size_t pitch, int width, int height, const unsigned char *cube,
                            int *bad) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (dst[y * pitch + x] != 7) {
                atomicAdd(bad, 1);
            }
        }
        if (dst[y * pitch + width] != 0) {
            atomicAdd(bad, 1); // the padding must stay untouched
        }
    }
    for (int i = 0; i < 4 * 4 * 4; i++) {
        if (cube[i] != ((i / 16 == 1 || i / 16 == 2) && (i / 4 % 4 == 1 || i / 4 % 4 == 2) && (i % 4 == 1 || i % 4 == 2)
                            ? 9 : 3)) {
            atomicAdd(bad, 1);
        }
    }
}

// make_cudaPitchedPtr & co. are host functions
__device__ cudaPitchedPtr pitched(void *p, size_t pitch, size_t xsize, size_t ysize) {
    cudaPitchedPtr r;
    r.ptr = p;
    r.pitch = pitch;
    r.xsize = xsize;
    r.ysize = ysize;
    return r;
}

__device__ cudaExtent extent(size_t w, size_t h, size_t d) {
    cudaExtent r;
    r.width = w;
    r.height = h;
    r.depth = d;
    return r;
}

__device__ cudaPos position(size_t x, size_t y, size_t z) {
    cudaPos r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

__global__ void copyParent(unsigned char *src, unsigned char *dst, size_t pitch, int width, int height,
                           unsigned char *cube, unsigned char *cube_src, int *bad) {
    if (threadIdx.x != 0) {
        return;
    }
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    cudaMemsetAsync(src, 7, width * height, s);
    cudaMemset2DAsync(dst, pitch, 0, width + 1, height, s);
    cudaEventRecord(e, s);
    cudaStreamWaitEvent(s, e, 0);
    cudaMemcpy2DAsync(dst, pitch, src, width, width, height, cudaMemcpyDeviceToDevice, s);
    // a 4x4x4 cube of 3s with the inner 2x2x2 copied from a cube of 9s
    cudaMemset3DAsync(pitched(cube, 4, 4, 4), 3, extent(4, 4, 4), s);
    cudaMemsetAsync(cube_src, 9, 64, s);
    cudaMemcpy3DParms p = {};
    p.srcPtr = pitched(cube_src, 4, 4, 4);
    p.dstPtr = pitched(cube, 4, 4, 4);
    p.srcPos = position(1, 1, 1);
    p.dstPos = position(1, 1, 1);
    p.extent = extent(2, 2, 2);
    p.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3DAsync(&p, s);
    checkCopies<<<1, 1, 0, s>>>(dst, pitch, width, height, cube, bad);
    cudaEventDestroy(e);
    cudaStreamDestroy(s);
    if (cudaGetLastError() != cudaSuccess) {
        atomicAdd(bad, 1000);
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
        EXPECT(host[0] == 2048 && host[1] == 16 * 256 - 2048, "%d children ran, %d failed launches seen", host[0],
               host[1]);
        CHECK(cudaFree(counts));
    }

    // device side copies, fills and events
    {
        const int width = 100, height = 20;
        const size_t pitch = 128;
        unsigned char *src, *dst, *cube, *cube_src;
        int *bad;
        CHECK(cudaMalloc(&src, width * height));
        CHECK(cudaMalloc(&dst, pitch * height));
        CHECK(cudaMalloc(&cube, 64));
        CHECK(cudaMalloc(&cube_src, 64));
        CHECK(cudaMalloc(&bad, sizeof(int)));
        CHECK(cudaMemset(dst, 0x55, pitch * height));
        CHECK(cudaMemset(bad, 0, sizeof(int)));
        copyParent<<<1, 32>>>(src, dst, pitch, width, height, cube, cube_src, bad);
        int host_bad = -1;
        CHECK(cudaMemcpy(&host_bad, bad, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT(host_bad == 0, "%d wrong bytes after device side copies", host_bad);
        std::vector<unsigned char> host(pitch * height);
        CHECK(cudaMemcpy(host.data(), dst, host.size(), cudaMemcpyDeviceToHost));
        EXPECT(host[width + 1] == 0x55 && host[pitch - 1] == 0x55, "the 2D fill wrote past its width");
        CHECK(cudaFree(src));
        CHECK(cudaFree(dst));
        CHECK(cudaFree(cube));
        CHECK(cudaFree(cube_src));
        CHECK(cudaFree(bad));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("dynamic_parallelism");
    return 0;
}
