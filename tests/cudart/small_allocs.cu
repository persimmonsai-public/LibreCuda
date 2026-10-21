// Many small cudaMallocs of mixed sizes: allocations must not overlap, stay 256-byte aligned and survive frees and
// reallocations of their neighbors (LibreCUDA serves them from a pool).
#include "check.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

__global__ void fillTag(unsigned *p, size_t n, unsigned tag) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        p[i] = tag ^ unsigned(i);
    }
}

__global__ void checkTag(const unsigned *p, size_t n, unsigned tag, unsigned *errors) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        if (p[i] != (tag ^ unsigned(i))) {
            atomicAdd(errors, 1u);
        }
    }
}

struct Alloc {
    unsigned *ptr;
    size_t bytes;
    unsigned tag;
};

int main() {
    std::mt19937 rng(1234);
    unsigned *errors;
    CHECK(cudaMalloc(&errors, sizeof(unsigned)));
    CHECK(cudaMemset(errors, 0, sizeof(unsigned)));

    std::vector<Alloc> live;
    unsigned next_tag = 1;
    auto allocate = [&]() {
        // sizes from 4 bytes up to 768 KiB (crossing the pool's largest size class)
        const size_t bytes = std::max<size_t>(4, (size_t(1) << (2 + rng() % 18)) + (rng() % 4096) * 4) & ~size_t(3);
        Alloc a{nullptr, bytes, next_tag++ * 2654435761u};
        CHECK(cudaMalloc(&a.ptr, bytes));
        EXPECT(reinterpret_cast<uintptr_t>(a.ptr) % 256 == 0, "allocation %p not 256-byte aligned", a.ptr);
        fillTag<<<32, 256>>>(a.ptr, bytes / 4, a.tag);
        live.push_back(a);
    };

    for (int i = 0; i < 2000; i++) {
        allocate();
    }
    for (int round = 0; round < 4; round++) {
        // free a random half, then refill
        std::shuffle(live.begin(), live.end(), rng);
        for (size_t i = 0; i < live.size() / 2; i++) {
            CHECK(cudaFree(live[i].ptr));
        }
        live.erase(live.begin(), live.begin() + live.size() / 2);
        for (int i = 0; i < 1000; i++) {
            allocate();
        }
    }

    // no two live allocations overlap
    std::vector<Alloc> sorted = live;
    std::sort(sorted.begin(), sorted.end(), [](const Alloc &a, const Alloc &b) { return a.ptr < b.ptr; });
    for (size_t i = 1; i < sorted.size(); i++) {
        const char *prev_end = reinterpret_cast<const char *>(sorted[i - 1].ptr) + sorted[i - 1].bytes;
        EXPECT(prev_end <= reinterpret_cast<const char *>(sorted[i].ptr), "allocations %p and %p overlap",
               sorted[i - 1].ptr, sorted[i].ptr);
    }

    // every allocation still holds its own data
    for (const Alloc &a: live) {
        checkTag<<<32, 256>>>(a.ptr, a.bytes / 4, a.tag, errors);
    }
    unsigned host_errors = 0;
    CHECK(cudaMemcpy(&host_errors, errors, sizeof(unsigned), cudaMemcpyDeviceToHost));
    EXPECT(host_errors == 0, "%u corrupted words", host_errors);

    // pointer attributes inside a small allocation
    cudaPointerAttributes attr;
    CHECK(cudaPointerGetAttributes(&attr, live[0].ptr + 1));
    EXPECT(attr.type == cudaMemoryTypeDevice, "small allocation is not device memory");

    // freeing an address inside an allocation fails
    {
        const Alloc &largest = *std::max_element(live.begin(), live.end(), [](const Alloc &a, const Alloc &b) {
            return a.bytes < b.bytes;
        });
        EXPECT(cudaFree(reinterpret_cast<char *>(largest.ptr) + 256) != cudaSuccess, "freeing an interior pointer succeeded");
        cudaGetLastError();
    }

    for (const Alloc &a: live) {
        CHECK(cudaFree(a.ptr));
    }

    // allocation latency
    {
        std::vector<void *> ptrs(1000);
        auto start = std::chrono::steady_clock::now();
        for (auto &p: ptrs) {
            CHECK(cudaMalloc(&p, 1024));
        }
        auto mid = std::chrono::steady_clock::now();
        for (auto &p: ptrs) {
            CHECK(cudaFree(p));
        }
        auto end = std::chrono::steady_clock::now();
        printf("    1 KiB cudaMalloc: %.2f us, cudaFree: %.2f us\n",
               std::chrono::duration<double, std::micro>(mid - start).count() / ptrs.size(),
               std::chrono::duration<double, std::micro>(end - mid).count() / ptrs.size());
    }
    CHECK(cudaFree(errors));
    PASS("small_allocs");
    return 0;
}
