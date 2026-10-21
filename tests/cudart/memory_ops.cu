// Memory operations: memset with odd sizes/alignments, D2D copies (overlapping offsets, unaligned), pinned and
// pageable round trips of many sizes, 2D copies & memset with pitch, managed memory, pointer attributes, mem info,
// many malloc/free cycles
#include "check.h"
#include <vector>
#include <cstring>
#include <random>

__global__ void fill(unsigned char *p, size_t n, unsigned char v) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void incrementManaged(int *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += i;
}

int main() {
    std::mt19937 rng(11);
    size_t free0 = 0, total = 0;
    CHECK(cudaMemGetInfo(&free0, &total));
    printf("memory: %.1f MiB free of %.1f MiB\n", free0 / 1048576.0, total / 1048576.0);
    EXPECT(total > 0 && free0 <= total, "mem info");

    // memset of odd sizes / alignments
    const size_t cap = 1 << 22;
    unsigned char *d;
    CHECK(cudaMalloc(&d, cap));
    std::vector<unsigned char> h(cap), expect(cap, 0);
    CHECK(cudaMemset(d, 0, cap));
    for (int iter = 0; iter < 50; iter++) {
        size_t off = rng() % 4096, len = rng() % (cap - off);
        if (iter < 10) len = iter; // tiny
        unsigned char v = (unsigned char) rng();
        CHECK(cudaMemset(d + off, v, len));
        memset(expect.data() + off, v, len);
    }
    CHECK(cudaMemcpy(h.data(), d, cap, cudaMemcpyDeviceToHost));
    EXPECT(h == expect, "memset mismatch");

    // device to device copies with arbitrary offsets
    unsigned char *d2;
    CHECK(cudaMalloc(&d2, cap));
    for (int iter = 0; iter < 30; iter++) {
        size_t src = rng() % (cap / 2), dst = rng() % (cap / 2), len = rng() % (cap / 2);
        CHECK(cudaMemcpy(d2 + dst, d + src, len, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(d + dst, d2 + dst, len, cudaMemcpyDefault));
        memmove(expect.data() + dst, expect.data() + src, len);
    }
    CHECK(cudaMemcpy(h.data(), d, cap, cudaMemcpyDeviceToHost));
    EXPECT(h == expect, "D2D mismatch");

    // round trips through pageable and pinned memory, sizes from 1 byte to 64 MiB
    unsigned char *pinned_src, *pinned_dst;
    const size_t big = 64 << 20;
    CHECK(cudaMallocHost(&pinned_src, big));
    CHECK(cudaHostAlloc(&pinned_dst, big, cudaHostAllocDefault));
    std::vector<unsigned char> pageable_src(big), pageable_dst(big);
    for (size_t i = 0; i < big; i++) pinned_src[i] = pageable_src[i] = (unsigned char) (i * 7 + i / 251);
    unsigned char *dbig;
    CHECK(cudaMalloc(&dbig, big));
    for (size_t size: {(size_t) 1, (size_t) 3, (size_t) 4096, (size_t) 65537, (size_t) 1 << 20, (size_t) 8 << 20,
                       ((size_t) 8 << 20) + 1, (size_t) 33333333, big}) {
        for (int mode = 0; mode < 2; mode++) {
            unsigned char *src = mode ? pinned_src : pageable_src.data();
            unsigned char *dst = mode ? pinned_dst : pageable_dst.data();
            memset(dst, 0, size);
            CHECK(cudaMemset(dbig, 0, size));
            CHECK(cudaMemcpy(dbig, src + (mode ? 0 : 1) * 0, size, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(dst, dbig, size, cudaMemcpyDeviceToHost));
            EXPECT(memcmp(src, dst, size) == 0, "%s round trip of %zu bytes", mode ? "pinned" : "pageable", size);
        }
    }

    // a kernel can access pinned host memory directly (zero copy)
    unsigned char *mapped;
    CHECK(cudaHostGetDevicePointer((void **) &mapped, pinned_dst, 0));
    fill<<<(1000 + 255) / 256, 256>>>(mapped, 1000, 0x5a);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 1000; i++) EXPECT(pinned_dst[i] == 0x5a, "zero copy byte %d = %d", i, pinned_dst[i]);

    // 2D copies and memset
    const size_t width = 1000, height = 300;
    size_t pitch = 0;
    unsigned char *d2d;
    CHECK(cudaMallocPitch((void **) &d2d, &pitch, width, height));
    EXPECT(pitch >= width, "pitch %zu", pitch);
    CHECK(cudaMemset2D(d2d, pitch, 0x11, width, height));
    std::vector<unsigned char> img(width * height), back(width * height);
    for (size_t i = 0; i < img.size(); i++) img[i] = (unsigned char) (i % 253);
    CHECK(cudaMemcpy2D(d2d, pitch, img.data(), width, width / 2, height, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(back.data(), width, d2d, pitch, width, height, cudaMemcpyDeviceToHost));
    for (size_t y = 0; y < height; y++)
        for (size_t x = 0; x < width; x++) {
            unsigned char e = x < width / 2 ? img[y * width + x] : 0x11;
            EXPECT(back[y * width + x] == e, "2D (%zu, %zu) = %d, expected %d", x, y, back[y * width + x], e);
        }

    // managed memory: accessible from host and device
    int *managed;
    const int nm = 1 << 16;
    CHECK(cudaMallocManaged(&managed, nm * sizeof(int)));
    for (int i = 0; i < nm; i++) managed[i] = 1000;
    incrementManaged<<<(nm + 255) / 256, 256>>>(managed, nm);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < nm; i++) EXPECT(managed[i] == 1000 + i, "managed[%d] = %d", i, managed[i]);

    // pointer attributes
    cudaPointerAttributes attr;
    CHECK(cudaPointerGetAttributes(&attr, d + 100));
    EXPECT(attr.type == cudaMemoryTypeDevice && attr.devicePointer == d + 100, "device pointer attributes");
    CHECK(cudaPointerGetAttributes(&attr, pinned_src + 5));
    EXPECT(attr.type == cudaMemoryTypeHost, "pinned pointer attributes");
    CHECK(cudaPointerGetAttributes(&attr, pageable_src.data()));
    EXPECT(attr.type == cudaMemoryTypeUnregistered, "pageable pointer attributes");

    // allocation churn: memory is returned to the device
    size_t free_before = 0, free_after = 0;
    CHECK(cudaMemGetInfo(&free_before, &total));
    for (int i = 0; i < 200; i++) {
        void *p;
        CHECK(cudaMalloc(&p, (size_t) (1 + rng() % 64) << 20));
        CHECK(cudaMemset(p, 1, 4096));
        CHECK(cudaFree(p));
    }
    CHECK(cudaMemGetInfo(&free_after, &total));
    printf("free memory before/after 200 alloc/free cycles: %.1f / %.1f MiB\n", free_before / 1048576.0,
           free_after / 1048576.0);
    EXPECT(free_after + (64 << 20) >= free_before, "memory leaked");

    CHECK(cudaFree(d)); CHECK(cudaFree(d2)); CHECK(cudaFree(dbig)); CHECK(cudaFree(d2d)); CHECK(cudaFree(managed));
    CHECK(cudaFreeHost(pinned_src)); CHECK(cudaFreeHost(pinned_dst));
    PASS("memory_ops");
    return 0;
}
