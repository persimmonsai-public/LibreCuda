// Managed memory: CPU and GPU access with migration both ways, pointer attributes and device attributes, copies and
// fills between managed, device, page-locked and pageable memory (also stream ordered after kernels), 2D copies and
// copies to and from CUDA arrays, prefetching and advice, concurrent CPU and GPU access to different pages, large
// allocations, and two devices sharing an allocation (with 2 devices). Checked against NVIDIA's runtime too.
#include "check.h"
#include <cstring>
#include <vector>

__global__ void addOne(int *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] += 1;
    }
}

__global__ void fillSequence(int *p, size_t n, int base) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] = base + int(i);
    }
}

// spins for about the given number of clock cycles
__global__ void spin(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

static void expectSequence(const int *p, size_t n, int base, const char *what) {
    for (size_t i = 0; i < n; i++) {
        EXPECT(p[i] == base + int(i), "%s: [%zu] = %d, expected %d", what, i, p[i], base + int(i));
    }
}

static void testBasics() {
    int value = 0;
    CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, 0));
    EXPECT(value == 1, "cudaDevAttrManagedMemory %d", value);
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    EXPECT(prop.managedMemory == 1, "managedMemory %d", prop.managedMemory);

    void *none = nullptr;
    EXPECT(cudaMallocManaged(&none, 1024, 0x100) == cudaErrorInvalidValue, "invalid flags");
    cudaGetLastError();

    const size_t n = 1 << 20;
    int *m;
    CHECK(cudaMallocManaged(&m, n * sizeof(int)));
    cudaPointerAttributes attributes{};
    CHECK(cudaPointerGetAttributes(&attributes, m + 5));
    EXPECT(attributes.type == cudaMemoryTypeManaged && attributes.devicePointer == m + 5 &&
           attributes.hostPointer == m + 5, "attributes: type %d", int(attributes.type));

    // CPU writes, GPU updates, CPU reads, several times
    for (size_t i = 0; i < n; i++) {
        m[i] = int(i);
    }
    for (int round = 1; round <= 3; round++) {
        addOne<<<256, 256>>>(m, n);
        CHECK(cudaDeviceSynchronize());
        expectSequence(m, n, round, "CPU / GPU rounds");
    }
    // the GPU writes pages the CPU never touched
    int *fresh;
    CHECK(cudaMallocManaged(&fresh, n * sizeof(int)));
    fillSequence<<<256, 256>>>(fresh, n, 7);
    CHECK(cudaDeviceSynchronize());
    expectSequence(fresh, n, 7, "GPU first touch");
    CHECK(cudaFree(fresh));
    CHECK(cudaFree(m));
    PASS("managed basics");
}

static void testCopies() {
    const size_t n = 1 << 18, bytes = n * sizeof(int);
    int *m, *m2, *d, *pinned;
    CHECK(cudaMallocManaged(&m, bytes));
    CHECK(cudaMallocManaged(&m2, bytes));
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaMallocHost(&pinned, bytes));
    std::vector<int> pageable(n);

    // pageable -> managed -> device -> managed -> pinned
    for (size_t i = 0; i < n; i++) {
        pageable[i] = 3 + int(i);
    }
    CHECK(cudaMemcpy(m, pageable.data(), bytes, cudaMemcpyDefault));
    CHECK(cudaMemcpy(d, m, bytes, cudaMemcpyDefault));
    CHECK(cudaMemcpy(m2, d, bytes, cudaMemcpyDefault));
    CHECK(cudaMemcpy(pinned, m2, bytes, cudaMemcpyDefault));
    expectSequence(pinned, n, 3, "pageable -> managed -> device -> managed -> pinned");
    // managed -> managed and managed -> pageable
    CHECK(cudaMemcpy(m, pinned, bytes, cudaMemcpyDefault));
    CHECK(cudaMemcpy(m2, m, bytes, cudaMemcpyDefault));
    std::fill(pageable.begin(), pageable.end(), 0);
    CHECK(cudaMemcpy(pageable.data(), m2, bytes, cudaMemcpyDefault));
    expectSequence(pageable.data(), n, 3, "managed -> managed -> pageable");

    // stream order: a kernel writes managed memory, then asynchronous copies and fills follow it
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    fillSequence<<<256, 256, 0, stream>>>(m, n, 100);
    CHECK(cudaMemcpyAsync(d, m, bytes, cudaMemcpyDefault, stream));
    CHECK(cudaMemcpyAsync(pageable.data(), m, bytes, cudaMemcpyDefault, stream));
    CHECK(cudaStreamSynchronize(stream));
    expectSequence(pageable.data(), n, 100, "managed -> pageable after a kernel");
    CHECK(cudaMemcpy(pinned, d, bytes, cudaMemcpyDefault));
    expectSequence(pinned, n, 100, "managed -> device after a kernel");
    fillSequence<<<256, 256, 0, stream>>>(m, n, 0);
    CHECK(cudaMemsetAsync(m, 0x02, bytes / 2, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(m[0] == 0x02020202 && m[n / 2 - 1] == 0x02020202 && m[n / 2] == int(n / 2), "memset after a kernel: "
           "%x %x %d", m[0], m[n / 2 - 1], m[n / 2]);
    CHECK(cudaStreamDestroy(stream));

    // 2D copies between managed and pageable memory
    const size_t width = 100, height = 50, pitch = 128;
    std::vector<unsigned char> h2(pitch * height, 0);
    unsigned char *m2d;
    CHECK(cudaMallocManaged(&m2d, pitch * height));
    for (size_t i = 0; i < pitch * height; i++) {
        m2d[i] = static_cast<unsigned char>(i * 7);
    }
    CHECK(cudaMemcpy2D(h2.data(), pitch, m2d, pitch, width, height, cudaMemcpyDefault));
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            EXPECT(h2[y * pitch + x] == static_cast<unsigned char>((y * pitch + x) * 7), "2D copy at %zu,%zu", x, y);
        }
    }

    // CUDA array to and from managed memory
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaArray_t array;
    CHECK(cudaMallocArray(&array, &desc, 64, 32));
    for (int i = 0; i < 64 * 32; i++) {
        m[i] = 1000 + i;
    }
    CHECK(cudaMemcpy2DToArray(array, 0, 0, m, 64 * sizeof(int), 64 * sizeof(int), 32, cudaMemcpyDefault));
    CHECK(cudaMemset(m2, 0, 64 * 32 * sizeof(int)));
    CHECK(cudaMemcpy2DFromArray(m2, 64 * sizeof(int), array, 0, 0, 64 * sizeof(int), 32, cudaMemcpyDefault));
    CHECK(cudaDeviceSynchronize());
    expectSequence(m2, 64 * 32, 1000, "managed -> array -> managed");
    CHECK(cudaFreeArray(array));

    CHECK(cudaFree(m2d));
    CHECK(cudaFreeHost(pinned));
    CHECK(cudaFree(d));
    CHECK(cudaFree(m2));
    CHECK(cudaFree(m));
    PASS("managed copies and fills");
}

static void testPrefetchAndAdvice() {
    const size_t n = 4 << 20, bytes = n * sizeof(int);
    int *m;
    CHECK(cudaMallocManaged(&m, bytes));
    for (size_t i = 0; i < n; i++) {
        m[i] = int(i);
    }
    cudaMemLocation device0{cudaMemLocationTypeDevice, 0}, host{cudaMemLocationTypeHost, 0};
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemPrefetchAsync(m, bytes, device0, 0, stream));
    addOne<<<256, 256, 0, stream>>>(m, n);
    CHECK(cudaMemPrefetchAsync(m, bytes, host, 0, stream));
    CHECK(cudaStreamSynchronize(stream));
    expectSequence(m, n, 1, "prefetch to the device and back");
    // a part of an allocation
    CHECK(cudaMemPrefetchAsync(m + 1000, 4096, device0, 0, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseSetReadMostly, device0));
    addOne<<<256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(m, n, 2, "read mostly");
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseUnsetReadMostly, device0));
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseSetPreferredLocation, device0));
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseSetAccessedBy, host));
    addOne<<<256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(m, n, 3, "preferred location on the device, accessed by the CPU");
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseUnsetAccessedBy, host));
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseUnsetPreferredLocation, device0));
    CHECK(cudaMemAdvise(m, bytes, cudaMemAdviseSetPreferredLocation, host));
    addOne<<<256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(m, n, 4, "preferred location on the CPU");

    // device memory isn't managed
    int *d;
    CHECK(cudaMalloc(&d, 4096));
    EXPECT(cudaMemPrefetchAsync(d, 4096, device0, 0, stream) == cudaErrorInvalidValue, "prefetch of device memory");
    cudaGetLastError();
    CHECK(cudaFree(d));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(m));
    PASS("managed prefetch and advice");
}

static void testConcurrentAccess() {
    int concurrent = 0;
    CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0));
    if (!concurrent) {
        printf("PASS managed concurrent access (skipped: not supported)\n");
        return;
    }
    // the GPU works on the first half while the CPU writes the second half
    const size_t n = 1 << 20, half = n / 2;
    int *m;
    CHECK(cudaMallocManaged(&m, n * sizeof(int)));
    memset(m, 0, n * sizeof(int));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    spin<<<1, 1, 0, stream>>>(10000000);
    fillSequence<<<256, 256, 0, stream>>>(m, half, 0);
    for (size_t i = half; i < n; i++) {
        m[i] = int(i);
    }
    CHECK(cudaStreamSynchronize(stream));
    expectSequence(m, n, 0, "concurrent CPU and GPU writes");
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(m));
    PASS("managed concurrent access");
}

static void testLarge() {
    const size_t n = size_t(256) << 18; // 256 MiB
    int *m;
    CHECK(cudaMallocManaged(&m, n * sizeof(int)));
    for (size_t i = 0; i < n; i += 1024) {
        m[i] = int(i);
    }
    addOne<<<1024, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < n; i += 1024) {
        EXPECT(m[i] == int(i) + 1, "large: [%zu] = %d", i, m[i]);
    }
    CHECK(cudaFree(m));
    PASS("managed large allocation");
}

static void testTwoDevices() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        printf("PASS managed two devices (skipped: needs 2 devices)\n");
        return;
    }
    const size_t n = 1 << 20;
    int *m;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMallocManaged(&m, n * sizeof(int)));
    fillSequence<<<256, 256>>>(m, n, 5);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaSetDevice(1));
    addOne<<<256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(m, n, 6, "device 0 then device 1");
    cudaMemLocation device1{cudaMemLocationTypeDevice, 1};
    CHECK(cudaMemPrefetchAsync(m, n * sizeof(int), device1, 0, 0));
    addOne<<<256, 256>>>(m, n);
    // the devices run independently: device 0 updates the memory after device 1 finished
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaSetDevice(0));
    addOne<<<256, 256>>>(m, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(m, n, 8, "prefetched to device 1, used by both");
    CHECK(cudaFree(m));
    CHECK(cudaSetDevice(0));
    PASS("managed two devices");
}

int main() {
    testBasics();
    testCopies();
    testPrefetchAndAdvice();
    testConcurrentAccess();
    testLarge();
    testTwoDevices();
    CHECK(cudaGetLastError());
    return 0;
}
