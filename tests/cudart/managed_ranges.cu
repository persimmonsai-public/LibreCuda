// Managed memory policy queries: cudaMemRangeGetAttribute(s) after cudaMemAdvise and cudaMemPrefetchAsync (whole
// ranges, parts with different policies, the accessed-by list, location types and ids), invalid queries, batch
// prefetches and discards, stream attachment, and the policies of freed memory. Checked against NVIDIA's runtime too.
#include "check.h"

struct Policies {
    int readMostly, preferred, lastPrefetch, preferredType, preferredId, lastType, lastId;
    int accessedBy[4];
};

static Policies query(const void *ptr, size_t size) {
    Policies p{};
    void *data[] = {&p.readMostly, &p.preferred, &p.lastPrefetch, &p.preferredType, &p.preferredId, &p.lastType,
                    &p.lastId, p.accessedBy};
    size_t sizes[] = {4, 4, 4, 4, 4, 4, 4, sizeof(p.accessedBy)};
    cudaMemRangeAttribute attributes[] = {
        cudaMemRangeAttributeReadMostly, cudaMemRangeAttributePreferredLocation,
        cudaMemRangeAttributeLastPrefetchLocation, cudaMemRangeAttributePreferredLocationType,
        cudaMemRangeAttributePreferredLocationId, cudaMemRangeAttributeLastPrefetchLocationType,
        cudaMemRangeAttributeLastPrefetchLocationId, cudaMemRangeAttributeAccessedBy};
    CHECK(cudaMemRangeGetAttributes(data, sizes, attributes, 8, ptr, size));
    return p;
}

static void expectPolicies(int line, const void *ptr, size_t size, int read_mostly, int preferred, int last,
                           int accessed0, int accessed1) {
    const Policies p = query(ptr, size);
    EXPECT(p.readMostly == read_mostly && p.preferred == preferred && p.lastPrefetch == last &&
           p.accessedBy[0] == accessed0 && p.accessedBy[1] == accessed1 && p.accessedBy[2] == cudaInvalidDeviceId,
           "line %d: read mostly %d, preferred %d, last prefetch %d, accessed by %d %d %d", line, p.readMostly,
           p.preferred, p.lastPrefetch, p.accessedBy[0], p.accessedBy[1], p.accessedBy[2]);
    auto type = [](int id) { return id == cudaInvalidDeviceId ? 0 : id == cudaCpuDeviceId ? 2 : 1; };
    EXPECT(p.preferredType == type(preferred) && p.preferredId == preferred && p.lastType == type(last) &&
           p.lastId == last, "line %d: location types / ids %d %d %d %d", line, p.preferredType, p.preferredId,
           p.lastType, p.lastId);
}

#define EXPECT_POLICIES(...) expectPolicies(__LINE__, __VA_ARGS__)

int main() {
    const int none = cudaInvalidDeviceId, cpu = cudaCpuDeviceId;
    const size_t n = 8 << 20, half = n / 2;
    char *m;
    CHECK(cudaMallocManaged(&m, n));
    cudaMemLocation dev{cudaMemLocationTypeDevice, 0}, host{cudaMemLocationTypeHost, 0};

    EXPECT_POLICIES(m, n, 0, none, none, none, none);
    CHECK(cudaMemAdvise(m, n, cudaMemAdviseSetReadMostly, dev));
    EXPECT_POLICIES(m, n, 1, none, none, none, none);
    CHECK(cudaMemAdvise(m, half, cudaMemAdviseUnsetReadMostly, dev));
    EXPECT_POLICIES(m, n, 0, none, none, none, none);
    EXPECT_POLICIES(m + half, half, 1, none, none, none, none);
    CHECK(cudaMemAdvise(m, n, cudaMemAdviseUnsetReadMostly, dev));

    CHECK(cudaMemAdvise(m, n, cudaMemAdviseSetPreferredLocation, dev));
    EXPECT_POLICIES(m, n, 0, 0, none, none, none);
    CHECK(cudaMemAdvise(m, half, cudaMemAdviseSetPreferredLocation, host));
    EXPECT_POLICIES(m, n, 0, none, none, none, none);
    EXPECT_POLICIES(m, half, 0, cpu, none, none, none);
    // a query of a part of a page covers the page
    EXPECT_POLICIES(m + 100, 10, 0, cpu, none, none, none);
    CHECK(cudaMemAdvise(m, n, cudaMemAdviseUnsetPreferredLocation, dev));
    EXPECT_POLICIES(m, n, 0, none, none, none, none);

    CHECK(cudaMemAdvise(m, n, cudaMemAdviseSetAccessedBy, host));
    CHECK(cudaMemAdvise(m, n, cudaMemAdviseSetAccessedBy, dev));
    EXPECT_POLICIES(m, n, 0, none, none, 0, cpu);
    CHECK(cudaMemAdvise(m, half, cudaMemAdviseUnsetAccessedBy, host));
    EXPECT_POLICIES(m, n, 0, none, none, 0, none);
    EXPECT_POLICIES(m + half, half, 0, none, none, 0, cpu);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemPrefetchAsync(m, n, dev, 0, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT_POLICIES(m, n, 0, none, 0, 0, none);
    CHECK(cudaMemPrefetchAsync(m, 4096, host, 0, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT_POLICIES(m, n, 0, none, none, 0, none);
    EXPECT_POLICIES(m, 4096, 0, none, cpu, 0, none);

    // invalid queries
    int value = 0;
    EXPECT(cudaMemRangeGetAttribute(&value, 2, cudaMemRangeAttributeReadMostly, m, n) == cudaErrorInvalidValue,
           "too small a buffer");
    int *d;
    CHECK(cudaMalloc(&d, 4096));
    EXPECT(cudaMemRangeGetAttribute(&value, 4, cudaMemRangeAttributeReadMostly, d, 4096) == cudaErrorInvalidValue,
           "device memory");
    EXPECT(cudaStreamAttachMemAsync(stream, d, 0, cudaMemAttachGlobal) == cudaErrorInvalidValue,
           "attaching device memory");
    cudaGetLastError();
    CHECK(cudaStreamAttachMemAsync(stream, m, 0, cudaMemAttachGlobal));
    CHECK(cudaStreamAttachMemAsync(stream, m, 0, cudaMemAttachSingle));

    // batch prefetch: first half to the CPU, second half to the device (not on the legacy default stream)
    void *ptrs[2] = {m, m + half};
    size_t sizes[2] = {half, half};
    cudaMemLocation locations[2] = {host, dev};
    size_t indices[2] = {0, 1};
    EXPECT(cudaMemPrefetchBatchAsync(ptrs, sizes, 2, locations, indices, 2, 0, 0) == cudaErrorInvalidValue,
           "batch prefetch on the legacy default stream");
    cudaGetLastError();
    CHECK(cudaMemPrefetchBatchAsync(ptrs, sizes, 2, locations, indices, 2, 0, stream));
    CHECK(cudaStreamSynchronize(stream));
    // (the location id NVIDIA's runtime reports after a batch prefetch to the host is the location's id, not
    // cudaCpuDeviceId: not checked)
    const Policies first = query(m, half), second = query(m + half, half);
    EXPECT(first.lastPrefetch == cpu && first.lastType == cudaMemLocationTypeHost, "batch prefetch to the host: %d %d",
           first.lastPrefetch, first.lastType);
    EXPECT(second.lastPrefetch == 0 && second.lastType == cudaMemLocationTypeDevice && second.lastId == 0,
           "batch prefetch to the device: %d %d %d", second.lastPrefetch, second.lastType, second.lastId);
    CHECK(cudaMemDiscardBatchAsync(ptrs, sizes, 2, 0, stream));
    CHECK(cudaStreamSynchronize(stream));

    // memory still works after all of that
    memset(m, 7, n);
    CHECK(cudaMemset(m, 1, half));
    CHECK(cudaDeviceSynchronize());
    EXPECT(m[0] == 1 && m[n - 1] == 7, "contents %d %d", m[0], m[n - 1]);

    // a new allocation (possibly at a freed one's place) starts without policies
    CHECK(cudaFree(m));
    char *m2;
    CHECK(cudaMallocManaged(&m2, n));
    EXPECT_POLICIES(m2, n, 0, none, none, none, none);
    CHECK(cudaFree(m2));
    CHECK(cudaFree(d));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGetLastError());
    PASS("managed memory range queries");
    return 0;
}
