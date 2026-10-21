// Thrust and CUB from the CUDA toolkit: heavily templated library code using many runtime APIs
// (device attributes, occupancy, temporary allocations, streams, ...)
#include "check.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

struct Square {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

struct IsOdd {
    __host__ __device__ bool operator()(int x) const { return x & 1; }
};

int main() {
    std::mt19937 rng(5);
    const size_t n = 3'000'000;
    std::vector<int> h(n);
    for (auto &x: h) x = (int) (rng() % 1000000) - 500000;

    // thrust sort / reduce / scan / transform / count
    thrust::device_vector<int> dv(h.begin(), h.end());
    thrust::sort(dv.begin(), dv.end());
    thrust::host_vector<int> sorted = dv;
    std::vector<int> ref = h;
    std::sort(ref.begin(), ref.end());
    EXPECT(std::equal(ref.begin(), ref.end(), sorted.begin()), "thrust::sort");

    long long sum = thrust::reduce(dv.begin(), dv.end(), 0LL, thrust::plus<long long>());
    EXPECT(sum == std::accumulate(h.begin(), h.end(), 0LL), "thrust::reduce %lld", sum);

    thrust::device_vector<int> scanned(n);
    thrust::inclusive_scan(dv.begin(), dv.end(), scanned.begin());
    thrust::host_vector<int> hs = scanned;
    int acc = 0;
    for (size_t i = 0; i < n; i++) { acc += ref[i]; EXPECT(hs[i] == acc, "inclusive_scan[%zu]", i); }

    thrust::device_vector<float> f(n);
    thrust::sequence(f.begin(), f.end(), 0.0f, 0.001f);
    thrust::device_vector<float> f2(n);
    thrust::transform(f.begin(), f.end(), f2.begin(), Square());
    float fsum = thrust::reduce(f2.begin(), f2.end());
    double fref = 0;
    for (size_t i = 0; i < n; i++) fref += (double) (i * 0.001f) * (i * 0.001f);
    EXPECT(nearlyEqual(fsum, fref, 1e-3), "transform-reduce %f vs %f", fsum, fref);

    size_t odd = thrust::count_if(dv.begin(), dv.end(), IsOdd());
    EXPECT(odd == (size_t) std::count_if(h.begin(), h.end(), [](int x) { return x & 1; }), "count_if");

    // thrust sort_by_key on floats with a custom comparator
    thrust::device_vector<float> keys(n);
    thrust::device_vector<int> vals(n);
    std::vector<float> hk(n);
    for (auto &x: hk) x = (float) (rng() % 100000) * 0.01f;
    thrust::copy(hk.begin(), hk.end(), keys.begin());
    thrust::sequence(vals.begin(), vals.end());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), vals.begin(), thrust::greater<float>());
    thrust::host_vector<float> sk = keys;
    thrust::host_vector<int> sv = vals;
    for (size_t i = 1; i < n; i++) {
        EXPECT(sk[i - 1] >= sk[i], "sort_by_key order at %zu", i);
        EXPECT(hk[sv[i]] == sk[i], "sort_by_key value at %zu", i);
    }

    // CUB device radix sort with explicit temporary storage, on a stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    unsigned *d_keys_in, *d_keys_out;
    std::vector<unsigned> uk(n);
    for (auto &x: uk) x = rng();
    CHECK(cudaMalloc(&d_keys_in, n * 4));
    CHECK(cudaMalloc(&d_keys_out, n * 4));
    CHECK(cudaMemcpyAsync(d_keys_in, uk.data(), n * 4, cudaMemcpyHostToDevice, stream));
    size_t temp_bytes = 0;
    CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_keys_in, d_keys_out, (int) n, 0, 32, stream));
    void *temp;
    CHECK(cudaMalloc(&temp, temp_bytes));
    CHECK(cub::DeviceRadixSort::SortKeys(temp, temp_bytes, d_keys_in, d_keys_out, (int) n, 0, 32, stream));
    std::vector<unsigned> uk_sorted(n);
    CHECK(cudaMemcpyAsync(uk_sorted.data(), d_keys_out, n * 4, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    std::sort(uk.begin(), uk.end());
    EXPECT(uk == uk_sorted, "cub::DeviceRadixSort");

    // CUB reduce (max)
    int *d_max;
    CHECK(cudaMalloc(&d_max, sizeof(int)));
    size_t temp2 = 0;
    int *d_in = thrust::raw_pointer_cast(dv.data());
    CHECK(cub::DeviceReduce::Max(nullptr, temp2, d_in, d_max, (int) n, stream));
    void *tmp2;
    CHECK(cudaMalloc(&tmp2, temp2));
    CHECK(cub::DeviceReduce::Max(tmp2, temp2, d_in, d_max, (int) n, stream));
    int hmax;
    CHECK(cudaMemcpyAsync(&hmax, d_max, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(hmax == ref.back(), "cub max %d vs %d", hmax, ref.back());

    CHECK(cudaStreamDestroy(stream));
    PASS("thrust_cub");
    return 0;
}
