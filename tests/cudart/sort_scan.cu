// Parallel primitives: block-level inclusive scan with dynamic shared memory, multi-block exclusive scan
// (three passes), LSD radix sort of 32-bit keys with key/value pairs, bitonic sort; all checked against std::sort
#include "check.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

constexpr int SCAN_BLOCK = 1024;

// Blelloch-free simple scan: warp shuffles + shared memory of warp totals
__device__ unsigned blockInclusiveScan(unsigned v, unsigned *warpSums) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned n = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) warpSums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        unsigned w = lane < (blockDim.x >> 5) ? warpSums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            unsigned n = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) w += n;
        }
        warpSums[lane] = w;
    }
    __syncthreads();
    if (warp > 0) v += warpSums[warp - 1];
    __syncthreads();
    return v;
}

__global__ void scanBlocks(const unsigned *in, unsigned *out, unsigned *blockSums, size_t n) {
    extern __shared__ unsigned warpSums[];
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    unsigned v = i < n ? in[i] : 0;
    unsigned inc = blockInclusiveScan(v, warpSums);
    if (i < n) out[i] = inc - v; // exclusive
    if (threadIdx.x == blockDim.x - 1) blockSums[blockIdx.x] = inc;
}

__global__ void addBlockOffsets(unsigned *data, const unsigned *offsets, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < n) data[i] += offsets[blockIdx.x];
}

void exclusiveScan(unsigned *d_in, unsigned *d_out, size_t n) {
    size_t blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    unsigned *d_sums, *d_offsets;
    CHECK(cudaMalloc(&d_sums, (blocks + 1) * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_offsets, (blocks + 1) * sizeof(unsigned)));
    scanBlocks<<<blocks, SCAN_BLOCK, 32 * sizeof(unsigned)>>>(d_in, d_out, d_sums, n);
    CHECK(cudaGetLastError());
    if (blocks > 1) {
        exclusiveScan(d_sums, d_offsets, blocks);
        addBlockOffsets<<<blocks, SCAN_BLOCK>>>(d_out, d_offsets, n);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaFree(d_sums));
    CHECK(cudaFree(d_offsets));
}

// radix sort: per pass, histogram of 4-bit digits per block, global scan of the digit/block matrix, stable scatter
constexpr int RADIX_BITS = 4, RADIX = 1 << RADIX_BITS, SORT_BLOCK = 256, ITEMS = 16;

__global__ void radixHistogram(const unsigned *keys, unsigned *hist, size_t n, int shift) {
    __shared__ unsigned local[RADIX];
    if (threadIdx.x < RADIX) local[threadIdx.x] = 0;
    __syncthreads();
    size_t base = blockIdx.x * (size_t) SORT_BLOCK * ITEMS;
    for (int k = 0; k < ITEMS; k++) {
        size_t i = base + k * SORT_BLOCK + threadIdx.x;
        if (i < n) atomicAdd(&local[(keys[i] >> shift) & (RADIX - 1)], 1u);
    }
    __syncthreads();
    if (threadIdx.x < RADIX) hist[threadIdx.x * gridDim.x + blockIdx.x] = local[threadIdx.x];
}

__global__ void radixScatter(const unsigned *keysIn, const unsigned *valsIn, unsigned *keysOut, unsigned *valsOut,
                             const unsigned *offsets, size_t n, int shift) {
    // stable scatter: one warp-sequential pass per digit using ballots
    __shared__ unsigned digitBase[RADIX];
    __shared__ unsigned keysS[SORT_BLOCK * ITEMS];
    __shared__ unsigned valsS[SORT_BLOCK * ITEMS];
    size_t base = blockIdx.x * (size_t) SORT_BLOCK * ITEMS;
    int count = (int) min((size_t) SORT_BLOCK * ITEMS, n - base);
    for (int i = threadIdx.x; i < count; i += SORT_BLOCK) { keysS[i] = keysIn[base + i]; valsS[i] = valsIn[base + i]; }
    if (threadIdx.x < RADIX) digitBase[threadIdx.x] = offsets[threadIdx.x * gridDim.x + blockIdx.x];
    __syncthreads();
    if (threadIdx.x < 32) {
        // warp 0 walks the tile in order, 32 elements at a time
        for (int start = 0; start < count; start += 32) {
            int i = start + threadIdx.x;
            bool valid = i < count;
            unsigned key = valid ? keysS[i] : 0;
            unsigned digit = (key >> shift) & (RADIX - 1);
            unsigned rank = 0, total = 0;
            for (int d = 0; d < RADIX; d++) {
                unsigned mask = __ballot_sync(0xffffffff, valid && digit == d);
                if (valid && digit == d) rank = __popc(mask & ((1u << threadIdx.x) - 1));
                if (threadIdx.x == d) total = __popc(mask);
            }
            if (valid) {
                unsigned pos = digitBase[digit] + rank;
                keysOut[pos] = key;
                valsOut[pos] = valsS[i];
            }
            __syncwarp();
            if (threadIdx.x < RADIX) digitBase[threadIdx.x] += total;
            __syncwarp();
        }
    }
}

void radixSort(unsigned *d_keys, unsigned *d_vals, size_t n) {
    size_t blocks = (n + SORT_BLOCK * ITEMS - 1) / (SORT_BLOCK * ITEMS);
    unsigned *d_keys2, *d_vals2, *d_hist, *d_offsets;
    CHECK(cudaMalloc(&d_keys2, n * 4)); CHECK(cudaMalloc(&d_vals2, n * 4));
    CHECK(cudaMalloc(&d_hist, blocks * RADIX * 4)); CHECK(cudaMalloc(&d_offsets, blocks * RADIX * 4));
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        radixHistogram<<<blocks, SORT_BLOCK>>>(d_keys, d_hist, n, shift);
        exclusiveScan(d_hist, d_offsets, blocks * RADIX);
        radixScatter<<<blocks, SORT_BLOCK>>>(d_keys, d_vals, d_keys2, d_vals2, d_offsets, n, shift);
        CHECK(cudaGetLastError());
        std::swap(d_keys, d_keys2);
        std::swap(d_vals, d_vals2);
    }
    // 8 passes: the result is back in the original buffers
    CHECK(cudaFree(d_keys2)); CHECK(cudaFree(d_vals2)); CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_offsets));
}

__global__ void bitonicStep(float *data, int j, int k) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ixj = i ^ j;
    if (ixj > i) {
        bool ascending = (i & k) == 0;
        float a = data[i], b = data[ixj];
        if ((a > b) == ascending) { data[i] = b; data[ixj] = a; }
    }
}

int main() {
    std::mt19937 rng(7);
    // scan
    const size_t n_scan = 3'000'001;
    std::vector<unsigned> v(n_scan);
    for (auto &x: v) x = rng() % 17;
    unsigned *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n_scan * 4));
    CHECK(cudaMalloc(&d_out, n_scan * 4));
    CHECK(cudaMemcpy(d_in, v.data(), n_scan * 4, cudaMemcpyHostToDevice));
    exclusiveScan(d_in, d_out, n_scan);
    std::vector<unsigned> scanned(n_scan);
    CHECK(cudaMemcpy(scanned.data(), d_out, n_scan * 4, cudaMemcpyDeviceToHost));
    unsigned acc = 0;
    for (size_t i = 0; i < n_scan; i++) {
        EXPECT(scanned[i] == acc, "scan[%zu] = %u, expected %u", i, scanned[i], acc);
        acc += v[i];
    }

    // radix sort key/value
    const size_t n_sort = 2'000'003;
    std::vector<unsigned> keys(n_sort), vals(n_sort);
    for (size_t i = 0; i < n_sort; i++) { keys[i] = rng(); vals[i] = (unsigned) i; }
    unsigned *d_keys, *d_vals;
    CHECK(cudaMalloc(&d_keys, n_sort * 4));
    CHECK(cudaMalloc(&d_vals, n_sort * 4));
    CHECK(cudaMemcpy(d_keys, keys.data(), n_sort * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vals, vals.data(), n_sort * 4, cudaMemcpyHostToDevice));
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    radixSort(d_keys, d_vals, n_sort);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    std::vector<unsigned> sorted_keys(n_sort), sorted_vals(n_sort);
    CHECK(cudaMemcpy(sorted_keys.data(), d_keys, n_sort * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sorted_vals.data(), d_vals, n_sort * 4, cudaMemcpyDeviceToHost));
    std::vector<size_t> perm(n_sort);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
    for (size_t i = 0; i < n_sort; i++) {
        EXPECT(sorted_keys[i] == keys[perm[i]], "key[%zu] = %u, expected %u", i, sorted_keys[i], keys[perm[i]]);
        EXPECT(sorted_vals[i] == perm[i], "value[%zu] = %u, expected %zu (not stable?)", i, sorted_vals[i], perm[i]);
    }
    printf("radix sort of %zu key/value pairs: %.2f ms\n", n_sort, ms);

    // bitonic sort (many small dependent launches)
    const int n_bitonic = 1 << 20;
    std::vector<float> f(n_bitonic);
    for (auto &x: f) x = (float) (rng() % 100000) / 7.0f;
    float *d_f;
    CHECK(cudaMalloc(&d_f, n_bitonic * 4));
    CHECK(cudaMemcpy(d_f, f.data(), n_bitonic * 4, cudaMemcpyHostToDevice));
    int launches = 0;
    CHECK(cudaEventRecord(t0));
    for (int k = 2; k <= n_bitonic; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicStep<<<n_bitonic / 512, 512>>>(d_f, j, k);
            launches++;
        }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    std::vector<float> fs(n_bitonic);
    CHECK(cudaMemcpy(fs.data(), d_f, n_bitonic * 4, cudaMemcpyDeviceToHost));
    std::sort(f.begin(), f.end());
    for (int i = 0; i < n_bitonic; i++) EXPECT(fs[i] == f[i], "bitonic[%d] = %f, expected %f", i, fs[i], f[i]);
    printf("bitonic sort of %d floats: %d launches, %.2f ms\n", n_bitonic, launches, ms);
    PASS("sort_scan");
    return 0;
}
