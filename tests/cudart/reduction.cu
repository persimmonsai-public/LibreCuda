// Reductions: warp shuffles, cooperative-style block reductions with dynamic shared memory, global atomics on
// int / float / double, templates instantiated for several types, multi-pass reductions
#include "check.h"
#include <vector>
#include <numeric>

template<typename T>
__device__ T warpReduceSum(T v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

template<typename T>
__global__ void blockReduce(const T *in, T *out, size_t n) {
    extern __shared__ unsigned char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);
    T sum = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        sum += in[i];
    sum = warpReduceSum(sum);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) smem[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < blockDim.x / 32 ? smem[lane] : T(0);
        sum = warpReduceSum(sum);
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

__global__ void atomicSums(const int *in, size_t n, unsigned long long *isum, float *fsum, double *dsum, int *imax) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int v = in[i];
        atomicAdd(isum, (unsigned long long) v);
        atomicAdd(fsum, 1.0f);
        atomicAdd(dsum, (double) v * 0.5);
        atomicMax(imax, v);
    }
}

template<typename T>
T reduceOnDevice(const T *d_in, size_t n, int sms) {
    int blocks = sms * 4, threads = 512;
    T *d_partial;
    CHECK(cudaMalloc(&d_partial, blocks * sizeof(T)));
    blockReduce<T><<<blocks, threads, (threads / 32) * sizeof(T)>>>(d_in, d_partial, n);
    CHECK(cudaGetLastError());
    T *d_result;
    CHECK(cudaMalloc(&d_result, sizeof(T)));
    blockReduce<T><<<1, 1024, 32 * sizeof(T)>>>(d_partial, d_result, blocks);
    CHECK(cudaGetLastError());
    T result;
    CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial));
    CHECK(cudaFree(d_result));
    return result;
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const size_t n = 1 << 24;
    std::vector<int> hi(n);
    std::vector<double> hd(n);
    for (size_t i = 0; i < n; i++) { hi[i] = (int) ((i * 2654435761u) % 1000); hd[i] = hi[i] * 0.001; }

    int *di; double *dd;
    CHECK(cudaMalloc(&di, n * sizeof(int)));
    CHECK(cudaMalloc(&dd, n * sizeof(double)));
    CHECK(cudaMemcpy(di, hi.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dd, hd.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    long long expect_i = std::accumulate(hi.begin(), hi.end(), 0LL);
    double expect_d = std::accumulate(hd.begin(), hd.end(), 0.0);
    // widen to 64 bit before reducing; the sum does not fit into an int
    long long *dl;
    std::vector<long long> hl(hi.begin(), hi.end());
    CHECK(cudaMalloc(&dl, n * sizeof(long long)));
    CHECK(cudaMemcpy(dl, hl.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    long long got_i = reduceOnDevice<long long>(dl, n, prop.multiProcessorCount);
    int got_small = reduceOnDevice<int>(di, 1 << 20, prop.multiProcessorCount);
    long long expect_small = std::accumulate(hi.begin(), hi.begin() + (1 << 20), 0LL);
    EXPECT(got_small == expect_small, "int sum %d != %lld", got_small, expect_small);
    double got_d = reduceOnDevice<double>(dd, n, prop.multiProcessorCount);
    EXPECT(got_i == expect_i, "long long sum %lld != %lld", got_i, expect_i);
    EXPECT(nearlyEqual(got_d, expect_d, 1e-9), "double sum %.10f != %.10f", got_d, expect_d);

    unsigned long long *isum; float *fsum; double *dsum; int *imax;
    CHECK(cudaMalloc(&isum, 8)); CHECK(cudaMalloc(&fsum, 4)); CHECK(cudaMalloc(&dsum, 8)); CHECK(cudaMalloc(&imax, 4));
    CHECK(cudaMemset(isum, 0, 8)); CHECK(cudaMemset(fsum, 0, 4)); CHECK(cudaMemset(dsum, 0, 8)); CHECK(cudaMemset(imax, 0, 4));
    const size_t m = 1 << 20;
    atomicSums<<<prop.multiProcessorCount * 2, 256>>>(di, m, isum, fsum, dsum, imax);
    CHECK(cudaGetLastError());
    unsigned long long h_isum; float h_fsum; double h_dsum; int h_imax;
    CHECK(cudaMemcpy(&h_isum, isum, 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_fsum, fsum, 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_dsum, dsum, 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_imax, imax, 4, cudaMemcpyDeviceToHost));
    long long expect_m = std::accumulate(hi.begin(), hi.begin() + m, 0LL);
    EXPECT(h_isum == (unsigned long long) expect_m, "atomic int sum %llu != %lld", h_isum, expect_m);
    EXPECT(h_fsum == (float) m, "atomic float count %f != %zu", h_fsum, m);
    EXPECT(nearlyEqual(h_dsum, expect_m * 0.5, 1e-12), "atomic double sum %f != %f", h_dsum, expect_m * 0.5);
    EXPECT(h_imax == 999, "atomic max %d", h_imax);
    PASS("reduction");
    return 0;
}
