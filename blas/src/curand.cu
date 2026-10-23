/**
 * LibreCUDA's cuRAND (libcurand.so.10): the host API (generators, distributions, direction vectors) on the generators
 * of the public device API (curand_kernel.h), in the orderings cuRAND's documentation specifies:
 *
 * - XORWOW: 4096 states; the result at offset n is element n / 4096 of subsequence n % 4096 (subsequences are 2^67
 *   apart). The seeded ordering seeds each of the 4096 states from the seed and its index instead.
 * - MRG32k3a: 81920 states (legacy ordering: 4096), element n / states of subsequence n % states (2^76 apart).
 * - Philox4_32_10: 65536 subsequences; four results from one subsequence are followed by four from the next.
 * - MTGP32: 192 parameter sets (legacy ordering: 64); 256 results from the sequence of one set are followed by 256 of
 *   the next set.
 * - MT19937: 8192 generators, generator g starting at output g * 2^1000 of the MT19937 sequence seeded like the CPU
 *   version; eight results from one generator are followed by eight of the next. (cuRAND permutes the results within
 *   these groups in an unpublished way, so this ordering is LibreCUDA's own.)
 * - Sobol' (32 and 64 bit, scrambled): Joe and Kuo's direction numbers; the results of each dimension follow each
 *   other. The scrambled variants use LibreCUDA's own scrambling (cuRAND's scrambled direction vectors and constants
 *   aren't published).
 *
 * XORWOW, MRG32k3a and Philox keep their states between calls (every thread owns the results of its subsequence), so
 * distributions consuming several values (Box-Muller normals, doubles) continue where the last call stopped. MTGP32,
 * MT19937 and Sobol' produce one value per result, transformed like the device API does for them (normals through the
 * inverse distribution function). Poisson and binomial results come from a square histogram (Marsaglia, Tsang and
 * Wang), or from a normal approximation for large variances.
 *
 * Generators created with curandCreateGeneratorHost run on the GPU as well; the results are copied to host memory.
 */
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <cuda_runtime_api.h>
#include <library_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <vector>

namespace {

#include "mt19937_jump.inc"
#include "sobol_joe_kuo.inc"

constexpr uint32_t MAGIC = 0x6372616e;
constexpr int XORWOW_STATES = 4096, MRG_STATES = 81920, MRG_LEGACY_STATES = 4096, PHILOX_STATES = 65536;
constexpr int MTGP_SETS = 192, MTGP_LEGACY_SETS = 64, MTGP_THREADS = 256;
constexpr int MT_N = 624, MT_M = 397, MT_GENERATORS = 8192, MT_GROUP = 8;
constexpr double GAUSSIAN_ABOVE = 2000;  // variance above which discrete distributions are normal approximations

enum class Dist { BITS, UNIFORM, NORMAL, LOG_NORMAL, DISCRETE };

struct DistParams {
    Dist dist;
    double mean, stddev;
    curandDiscreteDistribution_t discrete;
};

bool isQuasi(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SOBOL32 || t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || t == CURAND_RNG_QUASI_SOBOL64 ||
           t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

bool is64(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SOBOL64 || t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

bool isScrambled(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// ---- device side: draws from states, transforms of raw values

__device__ unsigned long long mixSeed(unsigned long long seed, unsigned int t) {
    unsigned long long z = seed + (t + 1ull) * 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

template<typename S, typename Out>
__device__ Out drawFromState(S *s, const DistParams &p) {
    if constexpr (std::is_same_v<Out, unsigned int>) {
        if (p.dist == Dist::DISCRETE) return curand_discrete(s, p.discrete);
        return curand(s);
    } else if constexpr (std::is_same_v<Out, float>) {
        if (p.dist == Dist::NORMAL) return float(p.mean) + float(p.stddev) * curand_normal(s);
        if (p.dist == Dist::LOG_NORMAL) return curand_log_normal(s, float(p.mean), float(p.stddev));
        return curand_uniform(s);
    } else {
        if (p.dist == Dist::NORMAL) return p.mean + p.stddev * curand_normal_double(s);
        if (p.dist == Dist::LOG_NORMAL) return curand_log_normal_double(s, p.mean, p.stddev);
        return curand_uniform_double(s);
    }
}

template<typename X, typename Out>
__device__ Out transform(X x, const DistParams &p) {
    if constexpr (std::is_same_v<Out, unsigned int>) {
        if (p.dist == Dist::DISCRETE) return _curand_discrete(x, p.discrete);
        return (unsigned int) x;
    } else if constexpr (std::is_same_v<Out, unsigned long long>) {
        return x;
    } else if constexpr (std::is_same_v<Out, float>) {
        if (p.dist == Dist::NORMAL) return float(p.mean) + float(p.stddev) * _curand_normal_icdf(x);
        if (p.dist == Dist::LOG_NORMAL) return expf(float(p.mean) + float(p.stddev) * _curand_normal_icdf(x));
        return _curand_uniform(x);
    } else {
        if (p.dist == Dist::NORMAL) return p.mean + p.stddev * _curand_normal_icdf_double(x);
        if (p.dist == Dist::LOG_NORMAL) return exp(p.mean + p.stddev * _curand_normal_icdf_double(x));
        return _curand_uniform_double(x);
    }
}

// the state of thread t positioned after the results of its subsequence before result pos (groups of `group` results
// per subsequence in turn)
template<typename S>
__global__ void initStates(S *states, unsigned int threads, unsigned int group, unsigned long long seed,
                           unsigned long long pos, bool seeded) {
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;
    const unsigned long long groups = pos / group, rest = pos % group;
    unsigned long long consumed = groups > t ? (groups - t + threads - 1) / threads * group : 0;
    if (groups % threads == t) consumed += rest;
    S s;
    if (seeded) {
        curand_init(mixSeed(seed, t), 0ull, consumed, &s);
    } else {
        curand_init(seed, (unsigned long long) t, consumed, &s);
    }
    states[t] = s;
}

// results [pos, pos + n): thread t produces the groups g with g % threads == t
template<typename S, typename Out>
__global__ void stateful(S *states, unsigned int threads, unsigned int group, unsigned long long pos, size_t n,
                         Out *out, DistParams p) {
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;
    S s = states[t];
    const unsigned long long end = pos + n;
    unsigned long long g = pos / group;
    g += (t + threads - unsigned(g % threads)) % threads;
    if (group == 1) {
#pragma unroll 4
        for (; g < end; g += threads) out[g - pos] = drawFromState<S, Out>(&s, p);
        states[t] = s;
        return;
    }
    for (; g * group < end; g += threads) {
        for (unsigned int j = 0; j < group; j++) {
            const unsigned long long e = g * group + j;
            if (e < pos) continue;
            if (e >= end) break;
            out[e - pos] = drawFromState<S, Out>(&s, p);
        }
    }
    states[t] = s;
}

template<typename Out>
__global__ void transformRaw(const unsigned int *raw, size_t n, Out *out, DistParams p) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        out[i] = transform<unsigned int, Out>(raw[i], p);
    }
}

// MTGP32: block b updates the state of parameter set b (in shared memory); round r's 256 results of set b go to
// (r * sets + b) * 256 (transformed; out may be nullptr: skipped)
template<typename Out>
__global__ void mtgpRounds(curandStateMtgp32_t *states, unsigned int sets, Out *out, size_t rounds, DistParams p) {
    __shared__ curandStateMtgp32_t s;
    const curandStateMtgp32_t &global = states[blockIdx.x];
    for (int k = threadIdx.x; k < MTGP32_STATE_SIZE; k += blockDim.x) s.s[k] = global.s[k];
    if (threadIdx.x == 0) {
        s.offset = global.offset;
        s.pIdx = global.pIdx;
        s.k = global.k;
    }
    __syncthreads();
    for (size_t r = 0; r < rounds; r++) {
        const unsigned int v = curand(&s);
        if (out != nullptr) out[(r * sets + blockIdx.x) * MTGP_THREADS + threadIdx.x] = transform<unsigned int, Out>(v, p);
    }
    curandStateMtgp32_t &back = states[blockIdx.x];
    for (int k = threadIdx.x; k < MTGP32_STATE_SIZE; k += blockDim.x) back.s[k] = s.s[k];
    if (threadIdx.x == 0) back.offset = s.offset;
}

// ---- MT19937

__host__ __device__ inline uint32_t mtNext(uint32_t first, uint32_t second, uint32_t far) {
    const uint32_t y = (first & 0x80000000u) | (second & 0x7fffffffu);
    return far ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0u);
}

__device__ inline uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    return y ^ (y >> 18);
}

constexpr int JUMP_THREADS = 256, JUMP_SPAN = MT_N - MT_M;  // words of the sequence computed in parallel

// one block per jump: generator g + half = P(T) generator g. With x_0, x_1, ... the word sequence of generator g's state
// (x_0 .. x_623 its words, oldest first), T^i of it is the window x_i .. x_{i + 623}, so word k of P(T) applied to it is
// the XOR of x_{i + k} over P's coefficients i. The sequence up to x_{degree + 623} is computed in shared memory (in
// steps of 227 words, whose inputs are all older), then each thread XORs its words. States are generator-major
// [MT_GENERATORS][MT_N], oldest word first.
__global__ void mtJump(uint32_t *states, unsigned int half, const unsigned long long *poly, int degree) {
    extern __shared__ uint32_t x[];
    const unsigned int g = blockIdx.x;
    if (g >= half) return;
    const int length = degree + MT_N;
    for (int k = threadIdx.x; k < MT_N; k += JUMP_THREADS) x[k] = states[size_t(g) * MT_N + k];
    __syncthreads();
    for (int first = MT_N; first < length; first += JUMP_SPAN) {
        for (int i = first + threadIdx.x; i < first + JUMP_SPAN && i < length; i += JUMP_THREADS) {
            x[i] = mtNext(x[i - MT_N], x[i - MT_N + 1], x[i - MT_N + MT_M]);
        }
        __syncthreads();
    }
    constexpr int PER_THREAD = (MT_N + JUMP_THREADS - 1) / JUMP_THREADS;
    uint32_t acc[PER_THREAD] = {};
    for (int w = 0; w <= degree / 64; w++) {
        unsigned long long bits = poly[w];
        while (bits != 0) {
            const int i = w * 64 + __ffsll((long long) bits) - 1;
            bits &= bits - 1;
#pragma unroll
            for (int j = 0; j < PER_THREAD; j++) {
                const int k = threadIdx.x + j * JUMP_THREADS;
                if (k < MT_N) acc[j] ^= x[i + k];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < PER_THREAD; j++) {
        const int k = threadIdx.x + j * JUMP_THREADS;
        if (k < MT_N) states[size_t(g + half) * MT_N + k] = acc[j];
    }
}

// generator-major [MT_GENERATORS][MT_N] to word-major [MT_N][MT_GENERATORS]
__global__ void mtTranspose(const uint32_t *in, uint32_t *out) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(MT_N) * MT_GENERATORS;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t word = i / MT_GENERATORS, g = i % MT_GENERATORS;
        out[i] = in[g * MT_N + word];
    }
}

// round r: generator g's 8 results go to (r * MT_GENERATORS + g) * 8 (transformed; out may be nullptr: skipped); all
// generators are at word ptr
template<typename Out>
__global__ void mtRounds(uint32_t *mt, int ptr, Out *out, size_t rounds, DistParams p) {
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= MT_GENERATORS) return;
    for (size_t r = 0; r < rounds; r++) {
        Out v[MT_GROUP];
        uint32_t first = mt[size_t(ptr) * MT_GENERATORS + g];
#pragma unroll
        for (int j = 0; j < MT_GROUP; j++) {
            const int next = ptr + 1 == MT_N ? 0 : ptr + 1, far = ptr + MT_M >= MT_N ? ptr + MT_M - MT_N : ptr + MT_M;
            const uint32_t second = mt[size_t(next) * MT_GENERATORS + g];
            const uint32_t word = mtNext(first, second, mt[size_t(far) * MT_GENERATORS + g]);
            mt[size_t(ptr) * MT_GENERATORS + g] = word;
            v[j] = transform<unsigned int, Out>(mtTemper(word), p);
            first = second;
            ptr = next;
        }
        if (out != nullptr) {
#pragma unroll
            for (int j = 0; j < MT_GROUP; j++) out[(r * MT_GENERATORS + g) * MT_GROUP + j] = v[j];
        }
    }
}

void mtInitGenrand(uint32_t *mt, uint32_t seed) {
    mt[0] = seed;
    for (int i = 1; i < MT_N; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i);
}

void mtInitByArray(uint32_t *mt, const uint32_t *key, int length) {
    mtInitGenrand(mt, 19650218u);
    int i = 1, j = 0;
    for (int k = std::max(MT_N, length); k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + uint32_t(j);
        i++;
        j++;
        if (i >= MT_N) {
            mt[0] = mt[MT_N - 1];
            i = 1;
        }
        if (j >= length) j = 0;
    }
    for (int k = MT_N - 1; k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - uint32_t(i);
        i++;
        if (i >= MT_N) {
            mt[0] = mt[MT_N - 1];
            i = 1;
        }
    }
    mt[0] = 0x80000000u;
}

// ---- Sobol'

// result i of `points` per dimension: dimension i / points, point first + i % points
template<typename X, typename Out>
__global__ void sobol(const X *directions, const X *scramble, unsigned int dims, size_t points,
                      unsigned long long first, Out *out, DistParams p) {
    constexpr int BITS = 8 * sizeof(X);
    const size_t n = size_t(dims) * points;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t d = i / points;
        const unsigned long long index = first + i % points;
        X gray = X(index ^ (index >> 1));
        X x = scramble != nullptr ? scramble[d] : X(0);
        const X *v = directions + d * BITS;
        for (int k = 0; gray != 0; k++, gray >>= 1) {
            if (gray & 1) x ^= v[k];
        }
        out[i] = transform<X, Out>(x, p);
    }
}

template<typename X>
struct SobolSet {
    std::vector<X> directions, scrambled, constants;
};

// Joe and Kuo's direction numbers; scrambled: each dimension's generator matrix multiplied by a random lower triangular
// matrix with a unit diagonal (Matousek's linear scrambling), and random constants
template<typename X>
const SobolSet<X> &sobolSet() {
    static SobolSet<X> set;
    static std::once_flag once;
    std::call_once(once, [] {
        constexpr int BITS = 8 * sizeof(X);
        set.directions.assign(size_t(SOBOL_DIMENSIONS) * BITS, 0);
        for (int k = 0; k < BITS; k++) set.directions[k] = X(1) << (BITS - 1 - k);
        size_t bit = 0;
        auto read = [&](int width) {
            uint32_t v = 0;
            for (int b = 0; b < width; b++, bit++) v |= ((SOBOL_PACKED[bit / 32] >> (bit % 32)) & 1u) << b;
            return v;
        };
        for (int d = 1; d < SOBOL_DIMENSIONS; d++) {
            const int s = int(read(5));
            const uint32_t a = read(s - 1);
            X *v = &set.directions[size_t(d) * BITS];
            for (int i = 1; i <= s; i++) {
                const X m = read(i);
                if (i <= BITS) v[i - 1] = m << (BITS - i);
            }
            for (int i = s + 1; i <= BITS; i++) {
                X value = v[i - s - 1] ^ (v[i - s - 1] >> s);
                for (int k = 1; k < s; k++) {
                    if ((a >> (s - 1 - k)) & 1) value ^= v[i - k - 1];
                }
                v[i - 1] = value;
            }
        }
        uint64_t random = BITS == 32 ? 0x5eed5c4a3b1e3232ull : 0x5eed5c4a3b1e6464ull;
        set.scrambled.resize(set.directions.size());
        set.constants.resize(SOBOL_DIMENSIONS);
        for (int d = 0; d < SOBOL_DIMENSIONS; d++) {
            // row r of the matrix (bit BITS - 1 - r of a vector) combines rows <= r
            X rows[BITS];
            for (int r = 0; r < BITS; r++) {
                const X above = r == 0 ? X(0) : X(~X(0) << (BITS - r));
                rows[r] = (X(1) << (BITS - 1 - r)) | (X(splitmix64(random)) & above);
            }
            for (int k = 0; k < BITS; k++) {
                const X v = set.directions[size_t(d) * BITS + k];
                X out = 0;
                for (int r = 0; r < BITS; r++) {
                    if (__builtin_popcountll((unsigned long long) (rows[r] & v)) & 1) out |= X(1) << (BITS - 1 - r);
                }
                set.scrambled[size_t(d) * BITS + k] = out;
            }
            set.constants[d] = X(splitmix64(random));
        }
    });
    return set;
}

// ---- discrete distributions

struct DiscreteBlock {
    curandDiscreteDistribution_st distribution;
    curandDistributionM2Shift_st m2;
    curandHistogramM2_st histogram;
};

std::mutex g_discreteMutex;
std::map<curandDiscreteDistribution_t, char *> g_discreteHost;  // device block -> host copy

// the distribution of probabilities pmf[k] of shift + k (a square histogram), or a normal approximation
curandStatus_t makeDiscrete(const std::vector<double> &pmf, unsigned int shift, double mean, double stddev,
                            bool gaussian, curandDiscreteDistribution_t *result) {
    const size_t n = gaussian ? 0 : pmf.size();
    const size_t vOffset = (sizeof(DiscreteBlock) + 15) / 16 * 16, kOffset = vOffset + n * sizeof(double);
    const size_t bytes = kOffset + n * sizeof(unsigned int);
    char *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) {
        cudaGetLastError();
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    char *host = static_cast<char *>(calloc(1, bytes));
    if (host == nullptr) {
        cudaFree(device);
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    auto *block = reinterpret_cast<DiscreteBlock *>(host);
    auto *v = reinterpret_cast<double *>(host + vOffset);
    auto *k = reinterpret_cast<unsigned int *>(host + kOffset);
    if (n > 0) {
        // Robin Hood: the smallest bar is topped up from the largest
        const double a = 1.0 / double(n);
        std::vector<double> q = pmf;
        for (size_t i = 0; i < n; i++) {
            k[i] = unsigned(i);
            v[i] = double(i + 1) * a;
        }
        for (size_t step = 0; step + 1 < n; step++) {
            size_t lo = 0, hi = 0;
            for (size_t i = 1; i < n; i++) {
                if (q[i] < q[lo]) lo = i;
                if (q[i] > q[hi]) hi = i;
            }
            if (q[hi] - q[lo] < 1e-300) break;
            k[lo] = unsigned(hi);
            v[lo] = double(lo) * a + q[lo];
            q[hi] -= a - q[lo];
            q[lo] = a;
        }
    }
    block->distribution.self_host_ptr = reinterpret_cast<curandDiscreteDistribution_t>(host);
    block->distribution.M2 = reinterpret_cast<curandDistributionM2Shift_t>(device + offsetof(DiscreteBlock, m2));
    block->distribution.host_M2 = &block->m2;
    block->distribution.stddev = stddev;
    block->distribution.mean = mean;
    block->distribution.method = gaussian ? CURAND_DISCRETE_GAUSS : CURAND_M2;
    block->m2.histogram = reinterpret_cast<curandHistogramM2_t>(device + offsetof(DiscreteBlock, histogram));
    block->m2.host_histogram = &block->histogram;
    block->m2.shift = shift;
    block->m2.length = unsigned(n);
    block->histogram.V = reinterpret_cast<curandHistogramM2V_t>(device + vOffset);
    block->histogram.host_V = v;
    block->histogram.K = reinterpret_cast<curandHistogramM2K_t>(device + kOffset);
    block->histogram.host_K = k;
    if (cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(device);
        free(host);
        return CURAND_STATUS_INITIALIZATION_FAILED;
    }
    *result = reinterpret_cast<curandDiscreteDistribution_t>(device);
    std::lock_guard lock(g_discreteMutex);
    g_discreteHost[*result] = host;
    return CURAND_STATUS_SUCCESS;
}

// probabilities of [lo, hi] from their logarithms, dropping the negligible tails
curandStatus_t discreteFromLog(long long lo, long long hi, double mean, double stddev,
                               double (*logp)(long long, double, double), double a, double b,
                               curandDiscreteDistribution_t *result) {
    std::vector<double> p;
    double top = -INFINITY;
    for (long long x = lo; x <= hi; x++) top = std::max(top, logp(x, a, b));
    long long first = -1, last = -1;
    for (long long x = lo; x <= hi; x++) {
        if (logp(x, a, b) - top > -40) {
            if (first < 0) first = x;
            last = x;
        }
    }
    double sum = 0;
    for (long long x = first; x <= last; x++) {
        p.push_back(std::exp(logp(x, a, b) - top));
        sum += p.back();
    }
    for (double &v: p) v /= sum;
    return makeDiscrete(p, unsigned(first), mean, stddev, false, result);
}

double poissonLog(long long k, double lambda, double) {
    return double(k) * std::log(lambda) - lambda - std::lgamma(double(k) + 1);
}

double binomialLog(long long k, double n, double p) {
    if (p == 0) return k == 0 ? 0 : -INFINITY;
    if (p == 1) return k == (long long) n ? 0 : -INFINITY;
    return std::lgamma(n + 1) - std::lgamma(double(k) + 1) - std::lgamma(n - double(k) + 1) + double(k) * std::log(p) +
           (n - double(k)) * std::log1p(-p);
}

curandStatus_t makePoisson(double lambda, curandDiscreteDistribution_t *result) {
    if (!(lambda > 0) || lambda > 400000) return CURAND_STATUS_OUT_OF_RANGE;
    if (lambda > GAUSSIAN_ABOVE) return makeDiscrete({}, 0, lambda, std::sqrt(lambda), true, result);
    const double sd = std::sqrt(lambda);
    return discreteFromLog(std::max(0ll, (long long) std::floor(lambda - 20 * sd - 10)),
                           (long long) std::ceil(lambda + 20 * sd + 30), lambda, sd, poissonLog, lambda, 0, result);
}

curandStatus_t makeBinomial(unsigned int n, double p, curandDiscreteDistribution_t *result) {
    if (!(p >= 0 && p <= 1)) return CURAND_STATUS_OUT_OF_RANGE;
    const double mean = double(n) * p, variance = mean * (1 - p), sd = std::sqrt(variance);
    if (variance > GAUSSIAN_ABOVE) return makeDiscrete({}, 0, mean, sd, true, result);
    return discreteFromLog(std::max(0ll, (long long) std::floor(mean - 20 * sd - 10)),
                           std::min((long long) n, (long long) std::ceil(mean + 20 * sd + 30)), mean, sd, binomialLog,
                           double(n), p, result);
}

curandStatus_t destroyDiscrete(curandDiscreteDistribution_t d) {
    char *host;
    {
        std::lock_guard lock(g_discreteMutex);
        auto it = g_discreteHost.find(d);
        if (it == g_discreteHost.end()) return CURAND_STATUS_NOT_INITIALIZED;
        host = it->second;
        g_discreteHost.erase(it);
    }
    cudaFree(d);
    free(host);
    return CURAND_STATUS_SUCCESS;
}

// ---- generators

struct DeviceGuard {
    int previous = -1;

    explicit DeviceGuard(int device) {
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
    }

    ~DeviceGuard() {
        int now = -1;
        cudaGetDevice(&now);
        if (previous >= 0 && now != previous) cudaSetDevice(previous);
    }
};

unsigned int blocksFor(size_t threads, unsigned int block = 256) {
    return unsigned((threads + block - 1) / block);
}

unsigned int gridFor(size_t n) {
    return unsigned(std::max<size_t>(1, std::min<size_t>((n + 255) / 256, 4096)));
}

} // namespace

struct curandGenerator_st {
    uint32_t magic = MAGIC;
    curandRngType_t type;
    bool host;
    int device = 0;
    cudaStream_t stream = nullptr;
    unsigned long long seed = 0, offset = 0;
    curandOrdering_t ordering;
    unsigned int dimensions = 1;
    bool ready = false;
    unsigned long long position = 0;  // results (pseudo) or points (quasi) since the start of the sequence
    // XORWOW, MRG32k3a, Philox: per thread states; MTGP32 per parameter set; MT19937 word-major
    void *states = nullptr;
    unsigned int threads = 0, group = 1;
    mtgp32_kernel_params_t *mtgpParams = nullptr;
    int mtPtr = 0;
    // MTGP32, MT19937: the current round of results, consumed up to roundUsed
    unsigned int *round = nullptr;
    size_t roundSize = 0, roundUsed = 0;
    // Sobol': this generator's dimensions' direction vectors and scramble constants
    void *directions = nullptr, *scramble = nullptr;
    // the last Poisson / binomial distribution
    curandDiscreteDistribution_t discrete = nullptr;
    int discreteKind = 0;
    double discreteA = 0, discreteB = 0;
    // results of host generators
    void *hostScratch = nullptr;
    size_t hostScratchBytes = 0;
};

namespace {

bool valid(curandGenerator_t g) {
    return g != nullptr && g->magic == MAGIC;
}

curandStatus_t cudaStatus(cudaError_t e) {
    if (e == cudaSuccess) return CURAND_STATUS_SUCCESS;
    cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? CURAND_STATUS_ALLOCATION_FAILED : CURAND_STATUS_LAUNCH_FAILURE;
}

curandStatus_t launched() {
    return cudaStatus(cudaGetLastError());
}

curandStatus_t ensure(void *&buffer, size_t &capacity, size_t bytes) {
    if (capacity >= bytes) return CURAND_STATUS_SUCCESS;
    cudaFree(buffer);
    buffer = nullptr;
    capacity = 0;
    if (cudaMalloc(&buffer, bytes) != cudaSuccess) {
        cudaGetLastError();
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    capacity = bytes;
    return CURAND_STATUS_SUCCESS;
}

void release(curandGenerator_t g) {
    cudaStreamSynchronize(g->stream);
    cudaFree(g->states);
    cudaFree(g->mtgpParams);
    cudaFree(g->round);
    cudaFree(g->directions);
    cudaFree(g->scramble);
    g->states = nullptr;
    g->mtgpParams = nullptr;
    g->round = nullptr;
    g->directions = g->scramble = nullptr;
    g->ready = false;
}

template<typename Out>
curandStatus_t generateRounds(curandGenerator_t g, Out *out, size_t rounds, const DistParams &p) {
    if (rounds == 0) return CURAND_STATUS_SUCCESS;
    if (g->type == CURAND_RNG_PSEUDO_MTGP32) {
        mtgpRounds<Out><<<g->threads, MTGP_THREADS, 0, g->stream>>>(static_cast<curandStateMtgp32_t *>(g->states),
                                                                     g->threads, out, rounds, p);
    } else {
        mtRounds<Out><<<blocksFor(MT_GENERATORS), 256, 0, g->stream>>>(static_cast<uint32_t *>(g->states), g->mtPtr,
                                                                         out, rounds, p);
        g->mtPtr = int((g->mtPtr + (rounds % MT_N) * MT_GROUP) % MT_N);
    }
    return launched();
}

// the next n results of MTGP32 / MT19937 (out may be nullptr: skipped): the rest of the cached round, whole rounds,
// then a new cached round's first values
template<typename Out>
curandStatus_t rawStream(curandGenerator_t g, Out *out, size_t n, const DistParams &p) {
    const size_t cached = std::min(n, g->roundSize - g->roundUsed);
    if (cached > 0 && out != nullptr) {
        transformRaw<Out><<<gridFor(cached), 256, 0, g->stream>>>(g->round + g->roundUsed, cached, out, p);
    }
    g->roundUsed += cached;
    n -= cached;
    if (out != nullptr) out += cached;
    curandStatus_t st = generateRounds(g, out, n / g->roundSize, p);
    if (st != CURAND_STATUS_SUCCESS) return st;
    const size_t rest = n % g->roundSize;
    if (rest > 0) {
        if (out != nullptr) out += n - rest;
        st = generateRounds<unsigned int>(g, g->round, 1, DistParams{Dist::BITS, 0, 0, nullptr});
        if (st != CURAND_STATUS_SUCCESS) return st;
        if (out != nullptr) transformRaw<Out><<<gridFor(rest), 256, 0, g->stream>>>(g->round, rest, out, p);
        g->roundUsed = rest;
    }
    return launched();
}

curandStatus_t seedMt19937(curandGenerator_t g) {
    std::vector<uint32_t> first(MT_N);
    if ((g->seed >> 32) == 0) {
        mtInitGenrand(first.data(), uint32_t(g->seed));
    } else {
        const uint32_t key[2] = {uint32_t(g->seed), uint32_t(g->seed >> 32)};
        mtInitByArray(first.data(), key, 2);
    }
    uint32_t *byGenerator = nullptr;
    unsigned long long *polys = nullptr;
    const size_t stateBytes = size_t(MT_N) * MT_GENERATORS * sizeof(uint32_t);
    curandStatus_t st = CURAND_STATUS_SUCCESS;
    if (cudaMalloc(&byGenerator, stateBytes) != cudaSuccess || cudaMalloc(&polys, sizeof(MT_JUMP)) != cudaSuccess ||
        cudaMalloc(&g->states, stateBytes) != cudaSuccess) {
        st = CURAND_STATUS_ALLOCATION_FAILED;
    }
    if (st == CURAND_STATUS_SUCCESS) {
        cudaMemcpyAsync(byGenerator, first.data(), MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice, g->stream);
        cudaMemcpyAsync(polys, MT_JUMP, sizeof(MT_JUMP), cudaMemcpyHostToDevice, g->stream);
        const size_t shared = size_t(MT_JUMP_WORDS * 64 + MT_N) * sizeof(uint32_t);
        cudaFuncSetAttribute(mtJump, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
        for (int level = 0; level < MT_JUMP_LEVELS; level++) {
            int degree = MT_JUMP_WORDS * 64 - 1;
            while (degree > 0 && !((MT_JUMP[level][degree / 64] >> (degree % 64)) & 1)) degree--;
            const unsigned int half = 1u << level;
            mtJump<<<half, JUMP_THREADS, shared, g->stream>>>(byGenerator, half,
                                                               polys + size_t(level) * MT_JUMP_WORDS, degree);
        }
        mtTranspose<<<gridFor(size_t(MT_N) * MT_GENERATORS), 256, 0, g->stream>>>(byGenerator,
                                                                                  static_cast<uint32_t *>(g->states));
        st = cudaStatus(cudaStreamSynchronize(g->stream));
    }
    cudaFree(byGenerator);
    cudaFree(polys);
    cudaGetLastError();
    g->mtPtr = 0;
    return st;
}

template<typename S>
curandStatus_t initStateful(curandGenerator_t g, unsigned int threads, unsigned int group, bool seeded) {
    g->threads = threads;
    g->group = group;
    if (cudaMalloc(&g->states, size_t(threads) * sizeof(S)) != cudaSuccess) {
        cudaGetLastError();
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    initStates<S><<<blocksFor(threads), 256, 0, g->stream>>>(static_cast<S *>(g->states), threads, group, g->seed,
                                                             g->offset, seeded);
    return launched();
}

curandStatus_t setup(curandGenerator_t g) {
    if (g->ready) return CURAND_STATUS_SUCCESS;
    release(g);
    g->position = g->offset;
    curandStatus_t st = CURAND_STATUS_SUCCESS;
    const bool legacy = g->ordering == CURAND_ORDERING_PSEUDO_LEGACY;
    switch (g->type) {
        case CURAND_RNG_PSEUDO_XORWOW:
            st = initStateful<curandStateXORWOW_t>(g, XORWOW_STATES, 1, g->ordering == CURAND_ORDERING_PSEUDO_SEEDED);
            break;
        case CURAND_RNG_PSEUDO_MRG32K3A:
            st = initStateful<curandStateMRG32k3a_t>(g, legacy ? MRG_LEGACY_STATES : MRG_STATES, 1, false);
            break;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
            st = initStateful<curandStatePhilox4_32_10_t>(g, PHILOX_STATES, 4, false);
            break;
        case CURAND_RNG_PSEUDO_MTGP32: {
            g->threads = legacy ? MTGP_LEGACY_SETS : MTGP_SETS;
            if (cudaMalloc(&g->mtgpParams, sizeof(mtgp32_kernel_params_t)) != cudaSuccess ||
                cudaMalloc(&g->states, g->threads * sizeof(curandStateMtgp32_t)) != cudaSuccess) {
                st = CURAND_STATUS_ALLOCATION_FAILED;
                break;
            }
            // (synchronous copies on the null stream)
            cudaStreamSynchronize(g->stream);
            st = curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, g->mtgpParams);
            if (st == CURAND_STATUS_SUCCESS) {
                st = curandMakeMTGP32KernelState(static_cast<curandStateMtgp32_t *>(g->states),
                                                 mtgp32dc_params_fast_11213, g->mtgpParams, int(g->threads), g->seed);
            }
            g->roundSize = size_t(g->threads) * MTGP_THREADS;
            break;
        }
        case CURAND_RNG_PSEUDO_MT19937:
            st = seedMt19937(g);
            g->roundSize = size_t(MT_GENERATORS) * MT_GROUP;
            break;
        default: {
            // Sobol'
            const bool wide = is64(g->type);
            const size_t bits = wide ? 64 : 32, element = wide ? 8 : 4;
            const void *directions, *constants;
            if (wide) {
                const SobolSet<uint64_t> &set = sobolSet<uint64_t>();
                directions = isScrambled(g->type) ? set.scrambled.data() : set.directions.data();
                constants = set.constants.data();
            } else {
                const SobolSet<uint32_t> &set = sobolSet<uint32_t>();
                directions = isScrambled(g->type) ? set.scrambled.data() : set.directions.data();
                constants = set.constants.data();
            }
            if (cudaMalloc(&g->directions, g->dimensions * bits * element) != cudaSuccess ||
                (isScrambled(g->type) && cudaMalloc(&g->scramble, g->dimensions * element) != cudaSuccess)) {
                st = CURAND_STATUS_ALLOCATION_FAILED;
                break;
            }
            // (synchronous: the tables are static, but the copies must be done before the call returns)
            if (cudaMemcpy(g->directions, directions, g->dimensions * bits * element, cudaMemcpyHostToDevice) !=
                        cudaSuccess ||
                (g->scramble != nullptr &&
                 cudaMemcpy(g->scramble, constants, g->dimensions * element, cudaMemcpyHostToDevice) != cudaSuccess)) {
                st = CURAND_STATUS_INITIALIZATION_FAILED;
            }
            break;
        }
    }
    if (st == CURAND_STATUS_SUCCESS && g->roundSize > 0) {
        if (cudaMalloc(&g->round, g->roundSize * sizeof(unsigned int)) != cudaSuccess) {
            st = CURAND_STATUS_ALLOCATION_FAILED;
        } else {
            // the offset: results skipped
            g->roundUsed = g->roundSize;
            st = rawStream<unsigned int>(g, nullptr, g->offset, DistParams{Dist::BITS, 0, 0, nullptr});
        }
    }
    cudaGetLastError();
    if (st != CURAND_STATUS_SUCCESS) {
        release(g);
        return st;
    }
    g->ready = true;
    return CURAND_STATUS_SUCCESS;
}

template<typename Out>
curandStatus_t generateDevice(curandGenerator_t g, Out *out, size_t n, const DistParams &p) {
    curandStatus_t st = setup(g);
    if (st != CURAND_STATUS_SUCCESS || n == 0) return st;
    switch (g->type) {
        case CURAND_RNG_PSEUDO_XORWOW:
            stateful<curandStateXORWOW_t, Out><<<blocksFor(g->threads), 256, 0, g->stream>>>(
                    static_cast<curandStateXORWOW_t *>(g->states), g->threads, g->group, g->position, n, out, p);
            break;
        case CURAND_RNG_PSEUDO_MRG32K3A:
            stateful<curandStateMRG32k3a_t, Out><<<blocksFor(g->threads), 256, 0, g->stream>>>(
                    static_cast<curandStateMRG32k3a_t *>(g->states), g->threads, g->group, g->position, n, out, p);
            break;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
            stateful<curandStatePhilox4_32_10_t, Out><<<blocksFor(g->threads), 256, 0, g->stream>>>(
                    static_cast<curandStatePhilox4_32_10_t *>(g->states), g->threads, g->group, g->position, n, out, p);
            break;
        case CURAND_RNG_PSEUDO_MTGP32:
        case CURAND_RNG_PSEUDO_MT19937:
            st = rawStream(g, out, n, p);
            if (st == CURAND_STATUS_SUCCESS) g->position += n;
            return st;
        default: {
            const size_t points = n / g->dimensions;
            if (is64(g->type)) {
                sobol<unsigned long long, Out><<<gridFor(n), 256, 0, g->stream>>>(
                        static_cast<const unsigned long long *>(g->directions),
                        static_cast<const unsigned long long *>(g->scramble), g->dimensions, points, g->position, out,
                        p);
            } else {
                sobol<unsigned int, Out><<<gridFor(n), 256, 0, g->stream>>>(
                        static_cast<const unsigned int *>(g->directions), static_cast<const unsigned int *>(g->scramble),
                        g->dimensions, points, g->position, out, p);
            }
            st = launched();
            if (st == CURAND_STATUS_SUCCESS) g->position += points;
            return st;
        }
    }
    st = launched();
    if (st == CURAND_STATUS_SUCCESS) g->position += n;
    return st;
}

// device generators write to `out`; host generators generate into device memory and copy the results
template<typename Out>
curandStatus_t generate(curandGenerator_t g, Out *out, size_t n, const DistParams &p) {
    if (!valid(g)) return CURAND_STATUS_NOT_INITIALIZED;
    if (out == nullptr && n > 0) return CURAND_STATUS_LAUNCH_FAILURE;
    DeviceGuard guard(g->device);
    if (!g->host) return generateDevice(g, out, n, p);
    curandStatus_t st = ensure(g->hostScratch, g->hostScratchBytes, std::max<size_t>(n, 1) * sizeof(Out));
    if (st != CURAND_STATUS_SUCCESS) return st;
    st = generateDevice(g, static_cast<Out *>(g->hostScratch), n, p);
    if (st != CURAND_STATUS_SUCCESS) return st;
    if (cudaMemcpyAsync(out, g->hostScratch, n * sizeof(Out), cudaMemcpyDeviceToHost, g->stream) != cudaSuccess ||
        cudaStreamSynchronize(g->stream) != cudaSuccess) {
        return cudaStatus(cudaGetLastError());
    }
    return CURAND_STATUS_SUCCESS;
}

// lengths: quasi-random results come in whole points; pseudo-random normals in Box-Muller pairs
curandStatus_t checkLength(curandGenerator_t g, size_t n, bool pairs) {
    if (!valid(g)) return CURAND_STATUS_NOT_INITIALIZED;
    if (isQuasi(g->type)) return n % g->dimensions == 0 ? CURAND_STATUS_SUCCESS : CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    return pairs && n % 2 != 0 ? CURAND_STATUS_LENGTH_NOT_MULTIPLE : CURAND_STATUS_SUCCESS;
}

template<typename Out>
curandStatus_t generateChecked(curandGenerator_t g, Out *out, size_t n, Dist dist, double mean, double stddev) {
    curandStatus_t st = checkLength(g, n, dist == Dist::NORMAL || dist == Dist::LOG_NORMAL);
    if (st != CURAND_STATUS_SUCCESS) return st;
    if (is64(g->type) && std::is_same_v<Out, unsigned int>) return CURAND_STATUS_TYPE_ERROR;
    return generate(g, out, n, DistParams{dist, mean, stddev, nullptr});
}

// Poisson (kind 1) or binomial (kind 2) results from the generator's cached distribution
curandStatus_t generateDiscrete(curandGenerator_t g, unsigned int *out, size_t n, int kind, double a, double b) {
    curandStatus_t st = checkLength(g, n, false);
    if (st != CURAND_STATUS_SUCCESS) return st;
    DeviceGuard guard(g->device);
    if (g->discrete == nullptr || g->discreteKind != kind || g->discreteA != a || g->discreteB != b) {
        curandDiscreteDistribution_t d = nullptr;
        st = kind == 1 ? makePoisson(a, &d) : makeBinomial((unsigned int) a, b, &d);
        if (st != CURAND_STATUS_SUCCESS) return st;
        if (g->discrete != nullptr) {
            cudaStreamSynchronize(g->stream);
            destroyDiscrete(g->discrete);
        }
        g->discrete = d;
        g->discreteKind = kind;
        g->discreteA = a;
        g->discreteB = b;
    }
    return generate(g, out, n, DistParams{Dist::DISCRETE, 0, 0, g->discrete});
}

bool orderingValid(const curandGenerator_t g, curandOrdering_t o) {
    if (isQuasi(g->type)) return o == CURAND_ORDERING_QUASI_DEFAULT;
    switch (o) {
        case CURAND_ORDERING_PSEUDO_DEFAULT:
        case CURAND_ORDERING_PSEUDO_LEGACY:
            return true;
        case CURAND_ORDERING_PSEUDO_BEST:
            return !(g->type == CURAND_RNG_PSEUDO_MT19937 && g->host);
        case CURAND_ORDERING_PSEUDO_SEEDED:
            return g->type == CURAND_RNG_PSEUDO_XORWOW;
        case CURAND_ORDERING_PSEUDO_DYNAMIC:
            return !g->host && g->type != CURAND_RNG_PSEUDO_MT19937;
        default:
            return false;
    }
}

curandStatus_t create(curandGenerator_t *generator, curandRngType_t type, bool host) {
    if (generator == nullptr) return CURAND_STATUS_NOT_INITIALIZED;
    if (type == CURAND_RNG_PSEUDO_DEFAULT) type = CURAND_RNG_PSEUDO_XORWOW;
    if (type == CURAND_RNG_QUASI_DEFAULT) type = CURAND_RNG_QUASI_SOBOL32;
    switch (type) {
        case CURAND_RNG_PSEUDO_XORWOW:
        case CURAND_RNG_PSEUDO_MRG32K3A:
        case CURAND_RNG_PSEUDO_MTGP32:
        case CURAND_RNG_PSEUDO_MT19937:
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
        case CURAND_RNG_QUASI_SOBOL32:
        case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32:
        case CURAND_RNG_QUASI_SOBOL64:
        case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64:
            break;
        default:
            return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaFree(nullptr) != cudaSuccess) {
        cudaGetLastError();
        return CURAND_STATUS_INITIALIZATION_FAILED;
    }
    auto *g = new curandGenerator_st;
    g->type = type;
    g->host = host;
    cudaGetDevice(&g->device);
    g->ordering = isQuasi(type) ? CURAND_ORDERING_QUASI_DEFAULT : CURAND_ORDERING_PSEUDO_DEFAULT;
    *generator = g;
    return CURAND_STATUS_SUCCESS;
}

} // namespace

extern "C" {

curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    return create(generator, rng_type, false);
}

curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type) {
    return create(generator, rng_type, true);
}

curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    {
        DeviceGuard guard(generator->device);
        release(generator);
        if (generator->discrete != nullptr) destroyDiscrete(generator->discrete);
        cudaFree(generator->hostScratch);
    }
    generator->magic = 0;
    delete generator;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGetVersion(int *version) {
    if (version == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    *version = CURAND_VERSION;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    switch (type) {
        case MAJOR_VERSION: *value = CURAND_VER_MAJOR; return CURAND_STATUS_SUCCESS;
        case MINOR_VERSION: *value = CURAND_VER_MINOR; return CURAND_STATUS_SUCCESS;
        case PATCH_LEVEL: *value = CURAND_VER_PATCH; return CURAND_STATUS_SUCCESS;
        default: return CURAND_STATUS_OUT_OF_RANGE;
    }
}

curandStatus_t CURANDAPI curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    if (!generator->host) generator->stream = stream;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    if (isQuasi(generator->type)) return CURAND_STATUS_TYPE_ERROR;
    generator->seed = seed;
    generator->ready = false;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    generator->offset = offset;
    generator->ready = false;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    if (!orderingValid(generator, order)) return CURAND_STATUS_OUT_OF_RANGE;
    generator->ordering = order;
    generator->ready = false;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                                 unsigned int num_dimensions) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    if (!isQuasi(generator->type)) return CURAND_STATUS_TYPE_ERROR;
    if (num_dimensions < 1 || num_dimensions > unsigned(SOBOL_DIMENSIONS)) return CURAND_STATUS_OUT_OF_RANGE;
    generator->dimensions = num_dimensions;
    generator->ready = false;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num) {
    return generateChecked(generator, outputPtr, num, Dist::BITS, 0, 0);
}

curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr,
                                                size_t num) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    if (!is64(generator->type)) return CURAND_STATUS_TYPE_ERROR;
    return generateChecked(generator, outputPtr, num, Dist::BITS, 0, 0);
}

curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) {
    return generateChecked(generator, outputPtr, num, Dist::UNIFORM, 0, 0);
}

curandStatus_t CURANDAPI curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num) {
    return generateChecked(generator, outputPtr, num, Dist::UNIFORM, 0, 0);
}

curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean,
                                              float stddev) {
    return generateChecked(generator, outputPtr, n, Dist::NORMAL, mean, stddev);
}

curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n,
                                                    double mean, double stddev) {
    return generateChecked(generator, outputPtr, n, Dist::NORMAL, mean, stddev);
}

curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean,
                                                 float stddev) {
    return generateChecked(generator, outputPtr, n, Dist::LOG_NORMAL, mean, stddev);
}

curandStatus_t CURANDAPI curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n,
                                                       double mean, double stddev) {
    return generateChecked(generator, outputPtr, n, Dist::LOG_NORMAL, mean, stddev);
}

curandStatus_t CURANDAPI curandCreatePoissonDistribution(double lambda,
                                                         curandDiscreteDistribution_t *discrete_distribution) {
    if (discrete_distribution == nullptr) return CURAND_STATUS_NOT_INITIALIZED;
    return makePoisson(lambda, discrete_distribution);
}

curandStatus_t CURANDAPI curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
    return destroyDiscrete(discrete_distribution);
}

curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t n,
                                               double lambda) {
    if (!(lambda > 0) || lambda > 400000) return CURAND_STATUS_OUT_OF_RANGE;
    return generateDiscrete(generator, outputPtr, n, 1, lambda, 0);
}

curandStatus_t CURANDAPI curandGeneratePoissonMethod(curandGenerator_t generator, unsigned int *outputPtr, size_t n,
                                                     double lambda, curandMethod_t) {
    return curandGeneratePoisson(generator, outputPtr, n, lambda);
}

curandStatus_t CURANDAPI curandGenerateBinomial(curandGenerator_t generator, unsigned int *outputPtr, size_t num,
                                                unsigned int n, double p) {
    if (!(p >= 0 && p <= 1)) return CURAND_STATUS_OUT_OF_RANGE;
    return generateDiscrete(generator, outputPtr, num, 2, double(n), p);
}

curandStatus_t CURANDAPI curandGenerateBinomialMethod(curandGenerator_t generator, unsigned int *outputPtr,
                                                      size_t num, unsigned int n, double p, curandMethod_t) {
    return curandGenerateBinomial(generator, outputPtr, num, n, p);
}

curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t generator) {
    if (!valid(generator)) return CURAND_STATUS_NOT_INITIALIZED;
    DeviceGuard guard(generator->device);
    return setup(generator);
}

curandStatus_t CURANDAPI curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[],
                                                     curandDirectionVectorSet_t set) {
    if (vectors == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    const SobolSet<uint32_t> &s = sobolSet<uint32_t>();
    if (set == CURAND_DIRECTION_VECTORS_32_JOEKUO6) {
        *vectors = reinterpret_cast<curandDirectionVectors32_t *>(const_cast<uint32_t *>(s.directions.data()));
    } else if (set == CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6) {
        *vectors = reinterpret_cast<curandDirectionVectors32_t *>(const_cast<uint32_t *>(s.scrambled.data()));
    } else {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGetScrambleConstants32(unsigned int **constants) {
    if (constants == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    *constants = const_cast<unsigned int *>(sobolSet<uint32_t>().constants.data());
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[],
                                                     curandDirectionVectorSet_t set) {
    if (vectors == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    const SobolSet<uint64_t> &s = sobolSet<uint64_t>();
    if (set == CURAND_DIRECTION_VECTORS_64_JOEKUO6) {
        *vectors = reinterpret_cast<curandDirectionVectors64_t *>(const_cast<uint64_t *>(s.directions.data()));
    } else if (set == CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6) {
        *vectors = reinterpret_cast<curandDirectionVectors64_t *>(const_cast<uint64_t *>(s.scrambled.data()));
    } else {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t CURANDAPI curandGetScrambleConstants64(unsigned long long **constants) {
    if (constants == nullptr) return CURAND_STATUS_OUT_OF_RANGE;
    *constants = reinterpret_cast<unsigned long long *>(const_cast<uint64_t *>(sobolSet<uint64_t>().constants.data()));
    return CURAND_STATUS_SUCCESS;
}

} // extern "C"
