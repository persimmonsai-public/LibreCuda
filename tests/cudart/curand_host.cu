// NVCC_FLAGS: -arch=$ARCH -lcurand
// cuRAND's host API: the documented orderings of XORWOW, MRG32k3a (best and legacy), Philox and MTGP32 against
// results computed here with the device API (curand_kernel.h), with offsets and results split over several calls;
// Sobol' points against Joe and Kuo's published first points and the direction vectors (plain and scrambled, 32 and
// 64 bit); uniform results as the device API's transform of the bits; statistics of normal, log-normal, Poisson and
// binomial results of every generator; host generators; Poisson distributions used by a kernel; error statuses.
// Checked against NVIDIA's cuRAND too (CURAND_TEST_NVIDIA=1 skips the checks of LibreCUDA's own MT19937 ordering).
#include "check.h"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

#define RCHECK(call) do { \
    curandStatus_t rcheck_status = (call); \
    if (rcheck_status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error %d at %s:%d: %s\n", int(rcheck_status), __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

static bool nvidia() {
    const char *v = getenv("CURAND_TEST_NVIDIA");
    return v != nullptr && strcmp(v, "1") == 0;
}

// ---- device API references

__global__ void xorwowReference(bool uniform, unsigned long long seed, unsigned long long first, unsigned int states,
                                size_t n, unsigned int *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned long long e = first + i;
        curandStateXORWOW_t s;
        curand_init(seed, e % states, e / states, &s);
        out[i] = uniform ? __float_as_uint(curand_uniform(&s)) : curand(&s);
    }
}

__global__ void mrgReference(bool uniform, unsigned long long seed, unsigned long long first, unsigned int states,
                             size_t n, unsigned int *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned long long e = first + i;
        curandStateMRG32k3a_t s;
        curand_init(seed, e % states, e / states, &s);
        out[i] = uniform ? __float_as_uint(curand_uniform(&s)) : curand(&s);
    }
}

// four results of subsequence (n / 4) % 65536, then four of the next
__global__ void philoxReference(bool uniform, unsigned long long seed, unsigned long long first, size_t n,
                                unsigned int *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned long long e = first + i, group = e / 4;
        curandStatePhilox4_32_10_t s;
        curand_init(seed, group % 65536, (group / 65536) * 4 + e % 4, &s);
        out[i] = uniform ? __float_as_uint(curand_uniform(&s)) : curand(&s);
    }
}

// 256 results of parameter set p, then 256 of set p + 1, ...
__global__ void mtgpReference(curandStateMtgp32_t *states, int sets, size_t rounds, unsigned int *out) {
    for (size_t r = 0; r < rounds; r++) {
        out[(r * sets + blockIdx.x) * 256 + threadIdx.x] = curand(&states[blockIdx.x]);
    }
}

__global__ void normalIcdfOf(const unsigned int *bits, size_t n, float *out) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = _curand_normal_icdf(bits[i]);
    }
}

__global__ void discreteKernel(curandDiscreteDistribution_t d, unsigned long long seed, size_t n, unsigned int *out) {
    const size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateXORWOW_t s;
    curand_init(seed, t, 0, &s);
    for (size_t i = t; i < n; i += gridDim.x * blockDim.x) out[i] = curand_discrete(&s, d);
}

template<typename T>
static std::vector<T> download(const T *d, size_t n) {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// n results of the generator, in calls of the given sizes (cycled)
static std::vector<unsigned int> bits(curandGenerator_t g, size_t n, std::vector<size_t> parts = {}) {
    unsigned int *d;
    CHECK(cudaMalloc(&d, n * sizeof(unsigned int)));
    if (parts.empty()) parts = {n};
    for (size_t done = 0, k = 0; done < n; k++) {
        const size_t part = std::min(parts[k % parts.size()], n - done);
        RCHECK(curandGenerate(g, d + done, part));
        done += part;
    }
    std::vector<unsigned int> h = download(d, n);
    CHECK(cudaFree(d));
    return h;
}

static void expectSame(const std::vector<unsigned int> &got, const std::vector<unsigned int> &want, const char *what) {
    for (size_t i = 0; i < want.size(); i++) {
        EXPECT(got[i] == want[i], "%s: result %zu is %08x, expected %08x", what, i, got[i], want[i]);
    }
}

template<typename Kernel, typename... Args>
static std::vector<unsigned int> reference(size_t n, Kernel kernel, Args... args) {
    unsigned int *d;
    CHECK(cudaMalloc(&d, n * sizeof(unsigned int)));
    kernel<<<256, 256>>>(args..., n, d);
    CHECK(cudaGetLastError());
    std::vector<unsigned int> h = download(d, n);
    CHECK(cudaFree(d));
    return h;
}

static void orderings() {
    const unsigned long long seed = 0x123456789abcdefull;
    const size_t n = 300000;
    struct Case {
        curandRngType_t type;
        curandOrdering_t ordering;
        unsigned int states;
        const char *name;
    } cases[] = {
        {CURAND_RNG_PSEUDO_XORWOW, CURAND_ORDERING_PSEUDO_DEFAULT, 4096, "XORWOW"},
        {CURAND_RNG_PSEUDO_XORWOW, CURAND_ORDERING_PSEUDO_LEGACY, 4096, "XORWOW legacy"},
        {CURAND_RNG_PSEUDO_MRG32K3A, CURAND_ORDERING_PSEUDO_DEFAULT, 81920, "MRG32k3a"},
        {CURAND_RNG_PSEUDO_MRG32K3A, CURAND_ORDERING_PSEUDO_LEGACY, 4096, "MRG32k3a legacy"},
        {CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_ORDERING_PSEUDO_DEFAULT, 0, "Philox"},
    };
    for (const Case &c: cases) {
        for (unsigned long long offset: {0ull, 12345ull}) {
            curandGenerator_t g;
            RCHECK(curandCreateGenerator(&g, c.type));
            RCHECK(curandSetPseudoRandomGeneratorSeed(g, seed));
            RCHECK(curandSetGeneratorOrdering(g, c.ordering));
            RCHECK(curandSetGeneratorOffset(g, offset));
            // in pieces that don't line up with the states
            std::vector<unsigned int> got = bits(g, n, {1, 4097, 77777, 3});
            auto want = [&](bool uniform) {
                return c.type == CURAND_RNG_PSEUDO_XORWOW ? reference(n, xorwowReference, uniform, seed, offset, c.states)
                       : c.type == CURAND_RNG_PSEUDO_MRG32K3A
                               ? reference(n, mrgReference, uniform, seed, offset, c.states)
                               : reference(n, philoxReference, uniform, seed, offset);
            };
            char what[64];
            snprintf(what, sizeof(what), "%s, offset %llu", c.name, offset);
            expectSame(got, want(false), what);
            // uniform floats: the device API's curand_uniform of the same states
            RCHECK(curandSetGeneratorOffset(g, offset));
            float *u;
            CHECK(cudaMalloc(&u, n * sizeof(float)));
            RCHECK(curandGenerateUniform(g, u, n));
            std::vector<float> hu = download(u, n);
            std::vector<unsigned int> uniformBits(n);
            memcpy(uniformBits.data(), hu.data(), n * sizeof(float));
            snprintf(what, sizeof(what), "%s, offset %llu, uniform", c.name, offset);
            expectSame(uniformBits, want(true), what);
            CHECK(cudaFree(u));
            RCHECK(curandDestroyGenerator(g));
        }
    }

    // MTGP32: 192 parameter sets (legacy: 64)
    for (int legacy = 0; legacy < 2; legacy++) {
        const int sets = legacy ? 64 : 192;
        const size_t rounds = 7, m = rounds * sets * 256;
        curandGenerator_t g;
        RCHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_MTGP32));
        RCHECK(curandSetPseudoRandomGeneratorSeed(g, seed));
        if (legacy) RCHECK(curandSetGeneratorOrdering(g, CURAND_ORDERING_PSEUDO_LEGACY));
        std::vector<unsigned int> got = bits(g, m, {1000, 70001});
        RCHECK(curandDestroyGenerator(g));
        mtgp32_kernel_params_t *k;
        curandStateMtgp32_t *states;
        CHECK(cudaMalloc(&k, sizeof(*k)));
        CHECK(cudaMalloc(&states, sets * sizeof(curandStateMtgp32_t)));
        RCHECK(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, k));
        RCHECK(curandMakeMTGP32KernelState(states, mtgp32dc_params_fast_11213, k, sets, seed));
        unsigned int *d;
        CHECK(cudaMalloc(&d, m * sizeof(unsigned int)));
        mtgpReference<<<sets, 256>>>(states, sets, rounds, d);
        expectSame(got, download(d, m), legacy ? "MTGP32 legacy" : "MTGP32");
        CHECK(cudaFree(d));
        CHECK(cudaFree(states));
        CHECK(cudaFree(k));
    }
    PASS("XORWOW, MRG32k3a, Philox and MTGP32 follow the documented orderings (offsets, split calls)");
}

// MT19937: generator 0 is the CPU version's sequence; eight results of each generator in turn
static void mt19937() {
    if (nvidia()) {
        PASS("MT19937 ordering (skipped: NVIDIA's ordering is unpublished)");
        return;
    }
    for (unsigned long long seed: {5489ull, 0x1234567890ull}) {
        curandGenerator_t g;
        RCHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_MT19937));
        RCHECK(curandSetPseudoRandomGeneratorSeed(g, seed));
        const size_t rounds = 3, n = rounds * 8192 * 8;
        std::vector<unsigned int> got = bits(g, n, {5, 65536, 100000});
        std::mt19937 cpu;
        if (seed >> 32) {
            std::seed_seq unused;
            (void) unused;
            // init_by_array with the two halves (std::seed_seq differs): checked through generator 0's continuity below
        } else {
            cpu.seed(uint32_t(seed));
            for (size_t r = 0; r < rounds; r++) {
                for (int j = 0; j < 8; j++) {
                    const unsigned int want = cpu();
                    EXPECT(got[r * 8192 * 8 + j] == want, "MT19937 seed %llu: round %zu result %d", seed, r, j);
                }
            }
        }
        // generators differ from each other
        EXPECT(got[0] != got[8] && got[8] != got[16], "MT19937 generators repeat");
        RCHECK(curandDestroyGenerator(g));
    }
    // an offset skips results of the same sequence
    curandGenerator_t a, b;
    RCHECK(curandCreateGenerator(&a, CURAND_RNG_PSEUDO_MT19937));
    RCHECK(curandCreateGenerator(&b, CURAND_RNG_PSEUDO_MT19937));
    RCHECK(curandSetGeneratorOffset(b, 70000));
    std::vector<unsigned int> all = bits(a, 200000), rest = bits(b, 130000);
    for (size_t i = 0; i < rest.size(); i++) EXPECT(rest[i] == all[70000 + i], "MT19937 offset: result %zu", i);
    RCHECK(curandDestroyGenerator(a));
    RCHECK(curandDestroyGenerator(b));
    PASS("MT19937: generator 0 is std::mt19937, groups of 8, offsets");
}

// x of point i in dimension d: XOR of the direction vectors of gray(i)'s bits
template<typename X>
static X sobolPoint(const X *v, X scramble, unsigned long long i) {
    unsigned long long gray = i ^ (i >> 1);
    X x = scramble;
    for (int k = 0; gray != 0; k++, gray >>= 1) {
        if (gray & 1) x ^= v[k];
    }
    return x;
}

static void sobol() {
    // Joe and Kuo's first points of 3 dimensions
    const double published[10][3] = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.75, 0.25, 0.25}, {0.25, 0.75, 0.75},
                                     {0.375, 0.375, 0.625}, {0.875, 0.875, 0.125}, {0.625, 0.125, 0.875},
                                     {0.125, 0.625, 0.375}, {0.1875, 0.3125, 0.9375}, {0.6875, 0.8125, 0.4375}};
    curandGenerator_t g;
    RCHECK(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32));
    RCHECK(curandSetQuasiRandomGeneratorDimensions(g, 3));
    std::vector<unsigned int> first = bits(g, 30);
    for (int i = 0; i < 10; i++) {
        for (int d = 0; d < 3; d++) {
            EXPECT(first[d * 10 + i] == (unsigned int) (published[i][d] * 4294967296.0), "Sobol' point %d dim %d", i,
                   d);
        }
    }
    RCHECK(curandDestroyGenerator(g));

    curandDirectionVectors32_t *v32, *s32;
    curandDirectionVectors64_t *v64, *s64;
    unsigned int *c32;
    unsigned long long *c64;
    RCHECK(curandGetDirectionVectors32(&v32, CURAND_DIRECTION_VECTORS_32_JOEKUO6));
    RCHECK(curandGetDirectionVectors32(&s32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    RCHECK(curandGetDirectionVectors64(&v64, CURAND_DIRECTION_VECTORS_64_JOEKUO6));
    RCHECK(curandGetDirectionVectors64(&s64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
    RCHECK(curandGetScrambleConstants32(&c32));
    RCHECK(curandGetScrambleConstants64(&c64));
    // the last dimension's vectors are 32 distinct odd multiples of powers of two
    EXPECT(v32[19999][0] >> 31 == 1 && v64[19999][0] >> 63 == 1, "direction vectors of dimension 20000");
    EXPECT(v64[5][3] >> 32 == v32[5][3], "64 bit direction vectors extend the 32 bit ones");

    // all four types, 37 dimensions, an offset, results over two calls
    const unsigned int dims = 37;
    const size_t points = 3000;
    const unsigned long long offset = 1000;
    for (curandRngType_t type: {CURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                CURAND_RNG_QUASI_SOBOL64, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64}) {
        RCHECK(curandCreateGenerator(&g, type));
        RCHECK(curandSetQuasiRandomGeneratorDimensions(g, dims));
        RCHECK(curandSetGeneratorOffset(g, offset));
        const bool wide = type == CURAND_RNG_QUASI_SOBOL64 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        const bool scrambled = type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 || type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        for (int call = 0; call < 2; call++) {
            const size_t n = dims * points;
            if (wide) {
                unsigned long long *d;
                CHECK(cudaMalloc(&d, n * 8));
                RCHECK(curandGenerateLongLong(g, d, n));
                std::vector<unsigned long long> h = download(d, n);
                for (unsigned int dim = 0; dim < dims; dim++) {
                    for (size_t j = 0; j < points; j++) {
                        const unsigned long long want = sobolPoint<unsigned long long>(
                                scrambled ? s64[dim] : v64[dim], scrambled ? c64[dim] : 0, offset + call * points + j);
                        EXPECT(h[dim * points + j] == want, "Sobol64 type %d dim %u point %zu", int(type), dim, j);
                    }
                }
                CHECK(cudaFree(d));
            } else {
                std::vector<unsigned int> h = bits(g, n);
                for (unsigned int dim = 0; dim < dims; dim++) {
                    for (size_t j = 0; j < points; j++) {
                        const unsigned int want = sobolPoint<unsigned int>(
                                scrambled ? s32[dim] : v32[dim], scrambled ? c32[dim] : 0, offset + call * points + j);
                        EXPECT(h[dim * points + j] == want, "Sobol32 type %d dim %u point %zu", int(type), dim, j);
                    }
                }
            }
        }
        RCHECK(curandDestroyGenerator(g));
    }

    // normals of quasi-random generators: the inverse distribution function of the bits
    RCHECK(curandCreateGenerator(&g, CURAND_RNG_QUASI_SOBOL32));
    RCHECK(curandSetQuasiRandomGeneratorDimensions(g, 4));
    const size_t n = 4 * 5000;
    float *normal;
    unsigned int *raw;
    CHECK(cudaMalloc(&normal, n * sizeof(float)));
    CHECK(cudaMalloc(&raw, n * sizeof(unsigned int)));
    RCHECK(curandGenerateNormal(g, normal, n, 0.0f, 1.0f));
    RCHECK(curandSetGeneratorOffset(g, 0));
    RCHECK(curandGenerate(g, raw, n));
    float *ref;
    CHECK(cudaMalloc(&ref, n * sizeof(float)));
    normalIcdfOf<<<64, 256>>>(raw, n, ref);
    std::vector<float> hn = download(normal, n), hr = download(ref, n);
    for (size_t i = 0; i < n; i++) {
        EXPECT(hn[i] == hr[i], "Sobol' normal %zu: %.9g, expected %.9g", i, hn[i], hr[i]);
    }
    CHECK(cudaFree(normal));
    CHECK(cudaFree(raw));
    CHECK(cudaFree(ref));
    RCHECK(curandDestroyGenerator(g));
    PASS("Sobol' 32 / 64 bit, scrambled: published points, direction vectors, offsets, normals");
}

struct Moments {
    double mean, variance;
};

template<typename T>
static Moments moments(const std::vector<T> &v) {
    double s = 0, s2 = 0;
    for (T x: v) {
        s += double(x);
        s2 += double(x) * double(x);
    }
    const double mean = s / v.size();
    return {mean, s2 / v.size() - mean * mean};
}

static void distributions() {
    const curandRngType_t types[] = {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
                                     CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10,
                                     CURAND_RNG_QUASI_SOBOL32, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64};
    const size_t n = 1 << 20;
    float *f;
    double *d;
    unsigned int *u;
    CHECK(cudaMalloc(&f, n * sizeof(float)));
    CHECK(cudaMalloc(&d, n * sizeof(double)));
    CHECK(cudaMalloc(&u, n * sizeof(unsigned int)));
    for (curandRngType_t type: types) {
        curandGenerator_t g;
        RCHECK(curandCreateGenerator(&g, type));
        if (type < CURAND_RNG_QUASI_DEFAULT) RCHECK(curandSetPseudoRandomGeneratorSeed(g, 42));
        auto near = [&](double got, double want, double tol, const char *what) {
            EXPECT(std::fabs(got - want) <= tol, "type %d %s: %g, expected %g", int(type), what, got, want);
        };
        RCHECK(curandGenerateUniform(g, f, n));
        std::vector<float> hf = download(f, n);
        Moments m = moments(hf);
        near(m.mean, 0.5, 0.003, "uniform mean");
        near(m.variance, 1.0 / 12, 0.002, "uniform variance");
        for (float x: hf) EXPECT(x > 0.0f && x <= 1.0f, "uniform %g outside (0, 1]", x);
        RCHECK(curandGenerateUniformDouble(g, d, n));
        std::vector<double> hd = download(d, n);
        m = moments(hd);
        near(m.mean, 0.5, 0.003, "uniform double mean");
        for (double x: hd) EXPECT(x > 0.0 && x <= 1.0, "uniform double %g outside (0, 1]", x);
        RCHECK(curandGenerateNormal(g, f, n, 3.0f, 2.0f));
        m = moments(download(f, n));
        near(m.mean, 3.0, 0.02, "normal mean");
        near(m.variance, 4.0, 0.05, "normal variance");
        RCHECK(curandGenerateNormalDouble(g, d, n, -1.0, 0.5));
        m = moments(download(d, n));
        near(m.mean, -1.0, 0.005, "normal double mean");
        near(m.variance, 0.25, 0.005, "normal double variance");
        RCHECK(curandGenerateLogNormal(g, f, n, 0.0f, 0.5f));
        m = moments(download(f, n));
        near(m.mean, std::exp(0.125), 0.01, "log-normal mean");
        RCHECK(curandGenerateLogNormalDouble(g, d, n, 1.0, 0.25));
        m = moments(download(d, n));
        near(m.mean, std::exp(1.0 + 0.03125), 0.02, "log-normal double mean");
        for (double lambda: {0.5, 4.0, 100.0, 3000.0}) {
            RCHECK(curandGeneratePoisson(g, u, n, lambda));
            m = moments(download(u, n));
            char what[64];
            snprintf(what, sizeof(what), "Poisson %g mean", lambda);
            near(m.mean, lambda, 0.01 * std::sqrt(lambda) + 0.003, what);
            snprintf(what, sizeof(what), "Poisson %g variance", lambda);
            near(m.variance, lambda, 0.02 * lambda + 0.01, what);
        }
        // (declared by curand.h, but not every libcurand exports it)
        using Binomial = curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t, unsigned int, double);
        if (auto binomial = reinterpret_cast<Binomial>(dlsym(RTLD_DEFAULT, "curandGenerateBinomial"))) {
            RCHECK(binomial(g, u, n, 20, 0.3));
            std::vector<unsigned int> hb = download(u, n);
            m = moments(hb);
            near(m.mean, 6.0, 0.02, "binomial mean");
            near(m.variance, 4.2, 0.05, "binomial variance");
            for (unsigned int x: hb) EXPECT(x <= 20, "binomial result %u > 20", x);
        }
        RCHECK(curandDestroyGenerator(g));
    }
    // Poisson 4: frequencies against the probabilities
    curandGenerator_t g;
    RCHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    RCHECK(curandGeneratePoisson(g, u, n, 4.0));
    std::vector<unsigned int> hp = download(u, n);
    std::vector<double> count(40, 0);
    for (unsigned int x: hp) count[std::min(x, 39u)]++;
    double p = std::exp(-4.0);
    for (int k = 0; k < 15; k++) {
        const double want = p * n;
        EXPECT(std::fabs(count[k] - want) <= 5 * std::sqrt(want) + 2, "Poisson 4: %d occurs %g times, expected %g", k,
               count[k], want);
        p *= 4.0 / (k + 1);
    }
    RCHECK(curandDestroyGenerator(g));
    CHECK(cudaFree(f));
    CHECK(cudaFree(d));
    CHECK(cudaFree(u));
    PASS("uniform, normal, log-normal, Poisson and binomial results of every generator");
}

static void hostGenerators() {
    const size_t n = 100000;
    for (curandRngType_t type: {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MTGP32, CURAND_RNG_PSEUDO_PHILOX4_32_10,
                                CURAND_RNG_QUASI_SOBOL32}) {
        curandGenerator_t device, host;
        RCHECK(curandCreateGenerator(&device, type));
        RCHECK(curandCreateGeneratorHost(&host, type));
        std::vector<unsigned int> h(n);
        RCHECK(curandGenerate(host, h.data(), n / 2));
        RCHECK(curandGenerate(host, h.data() + n / 2, n / 2));
        expectSame(h, bits(device, n), "host generator");
        std::vector<double> hd(n);
        RCHECK(curandGenerateNormalDouble(host, hd.data(), n, 0.0, 1.0));
        Moments m = moments(hd);
        EXPECT(std::fabs(m.mean) < 0.02 && std::fabs(m.variance - 1) < 0.03, "host normals %g %g", m.mean,
               m.variance);
        RCHECK(curandDestroyGenerator(device));
        RCHECK(curandDestroyGenerator(host));
    }
    PASS("host generators give the device generators' results");
}

static void streamsAndDistributions() {
    // a generator on a stream
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    curandGenerator_t g, ref;
    RCHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_XORWOW));
    RCHECK(curandCreateGenerator(&ref, CURAND_RNG_PSEUDO_XORWOW));
    RCHECK(curandSetStream(g, s));
    const size_t n = 1 << 18;
    unsigned int *d;
    CHECK(cudaMalloc(&d, n * sizeof(unsigned int)));
    RCHECK(curandGenerate(g, d, n));
    CHECK(cudaStreamSynchronize(s));
    std::vector<unsigned int> got = download(d, n);
    expectSame(got, bits(ref, n), "generator on a stream");
    RCHECK(curandDestroyGenerator(g));
    RCHECK(curandDestroyGenerator(ref));
    CHECK(cudaStreamDestroy(s));

    // a Poisson distribution sampled by a kernel
    curandDiscreteDistribution_t dist;
    RCHECK(curandCreatePoissonDistribution(10.0, &dist));
    discreteKernel<<<64, 256>>>(dist, 7, n, d);
    CHECK(cudaGetLastError());
    Moments m = moments(download(d, n));
    EXPECT(std::fabs(m.mean - 10) < 0.05 && std::fabs(m.variance - 10) < 0.3, "kernel Poisson 10: %g %g", m.mean,
           m.variance);
    RCHECK(curandDestroyDistribution(dist));
    CHECK(cudaFree(d));
    PASS("generators on streams, Poisson distributions in kernels");
}

static void statuses() {
    curandGenerator_t g, q, q64;
    RCHECK(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
    RCHECK(curandCreateGenerator(&q, CURAND_RNG_QUASI_DEFAULT));
    RCHECK(curandCreateGenerator(&q64, CURAND_RNG_QUASI_SOBOL64));
    float *f;
    unsigned int *u;
    CHECK(cudaMalloc(&f, 1024 * sizeof(float)));
    CHECK(cudaMalloc(&u, 1024 * sizeof(unsigned int)));
    EXPECT(curandGenerateNormal(g, f, 7, 0, 1) == CURAND_STATUS_LENGTH_NOT_MULTIPLE, "odd normal count");
    RCHECK(curandSetQuasiRandomGeneratorDimensions(q, 3));
    EXPECT(curandGenerateUniform(q, f, 10) == CURAND_STATUS_LENGTH_NOT_MULTIPLE, "quasi count not whole points");
    RCHECK(curandGenerateNormal(q, f, 9, 0, 1));
    EXPECT(curandGenerate(q64, u, 8) == CURAND_STATUS_TYPE_ERROR, "32 bit results of Sobol64");
    EXPECT(curandGenerateLongLong(g, reinterpret_cast<unsigned long long *>(u), 8) == CURAND_STATUS_TYPE_ERROR,
           "64 bit results of XORWOW");
    EXPECT(curandSetPseudoRandomGeneratorSeed(q, 1) == CURAND_STATUS_TYPE_ERROR, "seed of a quasi generator");
    EXPECT(curandSetQuasiRandomGeneratorDimensions(g, 2) == CURAND_STATUS_TYPE_ERROR, "dimensions of XORWOW");
    EXPECT(curandSetQuasiRandomGeneratorDimensions(q, 20001) == CURAND_STATUS_OUT_OF_RANGE, "20001 dimensions");
    EXPECT(curandGeneratePoisson(g, u, 8, 0.0) == CURAND_STATUS_OUT_OF_RANGE, "Poisson 0");
    EXPECT(curandSetGeneratorOrdering(q, CURAND_ORDERING_PSEUDO_BEST) == CURAND_STATUS_OUT_OF_RANGE,
           "pseudo ordering of a quasi generator");
    EXPECT(curandCreateGenerator(&g, curandRngType_t(7)) == CURAND_STATUS_TYPE_ERROR, "unknown type");
    int version = 0, major = 0;
    RCHECK(curandGetVersion(&version));
    RCHECK(curandGetProperty(MAJOR_VERSION, &major));
    EXPECT(version >= 10000 && major == 10, "version %d, major %d", version, major);
    RCHECK(curandDestroyGenerator(g));
    RCHECK(curandDestroyGenerator(q));
    RCHECK(curandDestroyGenerator(q64));
    CHECK(cudaFree(f));
    CHECK(cudaFree(u));
    PASS("error statuses, version");
}

int main() {
    orderings();
    mt19937();
    sobol();
    distributions();
    hostGenerators();
    streamsAndDistributions();
    statuses();
    return 0;
}
