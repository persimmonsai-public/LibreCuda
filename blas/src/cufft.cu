/**
 * LibreCUDA's cuFFT (libcufft.so.12): batched 1D / 2D / 3D complex-to-complex, real-to-complex and complex-to-real
 * transforms of half, float and double data, with cuFFT's advanced data layouts (embedding, stride, distance).
 *
 * Each dimension is transformed with self-sorting Stockham passes (mixed radix: 2 to 8 and generic small primes up to
 * 61; lengths with larger prime factors use Bluestein's algorithm with a power of two transform):
 *  - lines of up to 4096 elements (2048 in double precision) in shared memory, a thread block per group of lines,
 *    reading the caller's input / writing its output directly for the first / last dimension;
 *  - longer contiguous lines as Stockham passes of large radices (products of the factors, each transformed in
 *    shared memory), through two work buffers;
 *  - others (long strided lines) with a kernel per radix pass.
 * Real-to-complex transforms with an even last dimension 2m transform the pairs of values as complex lines of length
 * m and split the result into the half spectrum (complex-to-real ones the reverse); the other dimensions transform
 * the half spectrum. Other real transforms run as complex ones on the completed data. Data the transforms can't read or
 * write directly is gathered into / scattered from a work buffer. Half data is computed in float. Twiddle factors
 * come from tables computed in double precision when the plan is made.
 */
#include <cufft.h>
#include <cufftXt.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr int MAX_RADIX = 61;

enum class Kind {
    C2C, R2C, C2R
};

/**
 * A 1D transform of length n: in shared memory by one thread block per group of lines (DIRECT, n up to SHARED_LIMIT),
 * by Stockham passes of large radices (groups of the factors), each a transform in shared memory (LARGE, contiguous
 * lines), or by a kernel per radix pass through global memory (PASSES)
 */
enum class Mode {
    DIRECT, LARGE, PASSES
};

struct Transform {
    long long n;
    std::vector<int> radices;
    void *twiddles;  // e^{-2 pi i k / n} (complex of the compute precision)
    Mode mode;
    // LARGE: the groups and their roots of unity; e^{-2 pi i k / n} as low[k % 1024] * high[k / 1024]
    std::vector<std::vector<int>> groups;
    std::vector<void *> groupTwiddles;
    void *low, *high;
};

struct Axis {
    Transform t;
    // Bluestein: a transform of length m (power of two); device tables chirp e^{-i pi k^2 / n}, FFT_m of the
    // conjugate chirp
    bool bluestein;
    Transform mt;
    void *chirp, *chirpSpectrum;
};

struct Plan {
    bool made;
    int rank;
    long long n[3];
    long long inEmbed[3], outEmbed[3];
    long long inStride, inDist, outStride, outDist, batch;
    cudaDataType inType, outType;
    Kind kind;
    bool doubles;
    Axis axes[3];
    // real transforms with an even last axis of length 2m: as complex transforms of length m in shared memory (its
    // radices); the other axes then transform the half spectrum
    bool realHalf;
    std::vector<int> halfRadices;
    cudaStream_t stream;
    bool autoAllocate;
    void *workArea, *ownWork;
    size_t workSize;
};

std::mutex g_mutex;
std::map<cufftHandle, Plan *> g_plans;
cufftHandle g_next = 1;

Plan *planOf(cufftHandle handle) {
    std::lock_guard lock(g_mutex);
    auto it = g_plans.find(handle);
    return it != g_plans.end() ? it->second : nullptr;
}

template<typename R>
struct Cx {
    R x, y;
};

template<typename R>
__device__ inline Cx<R> cmul(Cx<R> a, Cx<R> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template<typename R>
__device__ inline Cx<R> conj(Cx<R> a) {
    return {a.x, -a.y};
}

#define GRID_LOOP(i, total) \
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < (total); \
         i += static_cast<long long>(gridDim.x) * blockDim.x)

// blocks of 256 threads for grid-stride loops over count elements (a few thousand blocks: the per-thread setup of the
// kernels is amortized over several elements)
unsigned int gridFor(long long count) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + 255) / 256, 4096)));
}

// ---- passes

// the lines of a transform: line L starts at (L / inner) * outer + L % inner; its elements are step apart
struct Lines {
    long long count, inner, outer, step;
};

__host__ __device__ inline long long lineBase(const Lines &l, long long line) {
    return (line / l.inner) * l.outer + line % l.inner;
}

/**
 * One Stockham pass of radix r over lines of length n: ns is the product of the radices of the passes before.
 * sign -1: forward (e^{-2 pi i ...}), +1: inverse. twiddles: e^{-2 pi i k / n}.
 */
template<typename R>
__global__ void stockhamPass(Lines lines, long long n, int r, long long ns, int sign, const Cx<R> *twiddles,
                             const Cx<R> *in, Cx<R> *out) {
    const long long per_line = n / r;
    GRID_LOOP(e, lines.count * per_line) {
        const long long line = e / per_line, j = e % per_line;
        const long long base = lineBase(lines, line);
        Cx<R> v[MAX_RADIX];
        const long long jn = j % ns;
        const long long twiddle_step = n / (ns * r);
        for (int q = 0; q < r; q++) {
            Cx<R> x = in[base + (j + q * per_line) * lines.step];
            if (ns > 1 && q > 0) {
                Cx<R> t = twiddles[(q * jn * twiddle_step) % n];
                if (sign > 0) t = conj(t);
                x = cmul(x, t);
            }
            v[q] = x;
        }
        const long long dst = (j / ns) * ns * r + jn;
        const long long unit = n / r;
        for (int q = 0; q < r; q++) {
            Cx<R> sum{0, 0};
            for (int s = 0; s < r; s++) {
                Cx<R> t = twiddles[((q * s) % r) * unit];
                if (sign > 0) t = conj(t);
                sum = s == 0 ? Cx<R>{sum.x + v[0].x, sum.y + v[0].y}
                             : Cx<R>{sum.x + cmul(v[s], t).x, sum.y + cmul(v[s], t).y};
            }
            out[base + (dst + q * ns) * lines.step] = sum;
        }
    }
}

// runs the passes of a transform on the lines; returns the buffer holding the result (a or b)
template<typename R>
Cx<R> *passLines(cudaStream_t stream, const Lines &lines, long long n, const std::vector<int> &radices, int sign,
                 const void *twiddles, Cx<R> *a, Cx<R> *b) {
    long long ns = 1;
    Cx<R> *in = a, *out = b;
    for (int r: radices) {
        stockhamPass<R><<<gridFor(lines.count * (n / r)), 256, 0, stream>>>(lines, n, r, ns, sign,
                                                                            static_cast<const Cx<R> *>(twiddles), in,
                                                                            out);
        ns *= r;
        std::swap(in, out);
    }
    return in;
}

// ---- gather and scatter

struct Layout {
    long long embed[3];
    long long stride, dist;
};

struct Shape {
    int rank;
    long long n[3];
    long long total;  // elements per transform
};

__device__ inline long long layoutOffset(const Layout &l, int rank, long long batch, const long long *index) {
    long long offset = 0, scale = 1;
    for (int d = rank - 1; d >= 0; d--) {
        offset += index[d] * scale;
        scale *= l.embed[d];
    }
    return batch * l.dist + offset * l.stride;
}

template<typename R>
__device__ inline Cx<R> loadComplex(const void *p, long long i, cudaDataType type) {
    switch (type) {
        case CUDA_C_16F: {
            const __half2 v = static_cast<const __half2 *>(p)[i];
            return {R(__low2float(v)), R(__high2float(v))};
        }
        case CUDA_C_32F: {
            const float2 v = static_cast<const float2 *>(p)[i];
            return {R(v.x), R(v.y)};
        }
        case CUDA_C_64F: {
            const double2 v = static_cast<const double2 *>(p)[i];
            return {R(v.x), R(v.y)};
        }
        case CUDA_R_16F: return {R(__half2float(static_cast<const __half *>(p)[i])), 0};
        case CUDA_R_32F: return {R(static_cast<const float *>(p)[i]), 0};
        default: return {R(static_cast<const double *>(p)[i]), 0};
    }
}

template<typename R>
__device__ inline void storeValue(void *p, long long i, cudaDataType type, Cx<R> v) {
    switch (type) {
        case CUDA_C_16F: static_cast<__half2 *>(p)[i] = __floats2half2_rn(float(v.x), float(v.y)); break;
        case CUDA_C_32F: static_cast<float2 *>(p)[i] = make_float2(float(v.x), float(v.y)); break;
        case CUDA_C_64F: static_cast<double2 *>(p)[i] = make_double2(double(v.x), double(v.y)); break;
        case CUDA_R_16F: static_cast<__half *>(p)[i] = __float2half(float(v.x)); break;
        case CUDA_R_32F: static_cast<float *>(p)[i] = float(v.x); break;
        default: static_cast<double *>(p)[i] = double(v.x); break;
    }
}

__device__ inline void unravel(const Shape &s, long long e, long long *index) {
    for (int d = s.rank - 1; d >= 0; d--) {
        index[d] = e % s.n[d];
        e /= s.n[d];
    }
}

// work[b][full index] from the input; complex-to-real inputs hold the first n/2 + 1 entries of the last dimension
template<typename R>
__global__ void gather(Shape s, long long batch, Kind kind, Layout in, cudaDataType type, const void *data,
                       Cx<R> *work) {
    GRID_LOOP(e, batch * s.total) {
        const long long b = e / s.total;
        long long index[3];
        unravel(s, e % s.total, index);
        bool mirror = false;
        if (kind == Kind::C2R && index[s.rank - 1] > s.n[s.rank - 1] / 2) {
            // X[k] = conj(X[-k mod n]) in every dimension
            for (int d = 0; d < s.rank; d++) index[d] = (s.n[d] - index[d]) % s.n[d];
            mirror = true;
        }
        Cx<R> v = loadComplex<R>(data, layoutOffset(in, s.rank, b, index), type);
        work[e] = mirror ? conj(v) : v;
    }
}

template<typename R>
__global__ void scatter(Shape s, long long batch, Kind kind, Layout out, cudaDataType type, const Cx<R> *work,
                        void *data) {
    GRID_LOOP(e, batch * s.total) {
        const long long b = e / s.total;
        long long index[3];
        unravel(s, e % s.total, index);
        if (kind == Kind::R2C && index[s.rank - 1] > s.n[s.rank - 1] / 2) continue;
        storeValue<R>(data, layoutOffset(out, s.rank, b, index), type, work[e]);
    }
}

// ---- transforms in shared memory

// shared memory of a block for a buffer of its lines (32 KiB): the longest transform in one block
constexpr size_t SHARED_BYTES = size_t(32) << 10;
constexpr int MAX_PASSES = 24;
// passes of a transform in shared memory (its factors grouped into radices up to 25: at most 4 for the lengths that fit)
constexpr int MAX_SHARED_PASSES = 8;
/**
 * Elements per thread and per block of the transforms in shared memory, and lines per block where the lines' elements
 * are strided (so that loads and stores across the lines take whole sectors). Small blocks with few elements per
 * thread hide the latency of global memory best: several blocks per SM, one loading while another computes (with 16
 * elements per thread and 4096 per block, the transforms took 1.5 times as long).
 */
constexpr int PER_THREAD = 8;
constexpr long long BLOCK_ELEMENTS = 1024;
constexpr long long STRIDED_LINES = 16;
// blocks of 256 threads per SM the kernels' registers allow (doubles: half as many)
constexpr int BLOCKS_PER_SM = 4;

template<typename R>
constexpr long long SHARED_LIMIT = SHARED_BYTES / sizeof(Cx<R>);
// large radix passes: blocks of the same size
template<typename R>
constexpr long long LARGE_LIMIT = SHARED_LIMIT<R>;

/**
 * Lines in memory (a batch and up to two other dimensions): line L starts at (L / c1) * s1 + ((L % c1) / c2) * s2 +
 * (L % c2) * s3, its elements are elemStride apart
 */
struct LineSet {
    long long count, c1, s1, c2, s2, s3, elemStride;
};

__host__ __device__ inline long long setBase(const LineSet &s, long long line) {
    if ((unsigned long long) (line | s.c1 | s.c2) <= 0xffffffffull) {
        // (32 bit divisions: the 64 bit ones are calls of a hundred instructions, which blocks waited for)
        const unsigned l = unsigned(line), c1 = unsigned(s.c1), c2 = unsigned(s.c2);
        return (l / c1) * s.s1 + ((l % c1) / c2) * s.s2 + (l % c2) * s.s3;
    }
    const long long within = line % s.c1;
    return (line / s.c1) * s.s1 + (within / s.c2) * s.s2 + (line % s.c2) * s.s3;
}

// lines of a buffer: the caller's (any data type and layout) or a work buffer (complex of the compute precision)
struct Endpoint {
    void *data;
    cudaDataType type;
    LineSet lines;
};

/**
 * What a shared memory transform reads and writes: complex lines (COMPLEX); the last axis of a real-to-complex
 * transform of length 2n as a complex transform of length n of the pairs, split into the n + 1 outputs
 * (REAL_FORWARD); complex-to-real, the reverse (REAL_INVERSE)
 */
enum class Io {
    COMPLEX, REAL_FORWARD, REAL_INVERSE
};

/**
 * Division of indices within a block by a divisor known at run time, as a multiply: m = ceil(2^32 / d) gives
 * floor(x / d) exactly while x * d < 2^32, which holds for the elements of a block (at most SHARED_LIMIT) and their
 * divisors. A division takes about 20 instructions; the kernels divided every element's and butterfly's index several
 * times, and spent most of their time doing it.
 */
struct FastDiv {
    unsigned d, m;  // m: 0 for d = 1
};

__host__ __device__ inline FastDiv fastDiv(unsigned d) {
    // (ceil(2^32 / d) as floor((2^32 - 1) / d) + 1: a 32 bit division, not a call to the 64 bit one)
    return {d, d <= 1 ? 0u : 0xffffffffu / d + 1};
}

__device__ inline unsigned divOf(unsigned x, FastDiv f) {
    return f.m == 0 ? x : __umulhi(x, f.m);
}

__device__ inline unsigned modOf(unsigned x, FastDiv f) {
    return x - divOf(x, f) * f.d;
}

/**
 * A pass of transforms of length n in shared memory: its radix, ns (the product of the earlier radices), the table
 * entries between butterflies' twiddles (step) and between the radix's roots of unity (unit), divisions by n / radix
 * and ns. Computed on the host: every thread worked them out in every pass, three divisions, as much as its butterflies.
 */

struct Pass {
    int radix, ns, step, unit;
    FastDiv per, nsDiv;
};

struct Passes {
    int count;
    Pass pass[MAX_SHARED_PASSES];
};

struct SharedFft {
    int n, lines;  // transform length, lines per block
    FastDiv nDiv, outputsDiv, linesDiv;  // divisions by n, n + 1, lines
    int sign;
    Passes passes;
};

template<typename R>
__device__ inline Cx<R> twiddle(const Cx<R> *table, long long index, int sign) {
    const Cx<R> t = table[index];
    return sign > 0 ? conj(t) : t;
}

template<typename R>
__device__ inline Cx<R> cadd(Cx<R> a, Cx<R> b) {
    return {a.x + b.x, a.y + b.y};
}

template<typename R>
__device__ inline Cx<R> csub(Cx<R> a, Cx<R> b) {
    return {a.x - b.x, a.y - b.y};
}

// a * e^{sign i pi / 2}
template<typename R>
__device__ inline Cx<R> quarter(Cx<R> a, int sign) {
    return sign < 0 ? Cx<R>{a.y, -a.x} : Cx<R>{-a.y, a.x};
}

// cos and sin of 2 pi j / RADIX, computed at compile time (Taylor series of the angle reduced to [-pi, pi])
template<int RADIX>
struct Roots {
    double cos[RADIX], sin[RADIX];
};

template<int RADIX>
__host__ __device__ constexpr Roots<RADIX> roots() {
    Roots<RADIX> r{};
    for (int j = 0; j < RADIX; j++) {
        const double x = 2 * 3.14159265358979323846264338327950288 * (j <= RADIX / 2 ? j : j - RADIX) / RADIX;
        double sinTerm = x, sinSum = x, cosTerm = 1, cosSum = 1;
        for (int i = 1; i < 30; i++) {
            sinTerm *= -x * x / ((2 * i) * (2 * i + 1));
            sinSum += sinTerm;
            cosTerm *= -x * x / ((2 * i - 1) * (2 * i));
            cosSum += cosTerm;
        }
        r.cos[j] = cosSum;
        r.sin[j] = sinSum;
    }
    return r;
}

/**
 * Composite radices (other than 4 and 8) as A x B with A the largest divisor up to the square root (0 for primes):
 * their DFTs run as DFTs of A and of B in registers, with constant twiddles. A pass of radix 10 or 16 does the work of
 * two, with one trip through shared memory instead of two.
 */
__host__ __device__ constexpr int splitOf(int n) {
    if (n == 4 || n == 8) return 0;
    int a = 0;
    for (int d = 2; d * d <= n; d++) {
        if (n % d == 0) a = d;
    }
    return a;
}

// in-place DFT of RADIX values; table[k * unit]: e^{-2 pi i k / RADIX}
template<typename R, int RADIX>
__device__ inline void dft(Cx<R> *v, int sign, const Cx<R> *table, int unit) {
    if constexpr (RADIX == 2) {
        const Cx<R> a = v[0], b = v[1];
        v[0] = cadd(a, b);
        v[1] = csub(a, b);
    } else if constexpr (RADIX == 4) {
        const Cx<R> t0 = cadd(v[0], v[2]), t1 = csub(v[0], v[2]), t2 = cadd(v[1], v[3]);
        const Cx<R> t3 = quarter(csub(v[1], v[3]), sign);
        v[0] = cadd(t0, t2);
        v[2] = csub(t0, t2);
        v[1] = cadd(t1, t3);
        v[3] = csub(t1, t3);
    } else if constexpr (RADIX == 8) {
        Cx<R> e[4] = {v[0], v[2], v[4], v[6]}, o[4] = {v[1], v[3], v[5], v[7]};
        dft<R, 4>(e, sign, table, unit);
        dft<R, 4>(o, sign, table, unit);
        const R h = R(0.70710678118654752440);
        // o[k] * e^{sign 2 pi i k / 8}
        const Cx<R> w1 = sign < 0 ? Cx<R>{h * (o[1].x + o[1].y), h * (o[1].y - o[1].x)}
                                  : Cx<R>{h * (o[1].x - o[1].y), h * (o[1].y + o[1].x)};
        const Cx<R> w2 = quarter(o[2], sign);
        const Cx<R> q3 = quarter(o[3], sign);
        const Cx<R> w3 = sign < 0 ? Cx<R>{h * (q3.x + q3.y), h * (q3.y - q3.x)}
                                  : Cx<R>{h * (q3.x - q3.y), h * (q3.y + q3.x)};
        v[0] = cadd(e[0], o[0]);
        v[4] = csub(e[0], o[0]);
        v[1] = cadd(e[1], w1);
        v[5] = csub(e[1], w1);
        v[2] = cadd(e[2], w2);
        v[6] = csub(e[2], w2);
        v[3] = cadd(e[3], w3);
        v[7] = csub(e[3], w3);
    } else if constexpr (splitOf(RADIX) != 0) {
        // x[B n1 + n2] -> DFT_A over n1 for each n2 -> times w^{n2 k1} -> DFT_B over n2 for each k1 -> X[k1 + A k2]
        constexpr int A = splitOf(RADIX), B = RADIX / A;
        constexpr Roots<RADIX> w = roots<RADIX>();
        Cx<R> y[RADIX];
#pragma unroll
        for (int n2 = 0; n2 < B; n2++) {
            Cx<R> t[A];
#pragma unroll
            for (int n1 = 0; n1 < A; n1++) t[n1] = v[B * n1 + n2];
            dft<R, A>(t, sign, table, unit);
#pragma unroll
            for (int k1 = 0; k1 < A; k1++) {
                const int m = (n2 * k1) % RADIX;
                if (m == 0) {
                    y[k1 * B + n2] = t[k1];
                } else {
                    // e^{sign 2 pi i m / RADIX}
                    const R c = R(w.cos[m]), sn = R(sign) * R(w.sin[m]);
                    y[k1 * B + n2] = {t[k1].x * c - t[k1].y * sn, t[k1].x * sn + t[k1].y * c};
                }
            }
        }
#pragma unroll
        for (int k1 = 0; k1 < A; k1++) {
            dft<R, B>(y + k1 * B, sign, table, unit);
#pragma unroll
            for (int k2 = 0; k2 < B; k2++) v[k1 + A * k2] = y[k1 * B + k2];
        }
    } else {
        // odd primes: with a_m = v_m + v_{R-m}, b_m = v_m - v_{R-m} (m <= H = (R - 1) / 2),
        // y_k = v_0 + sum_m a_m cos(2 pi m k / R) + i sign sum_m b_m sin(2 pi m k / R), y_{R-k} its other half
        // (real constants: a quarter of the multiplications of the plain sum and no table loads)
        constexpr int H = (RADIX - 1) / 2;
        constexpr Roots<RADIX> w = roots<RADIX>();
        Cx<R> a[H + 1], b[H + 1];
#pragma unroll
        for (int m = 1; m <= H; m++) {
            a[m] = cadd(v[m], v[RADIX - m]);
            b[m] = csub(v[m], v[RADIX - m]);
        }
        Cx<R> y0 = v[0];
#pragma unroll
        for (int m = 1; m <= H; m++) y0 = cadd(y0, a[m]);
#pragma unroll
        for (int k = 1; k <= H; k++) {
            Cx<R> re = v[0], im = {0, 0};
#pragma unroll
            for (int m = 1; m <= H; m++) {
                const R c = R(w.cos[(m * k) % RADIX]), sn = R(w.sin[(m * k) % RADIX]);
                re = {re.x + c * a[m].x, re.y + c * a[m].y};
                im = {im.x + sn * b[m].x, im.y + sn * b[m].y};
            }
            // i sign im
            const Cx<R> t = sign < 0 ? Cx<R>{im.y, -im.x} : Cx<R>{-im.y, im.x};
            v[k] = cadd(re, t);
            v[RADIX - k] = csub(re, t);
        }
        v[0] = y0;
    }
}

/**
 * Lines in shared memory: element k of line l at l * ls + k * es (lines one after another: ls n, es 1; interleaved:
 * ls 1, es count). Threads take butterflies along the lines' elements or across the lines, whichever is contiguous,
 * so shared memory accesses are free of bank conflicts.
 */
struct SharedLines {
    int count, n, ls, es;
    FastDiv countDiv;
};

/**
 * Element i of the lines in shared memory is at padded(i): a gap after every 16 complex values. A Stockham pass writes
 * each butterfly's outputs ns apart, a power of two apart in the first passes of power of two lengths, which put the
 * stores of a warp on a few banks (without the gaps, transforms of length 128 took 1.5 times as long).
 */
__host__ __device__ inline int padded(int i) {
    return i + (i >> 4);
}

// complex values of shared memory for count elements (padded)
__host__ __device__ inline long long paddedCount(long long count) {
    return count + count / 16 + 1;
}

__device__ inline void butterflyOf(const SharedLines &s, FastDiv per, int t, int &l, int &j) {
    if (s.ls == 1) {
        l = int(modOf(t, s.countDiv));
        j = int(divOf(t, s.countDiv));
    } else {
        l = int(divOf(t, per));
        j = int(modOf(t, per));
    }
}

/**
 * A Stockham pass of radix RADIX over the lines of a block (ns: product of the earlier radices): each thread takes
 * butterflies in turn, loads their values (load(line, index)), transforms them and stores them (store(line, index,
 * value)), out of place, from shared memory to shared memory or, the first and last passes, from and to the lines in
 * global memory. A thread holds one butterfly's values at a time: the kernels needed few registers and little code
 * (holding all of a thread's butterflies across a barrier for an in place pass took 255 registers with spills and
 * 300 KB of code).
 */
template<typename R, int RADIX, typename Load, typename Store>
__device__ inline void butterflies(const SharedLines &s, const Pass &d, int sign, const Cx<R> *table, Load load,
                                   Store store) {
    const int per = s.n / RADIX, ns = d.ns;
    const int count = s.count * per;
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        int l, j;
        butterflyOf(s, d.per, t, l, j);
        const int jn = int(modOf(j, d.nsDiv));
        const int to = (j - jn) * RADIX + jn;
        Cx<R> v[RADIX];
#pragma unroll
        for (int q = 0; q < RADIX; q++) v[q] = load(l, j + q * per);
        if (ns > 1) {
#pragma unroll
            for (int q = 1; q < RADIX; q++) v[q] = cmul(v[q], twiddle(table, q * jn * d.step, sign));
        }
        dft<R, RADIX>(v, sign, table, d.unit);
#pragma unroll
        for (int q = 0; q < RADIX; q++) store(l, to + q * ns, v[q]);
    }
}

// a pass of a radix with a specialized butterfly; false for other radices
template<typename R, typename Load, typename Store>
__device__ inline bool specializedPass(const SharedLines &s, const Pass &d, int sign, const Cx<R> *table, Load load,
                                       Store store) {
    switch (d.radix) {
        case 2: butterflies<R, 2>(s, d, sign, table, load, store); return true;
        case 3: butterflies<R, 3>(s, d, sign, table, load, store); return true;
        case 4: butterflies<R, 4>(s, d, sign, table, load, store); return true;
        case 5: butterflies<R, 5>(s, d, sign, table, load, store); return true;
        case 6: butterflies<R, 6>(s, d, sign, table, load, store); return true;
        case 7: butterflies<R, 7>(s, d, sign, table, load, store); return true;
        case 8: butterflies<R, 8>(s, d, sign, table, load, store); return true;
        case 9: butterflies<R, 9>(s, d, sign, table, load, store); return true;
        case 10: butterflies<R, 10>(s, d, sign, table, load, store); return true;
        case 11: butterflies<R, 11>(s, d, sign, table, load, store); return true;
        case 12: butterflies<R, 12>(s, d, sign, table, load, store); return true;
        case 13: butterflies<R, 13>(s, d, sign, table, load, store); return true;
        case 14: butterflies<R, 14>(s, d, sign, table, load, store); return true;
        case 15: butterflies<R, 15>(s, d, sign, table, load, store); return true;
        case 16: butterflies<R, 16>(s, d, sign, table, load, store); return true;
        case 20: butterflies<R, 20>(s, d, sign, table, load, store); return true;
        case 21: butterflies<R, 21>(s, d, sign, table, load, store); return true;
        case 25: butterflies<R, 25>(s, d, sign, table, load, store); return true;
        default: return false;
    }
}

// the same for other radices (primes up to MAX_RADIX) between buffers in shared memory, the values in local memory
template<typename R>
__device__ __noinline__ void sharedPassGeneric(const Cx<R> *src, Cx<R> *dst, SharedLines s, const Pass &d, int sign,
                                               const Cx<R> *table) {
    const int r = d.radix, per = s.n / r, ns = d.ns;
    Cx<R> v[MAX_RADIX];
    for (int t = threadIdx.x; t < s.count * per; t += blockDim.x) {
        int l, j;
        butterflyOf(s, d.per, t, l, j);
        const int jn = int(modOf(j, d.nsDiv));
        const int from = l * s.ls, to = l * s.ls + ((j - jn) * r + jn) * s.es;
        for (int q = 0; q < r; q++) {
            v[q] = src[padded(from + (j + q * per) * s.es)];
            if (q > 0 && ns > 1) v[q] = cmul(v[q], twiddle(table, q * jn * d.step, sign));
        }
        for (int k = 0; k < r; k++) {
            Cx<R> acc = v[0];
            for (int q = 1; q < r; q++) acc = cadd(acc, cmul(v[q], twiddle(table, ((q * k) % r) * d.unit, sign)));
            dst[padded(to + k * ns * s.es)] = acc;
        }
    }
}

// loads and stores of the lines of a block in shared memory
template<typename R>
struct SharedLoad {
    const Cx<R> *a;
    SharedLines s;

    __device__ Cx<R> operator()(int l, int k) const {
        return a[padded(l * s.ls + k * s.es)];
    }
};

template<typename R>
struct SharedStore {
    Cx<R> *a;
    SharedLines s;

    __device__ void operator()(int l, int k, Cx<R> v) const {
        a[padded(l * s.ls + k * s.es)] = v;
    }
};

/**
 * The passes of transforms in shared memory, alternating between a and b (PLAIN: only radices with a specialized
 * pass); returns the buffer holding the result. The passes come from shared memory (a copy of the kernel's): indexed
 * by the pass, the kernel parameter went through local memory.
 */
template<typename R, bool PLAIN = false>
__device__ Cx<R> *sharedPasses(Cx<R> *a, Cx<R> *b, SharedLines s, int count, const Pass *passes, int sign,
                               const Cx<R> *table) {
    for (int p = 0; p < count; p++) {
        const Pass &d = passes[p];
        if (!specializedPass<R>(s, d, sign, table, SharedLoad<R>{a, s}, SharedStore<R>{b, s})) {
            if constexpr (!PLAIN) sharedPassGeneric<R>(a, b, s, d, sign, table);
        }
        __syncthreads();
        Cx<R> *t = a;
        a = b;
        b = t;
    }
    return a;
}

// copies the passes to shared memory (with constant indices into the kernel parameter); the caller synchronizes
__device__ inline void copyPasses(const Passes &from, Pass *to) {
    if (threadIdx.x == 0) {
#pragma unroll
        for (int p = 0; p < MAX_SHARED_PASSES; p++) {
            if (p < from.count) to[p] = from.pass[p];
        }
    }
}

// the passes of transforms of length n with the radices (the table: roots of unity of length n * stride)
Passes passesFor(int n, const int *radices, int count, int stride) {
    Passes passes{};
    passes.count = count;
    int ns = 1;
    for (int p = 0; p < count; p++) {
        const int r = radices[p];
        passes.pass[p] = {r, ns, n / (ns * r) * stride, n / r * stride, fastDiv(unsigned(n / r)), fastDiv(unsigned(ns))};
        ns *= r;
    }
    return passes;
}

/**
 * Buffers of shared memory of a block of transforms (each of paddedCount(lines * n) values): the general kernel's
 * passes alternate between two; the lean kernel's first and last passes read and write global memory
 */
__host__ __device__ inline int sharedBuffers(bool plain, Io io, int passes) {
    if (!plain) return 2;
    if (passes == 0) return 1;
    const int staged = io == Io::REAL_FORWARD || passes == 1 ? passes : passes - 1;
    return staged < 2 ? staged : 2;
}

// element e of the lines of a block as (line, index): along the lines when their elements are contiguous, else across
__device__ inline void elementOf(const LineSet &set, int e, FastDiv count, FastDiv length, int &l, int &k) {
    if (set.elemStride == 1) {
        l = int(divOf(e, length));
        k = int(modOf(e, length));
    } else {
        l = int(modOf(e, count));
        k = int(divOf(e, count));
    }
}

/**
 * Transforms f.lines lines per block in shared memory: loads them (coalesced along or across the lines), runs the
 * passes, stores them; IO selects what is loaded and stored (see Io).
 *
 * PLAIN (the lean kernel): the endpoints hold complex values of R (real values of R for a real transform's real side)
 * and the radices all have specialized passes, so loads and stores need no switch on the data type, and the first and
 * last passes load from and store to the endpoints directly.
 */
template<typename R, Io IO, bool PLAIN>
__global__ __launch_bounds__(256, sizeof(R) == 8 ? BLOCKS_PER_SM / 2 : BLOCKS_PER_SM)
void sharedFft(SharedFft f, Endpoint in, Endpoint out, const Cx<R> *table) {
    extern __shared__ __align__(16) unsigned char shared[];
    const auto loadC = [&](long long i) -> Cx<R> {
        if constexpr (PLAIN) {
            return static_cast<const Cx<R> *>(in.data)[i];
        } else {
            return loadComplex<R>(in.data, i, in.type);
        }
    };
    const auto loadR = [&](long long i) -> R {
        if constexpr (PLAIN) {
            return static_cast<const R *>(in.data)[i];
        } else {
            return loadComplex<R>(in.data, i, in.type).x;
        }
    };
    const auto storeC = [&](long long i, Cx<R> v) {
        if constexpr (PLAIN) {
            static_cast<Cx<R> *>(out.data)[i] = v;
        } else {
            storeValue<R>(out.data, i, out.type, v);
        }
    };
    const auto storeR = [&](long long i, R v) {
        if constexpr (PLAIN) {
            static_cast<R *>(out.data)[i] = v;
        } else {
            storeValue<R>(out.data, i, out.type, Cx<R>{v, 0});
        }
    };
    const int n = f.n;
    Cx<R> *a = reinterpret_cast<Cx<R> *>(shared);
    Cx<R> *b = a + paddedCount(f.lines * n);
    long long *inBase = reinterpret_cast<long long *>(a + sharedBuffers(PLAIN, IO, f.passes.count) * paddedCount(f.lines * n));
    long long *outBase = inBase + f.lines;
    Pass *passes = reinterpret_cast<Pass *>(outBase + f.lines);
    copyPasses(f.passes, passes);
    const long long first = static_cast<long long>(blockIdx.x) * f.lines;
    const int count = int(min(static_cast<long long>(f.lines), in.lines.count - first));
    for (int l = threadIdx.x; l < count; l += blockDim.x) {
        inBase[l] = setBase(in.lines, first + l);
        outBase[l] = setBase(out.lines, first + l);
    }
    __syncthreads();
    const long long ies = in.lines.elemStride, oes = out.lines.elemStride;
    // lines one after another in shared memory when their elements are contiguous, else interleaved
    const FastDiv countDiv = count == f.lines ? f.linesDiv : fastDiv(count), nDiv = f.nDiv, outputsDiv = f.outputsDiv;
    const SharedLines s = ies == 1 ? SharedLines{count, n, n, 1, countDiv} : SharedLines{count, n, 1, count, countDiv};
    const int total = count * n;
    // element k of line l of the input
    const auto input = [&](int l, int k) -> Cx<R> {
        const long long base = inBase[l];
        if constexpr (IO == Io::COMPLEX) {
            return loadC(base + k * ies);
        } else if constexpr (IO == Io::REAL_FORWARD) {
            // z[k] = x[2k] + i x[2k + 1]
            return {loadR(base + 2 * k * ies), loadR(base + (2 * k + 1) * ies)};
        } else {
            // Z[k] = E[k] + i O[k]: E = X[k] + conj(X[n - k]), O = (X[k] - conj(X[n - k])) e^{2 pi i k / 2n}
            // (the imaginary parts of X[0] and X[n] are ignored)
            Cx<R> x = loadC(base + k * ies);
            Cx<R> y = loadC(base + (n - k) * ies);
            if (k == 0) {
                x.y = 0;
                y.y = 0;
            }
            y = conj(y);
            const Cx<R> e = cadd(x, y), o = cmul(csub(x, y), conj(table[k]));
            return {e.x - o.y, e.y + o.x};
        }
    };
    // the lines into shared memory (a)
    const auto loadLines = [&] {
        for (int e0 = threadIdx.x; e0 < total; e0 += 4 * blockDim.x) {
            // (4 loads in flight per thread)
            Cx<R> v[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int e = e0 + u * blockDim.x;
                if (e >= total) continue;
                int l, k;
                elementOf(in.lines, e, countDiv, nDiv, l, k);
                v[u] = input(l, k);
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int e = e0 + u * blockDim.x;
                if (e >= total) continue;
                int l, k;
                elementOf(in.lines, e, countDiv, nDiv, l, k);
                a[padded(l * s.ls + k * s.es)] = v[u];
            }
        }
        __syncthreads();
    };
    if (PLAIN && f.passes.count > 0) {
        /**
         * The first pass loads its butterflies from the input, the last stores them to the output: two trips through
         * shared memory fewer than loading the lines into it and storing them from it, and their barriers. (The last
         * pass of real-to-complex transforms stores to shared memory: an output takes two of its values; so does a
         * single pass, which may run in place.)
         */
        const int passCount = f.passes.count;
        const bool staged = IO == Io::REAL_FORWARD || passCount == 1;
        const auto output = [&](int l, int k, Cx<R> v) {
            if constexpr (IO == Io::COMPLEX) {
                storeC(outBase[l] + k * oes, v);
            } else if constexpr (IO == Io::REAL_INVERSE) {
                // x[2k] = Re z[k], x[2k + 1] = Im z[k]
                storeR(outBase[l] + 2 * k * oes, v.x);
                storeR(outBase[l] + (2 * k + 1) * oes, v.y);
            }
        };
        Cx<R> *src = a;
        for (int p = 0; p < passCount; p++) {
            const Pass &d = passes[p];
            Cx<R> *dst = p == 0 || src == b ? a : b;
            if (p == 0) {
                specializedPass<R>(s, d, f.sign, table, input, SharedStore<R>{dst, s});
            } else if (p < passCount - 1 || staged) {
                specializedPass<R>(s, d, f.sign, table, SharedLoad<R>{src, s}, SharedStore<R>{dst, s});
            } else if constexpr (IO != Io::REAL_FORWARD) {
                specializedPass<R>(s, d, f.sign, table, SharedLoad<R>{src, s}, output);
                return;
            }
            __syncthreads();
            src = dst;
        }
        a = src;
    } else {
        loadLines();
        a = sharedPasses<R, PLAIN>(a, b, s, f.passes.count, passes, f.sign, table);
    }
    if constexpr (IO == Io::COMPLEX) {
        for (int e = threadIdx.x; e < total; e += blockDim.x) {
            int l, k;
            elementOf(out.lines, e, countDiv, nDiv, l, k);
            storeC(outBase[l] + k * oes, a[padded(l * s.ls + k * s.es)]);
        }
    } else if constexpr (IO == Io::REAL_FORWARD) {
        // X[k] = E[k] + e^{-2 pi i k / 2n} O[k] (k <= n): E = (Z[k] + conj(Z[n - k])) / 2,
        // O = (Z[k] - conj(Z[n - k])) / 2i
        const int outputs = count * (n + 1);
        for (int e = threadIdx.x; e < outputs; e += blockDim.x) {
            int l, k;
            elementOf(out.lines, e, countDiv, outputsDiv, l, k);
            const Cx<R> z = a[padded(l * s.ls + (k == n ? 0 : k) * s.es)];
            const Cx<R> w = conj(a[padded(l * s.ls + (k == 0 ? 0 : n - k) * s.es)]);
            const Cx<R> sum = cadd(z, w), diff = csub(z, w);
            const Cx<R> even = {sum.x / 2, sum.y / 2}, odd = {diff.y / 2, -diff.x / 2};
            storeC(outBase[l] + k * oes, cadd(even, cmul(table[k], odd)));
        }
    } else {
        // x[2k] = Re z[k], x[2k + 1] = Im z[k]
        for (int e = threadIdx.x; e < total; e += blockDim.x) {
            int l, k;
            elementOf(out.lines, e, countDiv, nDiv, l, k);
            const Cx<R> z = a[padded(l * s.ls + k * s.es)];
            storeR(outBase[l] + 2 * k * oes, z.x);
            storeR(outBase[l] + (2 * k + 1) * oes, z.y);
        }
    }
}

// radices with a pass of their own (sharedPass); others take sharedPassGeneric
constexpr int SPECIALIZED[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 25};

bool specialized(int r) {
    return std::find(std::begin(SPECIALIZED), std::end(SPECIALIZED), r) != std::end(SPECIALIZED);
}

/**
 * The radices of transforms in shared memory: the factors 2, 3, 5 and 7 regrouped into as few specialized radices as
 * possible (of these, the smallest largest radix), other factors as they are, largest first
 */
std::vector<int> groupRadices(const std::vector<int> &radices) {
    int count[4] = {0, 0, 0, 0};
    const int primes[4] = {2, 3, 5, 7};
    std::vector<int> grouped;
    for (int r: radices) {
        int rest = r;
        for (int i = 0; i < 4; i++) {
            while (rest % primes[i] == 0) {
                count[i]++;
                rest /= primes[i];
            }
        }
        if (rest != 1) grouped.push_back(r);  // (a prime above 7)
    }
    // best[state]: (passes, largest radix, radix taken first) over the remaining exponents
    struct Best {
        int passes, largest, radix;
    };
    std::map<std::array<int, 4>, Best> best;
    std::function<Best(std::array<int, 4>)> solve = [&](std::array<int, 4> e) -> Best {
        if (e == std::array<int, 4>{0, 0, 0, 0}) return {0, 0, 0};
        if (auto it = best.find(e); it != best.end()) return it->second;
        Best b{INT32_MAX, INT32_MAX, 0};
        for (int r: SPECIALIZED) {
            std::array<int, 4> rest = e;
            int x = r;
            for (int i = 0; i < 4; i++) {
                while (x % primes[i] == 0) {
                    rest[i]--;
                    x /= primes[i];
                }
            }
            if (x != 1 || *std::min_element(rest.begin(), rest.end()) < 0) continue;
            const Best sub = solve(rest);
            const Best with{sub.passes + 1, std::max(sub.largest, r), r};
            if (with.passes < b.passes || (with.passes == b.passes && with.largest < b.largest)) b = with;
        }
        best[e] = b;
        return b;
    };
    for (std::array<int, 4> e{count[0], count[1], count[2], count[3]}; e != std::array<int, 4>{0, 0, 0, 0};) {
        const int r = solve(e).radix;
        grouped.push_back(r);
        for (int i = 0, x = r; i < 4; i++) {
            while (x % primes[i] == 0) {
                e[i]--;
                x /= primes[i];
            }
        }
    }
    std::sort(grouped.begin(), grouped.end(), std::greater<>());
    return grouped;
}

// (remembered: transforms run them on every execution)
std::vector<int> sharedRadices(const std::vector<int> &radices) {
    static std::mutex mutex;
    static std::map<std::vector<int>, std::vector<int>> known;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = known.find(radices);
    if (it == known.end()) it = known.emplace(radices, groupRadices(radices)).first;
    return it->second;
}

// threads for transforms of elements values in shared memory (PER_THREAD values each); 0 if more than a block has
long long threadsFor(long long elements) {
    const long long threads = std::max<long long>(32, ((elements + PER_THREAD - 1) / PER_THREAD + 31) / 32 * 32);
    return threads <= 1024 ? threads : 0;
}

/**
 * Threads of a block of shared memory transforms of elements values (at most threadsFor's): each pass's butterflies
 * spread over them in rounds, so the count leaving the fewest threads idle in the last rounds, the most of those
 * within 5 %
 */
long long blockThreads(long long elements, const Passes &passes) {
    const long long most = threadsFor(elements);
    double best = 0;
    long long chosen = most;
    std::vector<std::pair<long long, double>> options;
    for (long long threads = 32; threads <= most; threads += 32) {
        long long used = (elements + threads - 1) / threads * threads, work = elements;
        for (int p = 0; p < passes.count; p++) {
            const long long butterflies = elements / passes.pass[p].radix;
            used += (butterflies + threads - 1) / threads * threads;
            work += butterflies;
        }
        const double efficiency = double(work) / double(used);
        options.emplace_back(threads, efficiency);
        best = std::max(best, efficiency);
    }
    for (const auto &[threads, efficiency]: options) {
        if (efficiency >= 0.95 * best) chosen = threads;
    }
    return chosen;
}

// lets the kernel use bytes of dynamic shared memory (on the current device; set once, not on every launch)
template<typename R, typename Kernel>
bool allowShared(Kernel kernel, size_t bytes) {
    static std::mutex mutex;
    static std::map<std::pair<const void *, int>, size_t> allowed;  // by kernel and device
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    std::lock_guard<std::mutex> lock(mutex);
    size_t &done = allowed[{reinterpret_cast<const void *>(kernel), device}];
    if (bytes <= done) return true;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    done = bytes;
    return true;
}

/**
 * Lines per block of shared memory transforms of length n: BLOCK_ELEMENTS values (at least STRIDED_LINES lines where
 * they are strided), at most as many as fit and 256, fewer when their threads exceed what the kernel can launch; 0 if
 * none fit
 */
template<typename R, Io IO, bool PLAIN = false>
int linesFor(long long n, const std::vector<int> &radices, bool strided = false) {
    static const int maxThreads = [] {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, sharedFft<R, IO, PLAIN>) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return attributes.maxThreadsPerBlock;
    }();
    if (n > SHARED_LIMIT<R> || int(radices.size()) > MAX_PASSES) return 0;
    long long most = std::max<long long>(1, std::min<long long>(SHARED_LIMIT<R>, BLOCK_ELEMENTS) / n);
    if (strided) most = std::max<long long>(most, std::min<long long>(SHARED_LIMIT<R> / n, STRIDED_LINES));
    for (long long lines = std::min<long long>(most, 256); lines >= 1; lines /= 2) {
        const long long threads = threadsFor(lines * n);
        if (threads != 0 && threads <= maxThreads) return int(lines);
    }
    return 0;
}

// the lean kernel applies (sharedFft's PLAIN)
template<typename R, Io IO>
bool plainShared(const std::vector<int> &radices, const Endpoint &in, const Endpoint &out) {
    const cudaDataType complex = sizeof(R) == 8 ? CUDA_C_64F : CUDA_C_32F, real = sizeof(R) == 8 ? CUDA_R_64F : CUDA_R_32F;
    const cudaDataType inType = IO == Io::REAL_FORWARD ? real : complex, outType = IO == Io::REAL_INVERSE ? real : complex;
    if (in.type != inType || out.type != outType) return false;
    return std::all_of(radices.begin(), radices.end(), specialized);
}

template<typename R, Io IO, bool PLAIN>
bool launchShared(cudaStream_t stream, long long n, const std::vector<int> &radices, int sign, const void *table,
                  int tableStride, const Endpoint &in, const Endpoint &out) {
    SharedFft f{};
    f.n = int(n);
    f.lines = linesFor<R, IO, PLAIN>(n, radices, in.lines.elemStride != 1 || out.lines.elemStride != 1);
    if (f.lines == 0) return false;
    f.nDiv = fastDiv(unsigned(n));
    f.outputsDiv = fastDiv(unsigned(n + 1));
    f.linesDiv = fastDiv(unsigned(f.lines));
    f.sign = sign;
    f.passes = passesFor(int(n), radices.data(), int(radices.size()), tableStride);
    const long long threads = blockThreads(f.lines * n, f.passes);
    const size_t bytes = sharedBuffers(PLAIN, IO, f.passes.count) * size_t(paddedCount(f.lines * n)) * sizeof(Cx<R>) +
                         2 * size_t(f.lines) * sizeof(long long) + MAX_SHARED_PASSES * sizeof(Pass);
    const long long blocks = (in.lines.count + f.lines - 1) / f.lines;
    if (blocks > INT32_MAX || !allowShared<R>(sharedFft<R, IO, PLAIN>, bytes)) return false;
    sharedFft<R, IO, PLAIN><<<unsigned(blocks), unsigned(threads), bytes, stream>>>(f, in, out,
                                                                                    static_cast<const Cx<R> *>(table));
    return cudaGetLastError() == cudaSuccess;
}

// transforms in shared memory of length n (the table: roots of unity of length n * tableStride); false if they can't
template<typename R, Io IO>
bool runShared(cudaStream_t stream, long long n, const std::vector<int> &factors, int sign, const void *table,
               int tableStride, const Endpoint &in, const Endpoint &out) {
    const std::vector<int> radices = sharedRadices(factors);
    if (radices.size() > MAX_SHARED_PASSES) return false;
    if (plainShared<R, IO>(radices, in, out)) {
        return launchShared<R, IO, true>(stream, n, radices, sign, table, tableStride, in, out);
    }
    return launchShared<R, IO, false>(stream, n, radices, sign, table, tableStride, in, out);
}

template<typename R>
constexpr cudaDataType WORK_TYPE = sizeof(R) == 8 ? CUDA_C_64F : CUDA_C_32F;

// the lines of a work buffer as an endpoint
template<typename R>
Endpoint workEndpoint(Cx<R> *data, const Lines &lines) {
    return {data, WORK_TYPE<R>, {lines.count, lines.inner, lines.outer, 1, 1, 0, lines.step}};
}

// a Stockham pass of radix r (a product of radices, transformed in shared memory) over lines of length n
struct LargePass {
    int r;
    Passes passes;  // of the radix r transforms
    int n, ns;  // ns: product of the earlier passes' radices
    int sign;
    int js;  // butterflies per block
};

/**
 * Butterflies j0 .. j0 + js of a line per block: in[j + q * per] (q < r, per = n / r), twiddled by
 * e^{-2 pi i q (j % ns) / (ns r)}, transformed, to out[(j - j % ns) r + j % ns + k ns]. Loads and stores run along j
 * where consecutive butterflies are adjacent in memory, else along q / k.
 */
constexpr int LOW_BITS = 10;

constexpr int BATCH = 4;

template<typename R>
__global__ void largePass(LargePass f, Lines lines, const Cx<R> *groupTable, const Cx<R> *low, const Cx<R> *high,
                          const Cx<R> *src, Cx<R> *dst) {
    extern __shared__ __align__(16) unsigned char shared[];
    Cx<R> *a = reinterpret_cast<Cx<R> *>(shared);
    // (32 bit index arithmetic within a line)
    Cx<R> *b = a + paddedCount(f.js * f.r);
    Pass *passes = reinterpret_cast<Pass *>(b + paddedCount(f.js * f.r));
    copyPasses(f.passes, passes);
    const int n = f.n, r = f.r, ns = f.ns, per = n / r, tstep = n / (ns * r);
    const unsigned blocksPerLine = unsigned((per + f.js - 1) / f.js);
    const long long base = lineBase(lines, blockIdx.x / blocksPerLine);
    const int j0 = int(blockIdx.x % blocksPerLine) * f.js, jn0 = j0 % ns;
    const int count = min(f.js, per - j0), total = count * r;
    const bool alongJ = count >= 8;
    // butterflies interleaved in shared memory when loading along them, else one after another
    const SharedLines s = alongJ ? SharedLines{count, r, 1, count, fastDiv(count)} : SharedLines{count, r, r, 1, fastDiv(count)};
    // (BATCH loads in flight per thread before they are used)
    for (int e0 = threadIdx.x; e0 < total; e0 += BATCH * blockDim.x) {
        Cx<R> v[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; u++) {
            const int e = e0 + u * blockDim.x;
            const int jj = alongJ ? e % count : e / r, q = alongJ ? e / count : e % r;
            if (e < total) v[u] = src[base + static_cast<long long>(j0 + jj + q * per) * lines.step];
        }
#pragma unroll
        for (int u = 0; u < BATCH; u++) {
            const int e = e0 + u * blockDim.x;
            const int jj = alongJ ? e % count : e / r, q = alongJ ? e / count : e % r;
            if (e >= total) continue;
            if (q > 0 && ns > 1) {
                int jn = jn0 + jj;
                if (jn >= ns) jn %= ns;
                const int m = q * jn * tstep;
                v[u] = cmul(v[u], twiddle(low, m & ((1 << LOW_BITS) - 1), f.sign));
                v[u] = cmul(v[u], twiddle(high, m >> LOW_BITS, f.sign));
            }
            a[padded(jj * s.ls + q * s.es)] = v[u];
        }
    }
    __syncthreads();
    a = sharedPasses(a, b, s, f.passes.count, passes, f.sign, groupTable);
    const bool storeAlongJ = ns >= 8 && count >= 8;
    for (int e = threadIdx.x; e < total; e += blockDim.x) {
        const int jj = storeAlongJ ? e % count : e / r, k = storeAlongJ ? e / count : e % r;
        int jn = jn0 + jj;
        if (jn >= ns) jn %= ns;
        dst[base + static_cast<long long>((j0 + jj - jn) * r + jn + k * ns) * lines.step] = a[padded(jj * s.ls + k * s.es)];
    }
}

template<typename R>
bool runLarge(cudaStream_t stream, const Lines &lines, long long n, long long ns, const std::vector<int> &group,
              int sign, const void *groupTable, const void *low, const void *high, const Cx<R> *src, Cx<R> *dst) {
    LargePass f{};
    f.r = 1;
    for (int r: group) f.r *= r;
    if (group.size() > MAX_PASSES) return false;
    const std::vector<int> radices = sharedRadices(group);
    if (radices.size() > MAX_SHARED_PASSES) return false;
    f.passes = passesFor(f.r, radices.data(), int(radices.size()), 1);
    if (n > INT32_MAX) return false;
    f.n = int(n);
    f.ns = int(ns);
    f.sign = sign;
    static const int maxThreads = [] {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, largePass<R>) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return attributes.maxThreadsPerBlock;
    }();
    // butterflies per block: as many as fit, fewer when their threads exceed what the kernel can launch
    long long threads = 0;
    for (f.js = int(std::max<long long>(1, std::min<long long>(LARGE_LIMIT<R> / f.r, n / f.r))); f.js >= 1;
         f.js /= 2) {
        threads = threadsFor(f.js * f.r);
        if (threads != 0 && threads <= maxThreads) break;
    }
    const size_t bytes = 2 * size_t(paddedCount(std::max(f.js, 1) * f.r)) * sizeof(Cx<R>) + MAX_SHARED_PASSES * sizeof(Pass);
    if (f.js < 1 || !allowShared<R>(largePass<R>, bytes)) return false;
    const long long blocks = lines.count * ((n / f.r + f.js - 1) / f.js);
    if (blocks > INT32_MAX) return false;
    largePass<R><<<unsigned(blocks), unsigned(threads), bytes, stream>>>(
        f, lines, static_cast<const Cx<R> *>(groupTable), static_cast<const Cx<R> *>(low),
        static_cast<const Cx<R> *>(high), src, dst);
    return cudaGetLastError() == cudaSuccess;
}

/**
 * The large radix passes of contiguous lines from src (which is only read, unless it is a) through a and b to dst
 * (nullptr: a or b); returns the buffer holding the result, nullptr if they can't run
 */
template<typename R>
Cx<R> *largeLines(cudaStream_t stream, const Lines &lines, const Transform &t, int sign, const Cx<R> *src, Cx<R> *a,
                  Cx<R> *b, Cx<R> *dst) {
    const Cx<R> *in = src;
    long long ns = 1;
    for (size_t g = 0; g < t.groups.size(); g++) {
        Cx<R> *out = g + 1 == t.groups.size() && dst != nullptr ? dst : in == a ? b : a;
        if (!runLarge<R>(stream, lines, t.n, ns, t.groups[g], sign, t.groupTwiddles[g], t.low, t.high, in, out)) {
            return nullptr;
        }
        for (int r: t.groups[g]) ns *= r;
        in = out;
    }
    return const_cast<Cx<R> *>(in);
}

/**
 * Transforms the lines; returns the buffer holding the result (a or b). Large radix passes need contiguous lines
 * (others run the radix passes).
 */
template<typename R>
Cx<R> *transformLines(cudaStream_t stream, const Lines &lines, const Transform &t, int sign, Cx<R> *a, Cx<R> *b) {
    if (t.mode == Mode::DIRECT) {
        // (in place: a block reads its lines before it writes them)
        const Endpoint e = workEndpoint<R>(a, lines);
        if (runShared<R, Io::COMPLEX>(stream, t.n, t.radices, sign, t.twiddles, 1, e, e)) return a;
    } else if (t.mode == Mode::LARGE && lines.step == 1) {
        Cx<R> *result = largeLines<R>(stream, lines, t, sign, a, a, b, nullptr);
        if (result != nullptr) return result;
    }
    return passLines<R>(stream, lines, t.n, t.radices, sign, t.twiddles, a, b);
}

// ---- Bluestein

// padded[line][k] = x[k] * chirp[k] (k < n; conjugated for inverse transforms), zero beyond
template<typename R>
__global__ void bluesteinIn(Lines lines, long long n, long long m, int sign, const Cx<R> *chirp, const Cx<R> *x,
                            Cx<R> *padded) {
    GRID_LOOP(e, lines.count * m) {
        const long long line = e / m, k = e % m;
        Cx<R> v{0, 0};
        if (k < n) {
            Cx<R> xv = x[lineBase(lines, line) + k * lines.step];
            if (sign > 0) xv = conj(xv);
            v = cmul(xv, chirp[k]);
        }
        padded[e] = v;
    }
}

template<typename R>
__global__ void bluesteinMultiply(long long count, long long m, const Cx<R> *spectrum, Cx<R> *data) {
    GRID_LOOP(e, count * m) {
        data[e] = cmul(data[e], spectrum[e % m]);
    }
}

// x[k] = chirp[k] * conv[k] / m (conjugated back for inverse transforms)
template<typename R>
__global__ void bluesteinOut(Lines lines, long long n, long long m, int sign, const Cx<R> *chirp, const Cx<R> *conv,
                             Cx<R> *x) {
    GRID_LOOP(e, lines.count * n) {
        const long long line = e / n, k = e % n;
        Cx<R> v = cmul(conv[line * m + k], chirp[k]);
        v.x /= R(m);
        v.y /= R(m);
        if (sign > 0) v = conj(v);
        x[lineBase(lines, line) + k * lines.step] = v;
    }
}

// ---- plans

std::vector<int> factorize(long long n, bool &ok) {
    std::vector<int> radices;
    for (int r: {8, 4, 2, 3, 5, 7}) {
        while (n % r == 0) {
            radices.push_back(r);
            n /= r;
        }
    }
    for (int p = 11; p <= MAX_RADIX && n > 1; p += 2) {
        while (n % p == 0) {
            radices.push_back(p);
            n /= p;
        }
    }
    ok = n == 1;
    return radices;
}

using cd = std::complex<double>;

// host FFT of a power of two length (for the chirp spectrum)
void hostFft(std::vector<cd> &a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = -2 * M_PI / double(len);
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; k++) {
                const cd w = std::polar(1.0, angle * double(k));
                const cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

void *upload(const std::vector<cd> &values, bool doubles) {
    void *device = nullptr;
    const size_t element = doubles ? sizeof(double2) : sizeof(float2);
    if (cudaMalloc(&device, std::max<size_t>(1, values.size()) * element) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    std::vector<char> bytes(values.size() * element);
    for (size_t i = 0; i < values.size(); i++) {
        if (doubles) {
            reinterpret_cast<double2 *>(bytes.data())[i] = make_double2(values[i].real(), values[i].imag());
        } else {
            reinterpret_cast<float2 *>(bytes.data())[i] = make_float2(float(values[i].real()), float(values[i].imag()));
        }
    }
    cudaMemcpy(device, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
    return device;
}

std::vector<cd> rootsOfUnity(long long n) {
    std::vector<cd> t(n);
    for (long long k = 0; k < n; k++) {
        t[k] = std::polar(1.0, -2 * M_PI * double(k) / double(n));
    }
    return t;
}

void freeTransform(Transform &t) {
    for (void *p: t.groupTwiddles) cudaFree(p);
    for (void *p: {t.twiddles, t.low, t.high}) {
        if (p != nullptr) cudaFree(p);
    }
    t = Transform{};
}

void freeAxis(Axis &a) {
    freeTransform(a.t);
    freeTransform(a.mt);
    for (void *p: {a.chirp, a.chirpSpectrum}) {
        if (p != nullptr) cudaFree(p);
    }
    a = Axis{};
}

// a transform of length n (factors up to MAX_RADIX); ok false if n has larger prime factors
cufftResult makeTransform(Transform &t, long long n, bool doubles, bool &ok) {
    t = Transform{};
    t.n = n;
    t.radices = factorize(n, ok);
    t.twiddles = upload(rootsOfUnity(n), doubles);
    if (t.twiddles == nullptr) return CUFFT_ALLOC_FAILED;
    t.mode = Mode::PASSES;
    if (!ok) return CUFFT_SUCCESS;
    if (doubles ? linesFor<double, Io::COMPLEX>(n, t.radices) > 0 : linesFor<float, Io::COMPLEX>(n, t.radices) > 0) {
        t.mode = Mode::DIRECT;
        return CUFFT_SUCCESS;
    }
    // large radix passes: the prime factors in groups (first fit, largest first) of at most groupLimit, so that a
    // block holds 16 butterflies; each group's radices as factorize gives them
    const long long groupLimit = (doubles ? LARGE_LIMIT<double> : LARGE_LIMIT<float>) / 16;
    std::vector<long long> primes;
    for (long long rest = n, p = 2; rest > 1; p++) {
        while (rest % p == 0) {
            primes.push_back(p);
            rest /= p;
        }
    }
    std::sort(primes.rbegin(), primes.rend());
    std::vector<long long> products;
    for (long long p: primes) {
        if (p > groupLimit) return CUFFT_SUCCESS;
        size_t g = 0;
        while (g < products.size() && products[g] * p > groupLimit) g++;
        if (g == products.size()) products.push_back(1);
        products[g] *= p;
    }
    for (long long product: products) {
        bool factored = false;
        t.groups.push_back(factorize(product, factored));
    }
    for (const auto &group: t.groups) {
        if (int(group.size()) > MAX_PASSES) return CUFFT_SUCCESS;
    }
    for (long long r: products) {
        t.groupTwiddles.push_back(upload(rootsOfUnity(r), doubles));
        if (t.groupTwiddles.back() == nullptr) return CUFFT_ALLOC_FAILED;
    }
    std::vector<cd> low(std::min<long long>(n, 1 << LOW_BITS)), high((n >> LOW_BITS) + 1);
    for (size_t k = 0; k < low.size(); k++) low[k] = std::polar(1.0, -2 * M_PI * double(k) / double(n));
    for (size_t k = 0; k < high.size(); k++) {
        high[k] = std::polar(1.0, -2 * M_PI * double((k << LOW_BITS) % n) / double(n));
    }
    t.low = upload(low, doubles);
    t.high = upload(high, doubles);
    if (t.low == nullptr || t.high == nullptr) return CUFFT_ALLOC_FAILED;
    t.mode = Mode::LARGE;
    return CUFFT_SUCCESS;
}

cufftResult makeAxis(Axis &a, long long n, bool doubles) {
    a = Axis{};
    bool ok = false;
    cufftResult result = makeTransform(a.t, n, doubles, ok);
    if (result != CUFFT_SUCCESS || ok) return result;
    // Bluestein: m >= 2n - 1, a power of two
    a.bluestein = true;
    long long m = 1;
    while (m < 2 * n - 1) m *= 2;
    result = makeTransform(a.mt, m, doubles, ok);
    if (result != CUFFT_SUCCESS) return result;
    std::vector<cd> chirp(n), b(m, 0);
    for (long long k = 0; k < n; k++) {
        // e^{-i pi k^2 / n}, with k^2 reduced modulo 2n for accuracy
        const long long k2 = (k * k) % (2 * n);
        chirp[k] = std::polar(1.0, -M_PI * double(k2) / double(n));
    }
    for (long long k = 0; k < n; k++) {
        b[k] = std::conj(chirp[k]);
        if (k > 0) b[m - k] = std::conj(chirp[k]);
    }
    hostFft(b);
    a.chirp = upload(chirp, doubles);
    a.chirpSpectrum = upload(b, doubles);
    return a.chirp && a.chirpSpectrum ? CUFFT_SUCCESS : CUFFT_ALLOC_FAILED;
}

long long totalOf(const Plan &p) {
    long long total = 1;
    for (int d = 0; d < p.rank; d++) total *= p.n[d];
    return total;
}

size_t workSizeOf(const Plan &p) {
    const size_t element = p.doubles ? sizeof(double2) : sizeof(float2);
    const long long elements = totalOf(p) * p.batch;
    size_t bluestein = 0;
    for (int d = 0; d < p.rank; d++) {
        if (p.axes[d].bluestein) {
            bluestein = std::max<size_t>(bluestein, size_t(elements / p.n[d]) * p.axes[d].mt.n * 2 * element);
        }
    }
    return 2 * elements * element + bluestein + 256;
}

Kind kindOf(cudaDataType in, cudaDataType out, bool &ok) {
    const bool in_complex = in == CUDA_C_16F || in == CUDA_C_32F || in == CUDA_C_64F;
    const bool out_complex = out == CUDA_C_16F || out == CUDA_C_32F || out == CUDA_C_64F;
    ok = true;
    if (in_complex && out_complex) return Kind::C2C;
    if (!in_complex && out_complex) return Kind::R2C;
    if (in_complex && !out_complex) return Kind::C2R;
    ok = false;
    return Kind::C2C;
}

bool isDouble(cudaDataType t) {
    return t == CUDA_R_64F || t == CUDA_C_64F;
}

bool isHalf(cudaDataType t) {
    return t == CUDA_R_16F || t == CUDA_C_16F;
}

cufftResult makePlan(Plan *p, int rank, const long long *n, const long long *inembed, long long istride,
                     long long idist, cudaDataType inputtype, const long long *onembed, long long ostride,
                     long long odist, cudaDataType outputtype, long long batch, size_t *workSize) {
    if (rank < 1 || rank > 3 || n == nullptr || batch < 1) {
        return CUFFT_INVALID_VALUE;
    }
    bool ok = false;
    const Kind kind = kindOf(inputtype, outputtype, ok);
    if (!ok || isDouble(inputtype) != isDouble(outputtype) || isHalf(inputtype) != isHalf(outputtype)) {
        return CUFFT_INVALID_TYPE;
    }
    for (int d = 0; d < rank; d++) {
        if (n[d] < 1) return CUFFT_INVALID_SIZE;
    }
    for (int d = 0; d < p->rank && p->made; d++) freeAxis(p->axes[d]);
    p->rank = rank;
    p->kind = kind;
    p->inType = inputtype;
    p->outType = outputtype;
    p->doubles = isDouble(inputtype);
    p->batch = batch;
    for (int d = 0; d < rank; d++) p->n[d] = n[d];
    // basic layouts (no embedding given): contiguous, the half spectrum on the complex side of real transforms
    auto basic = [&](long long *embed, bool half) {
        for (int d = 0; d < rank; d++) embed[d] = n[d];
        if (half) embed[rank - 1] = n[rank - 1] / 2 + 1;
    };
    const bool in_half = kind == Kind::C2R, out_half = kind == Kind::R2C;
    if (inembed != nullptr) {
        for (int d = 0; d < rank; d++) p->inEmbed[d] = inembed[d];
        p->inStride = istride;
        p->inDist = idist;
    } else {
        basic(p->inEmbed, in_half);
        p->inStride = 1;
        p->inDist = 1;
        for (int d = 0; d < rank; d++) p->inDist *= p->inEmbed[d];
    }
    if (onembed != nullptr) {
        for (int d = 0; d < rank; d++) p->outEmbed[d] = onembed[d];
        p->outStride = ostride;
        p->outDist = odist;
    } else {
        basic(p->outEmbed, out_half);
        p->outStride = 1;
        p->outDist = 1;
        for (int d = 0; d < rank; d++) p->outDist *= p->outEmbed[d];
    }
    for (int d = 0; d < rank; d++) {
        const cufftResult result = makeAxis(p->axes[d], n[d], p->doubles);
        if (result != CUFFT_SUCCESS) return result;
    }
    const long long half = n[rank - 1] / 2;
    bool factored = false;
    p->halfRadices = n[rank - 1] % 2 == 0 ? factorize(half, factored) : std::vector<int>{};
    auto feasible = [&](auto lines) { return lines(half, p->halfRadices, false) > 0; };
    p->realHalf = kind != Kind::C2C && n[rank - 1] % 2 == 0 && factored &&
                  (kind == Kind::R2C ? (p->doubles ? feasible(linesFor<double, Io::REAL_FORWARD>)
                                                   : feasible(linesFor<float, Io::REAL_FORWARD>))
                                     : (p->doubles ? feasible(linesFor<double, Io::REAL_INVERSE>)
                                                   : feasible(linesFor<float, Io::REAL_INVERSE>)));
    p->workSize = workSizeOf(*p);
    if (p->ownWork != nullptr) {
        cudaFree(p->ownWork);
        p->ownWork = nullptr;
    }
    if (p->autoAllocate) {
        if (cudaMalloc(&p->ownWork, p->workSize) != cudaSuccess) {
            cudaGetLastError();
            return CUFFT_ALLOC_FAILED;
        }
        p->workArea = p->ownWork;
    }
    if (workSize != nullptr) *workSize = p->workSize;
    p->made = true;
    return CUFFT_SUCCESS;
}

// the lines of axis d of a work buffer with these dimensions
Lines workLines(const long long *dims, int rank, int d, long long elements) {
    long long inner = 1;
    for (int e = d + 1; e < rank; e++) inner *= dims[e];
    return {elements / dims[d], inner, dims[d] * inner, inner};
}

// the lines of axis d of the caller's data of these dimensions and layout
Endpoint userEndpoint(const void *data, cudaDataType type, int rank, const long long *dims, const Layout &layout,
                      long long batch, int d) {
    auto span = [&](int axis) {
        long long v = layout.stride;
        for (int e = axis + 1; e < rank; e++) v *= layout.embed[e];
        return v;
    };
    int others[2], count = 0;
    long long within = 1;
    for (int e = 0; e < rank; e++) {
        if (e == d) continue;
        others[count++] = e;
        within *= dims[e];
    }
    LineSet lines{batch * within, within, layout.dist, 1, 0, 0, span(d)};
    if (count >= 1) lines.s2 = span(others[0]);
    if (count == 2) {
        lines.c2 = dims[others[1]];
        lines.s3 = span(others[1]);
    }
    return {const_cast<void *>(data), type, lines};
}

// transforms axis d of a work buffer; returns the buffer holding the result (current or other)
template<typename R>
Cx<R> *transformAxis(cudaStream_t stream, const Axis &axis, const Lines &lines, int sign, Cx<R> *current,
                     Cx<R> *other, Cx<R> *extra) {
    if (!axis.bluestein) return transformLines<R>(stream, lines, axis.t, sign, current, other);
    // Bluestein: chirp, convolution with the conjugate chirp (power of two transforms), chirp
    const long long n = axis.t.n, m = axis.mt.n;
    Cx<R> *padded = extra, *spare = extra + lines.count * m;
    bluesteinIn<R><<<gridFor(lines.count * m), 256, 0, stream>>>(lines, n, m, sign,
                                                                 static_cast<const Cx<R> *>(axis.chirp), current,
                                                                 padded);
    const Lines contiguous{lines.count, 1, m, 1};
    Cx<R> *spectrum = transformLines<R>(stream, contiguous, axis.mt, -1, padded, spare);
    bluesteinMultiply<R><<<gridFor(lines.count * m), 256, 0, stream>>>(
        lines.count, m, static_cast<const Cx<R> *>(axis.chirpSpectrum), spectrum);
    Cx<R> *conv = transformLines<R>(stream, contiguous, axis.mt, 1, spectrum, spectrum == padded ? spare : padded);
    bluesteinOut<R><<<gridFor(lines.count * n), 256, 0, stream>>>(lines, n, m, sign,
                                                                  static_cast<const Cx<R> *>(axis.chirp), conv,
                                                                  current);
    return current;
}

// whether the caller's data is laid out like a work buffer of these dimensions (contiguous, of the work type)
bool sameLayout(const Plan &p, cudaDataType type, const Layout &layout, const long long *dims, cudaDataType work) {
    long long total = 1;
    for (int d = 0; d < p.rank; d++) {
        if (layout.embed[d] != dims[d] && d > 0) return false;
        total *= dims[d];
    }
    return type == work && layout.stride == 1 && (layout.dist == total || p.batch == 1);
}

/**
 * Transforms in shared memory read the caller's input and write its output directly where they can (the first and
 * last axes); other axes run on the work buffers, with the input gathered and the output scattered.
 */
template<typename R>
cufftResult execute(Plan *p, const void *input, void *output, int direction) {
    cudaStream_t stream = p->stream;
    const int rank = p->rank;
    const long long full = totalOf(*p) * p->batch;
    auto *a = static_cast<Cx<R> *>(p->workArea);
    Cx<R> *b = a + full, *extra = b + full;
    const Layout in{{p->inEmbed[0], p->inEmbed[1], p->inEmbed[2]}, p->inStride, p->inDist};
    const Layout out{{p->outEmbed[0], p->outEmbed[1], p->outEmbed[2]}, p->outStride, p->outDist};
    const int sign = p->kind == Kind::R2C ? -1 : p->kind == Kind::C2R ? 1 : direction == CUFFT_INVERSE ? 1 : -1;
    auto launched = [] { return cudaGetLastError() == cudaSuccess ? CUFFT_SUCCESS : CUFFT_EXEC_FAILED; };
    if (p->kind != Kind::C2C && !p->realHalf) {
        // real transforms as complex ones: the complete input, every axis, the output's half
        const Shape s{rank, {p->n[0], rank > 1 ? p->n[1] : 1, rank > 2 ? p->n[2] : 1}, totalOf(*p)};
        gather<R><<<gridFor(full), 256, 0, stream>>>(s, p->batch, p->kind, in, p->inType, input, a);
        Cx<R> *current = a;
        for (int d = 0; d < rank; d++) {
            current = transformAxis<R>(stream, p->axes[d], workLines(p->n, rank, d, full), sign, current,
                                       current == a ? b : a, extra);
        }
        scatter<R><<<gridFor(full), 256, 0, stream>>>(s, p->batch, p->kind, out, p->outType, current, output);
        return launched();
    }
    // the work buffer's dimensions: the half spectrum's for real transforms
    long long dims[3] = {p->n[0], rank > 1 ? p->n[1] : 1, rank > 2 ? p->n[2] : 1};
    if (p->kind != Kind::C2C) dims[rank - 1] = p->n[rank - 1] / 2 + 1;
    long long elements = p->batch;
    for (int d = 0; d < rank; d++) elements *= dims[d];
    const Shape halfShape{rank, {dims[0], dims[1], dims[2]}, elements / p->batch};
    const Axis &last = p->axes[rank - 1];
    const long long m = p->n[rank - 1] / 2;
    // the complex axes, in order
    std::vector<int> order;
    if (p->kind == Kind::C2C) {
        for (int d = rank - 1; d >= 0; d--) order.push_back(d);
    } else if (p->kind == Kind::R2C) {
        for (int d = rank - 2; d >= 0; d--) order.push_back(d);
    } else {
        for (int d = 0; d < rank - 1; d++) order.push_back(d);
    }
    Cx<R> *current = nullptr;  // (no work buffer yet: the data is the caller's input)
    if (p->kind == Kind::R2C) {
        const Endpoint src = userEndpoint(input, p->inType, rank, p->n, in, p->batch, rank - 1);
        const Endpoint dst = order.empty() ? userEndpoint(output, p->outType, rank, dims, out, p->batch, rank - 1)
                                           : workEndpoint<R>(a, workLines(dims, rank, rank - 1, elements));
        if (!runShared<R, Io::REAL_FORWARD>(stream, m, p->halfRadices, -1, last.t.twiddles, 2, src, dst)) {
            return CUFFT_EXEC_FAILED;
        }
        current = a;
    }
    for (size_t i = 0; i < order.size(); i++) {
        const int d = order[i];
        const bool lastStep = i + 1 == order.size() && p->kind != Kind::C2R;
        const Axis &axis = p->axes[d];
        const Lines lines = workLines(dims, rank, d, elements);
        if (!axis.bluestein && axis.t.mode == Mode::DIRECT) {
            const Endpoint src = current != nullptr ? workEndpoint<R>(current, lines)
                                                    : userEndpoint(input, p->inType, rank, dims, in, p->batch, d);
            Cx<R> *target = current != nullptr ? current : a;
            const Endpoint dst = lastStep ? userEndpoint(output, p->outType, rank, dims, out, p->batch, d)
                                          : workEndpoint<R>(target, lines);
            if (!runShared<R, Io::COMPLEX>(stream, axis.t.n, axis.t.radices, sign, axis.t.twiddles, 1, src, dst)) {
                return CUFFT_EXEC_FAILED;
            }
            current = target;
            continue;
        }
        // (the caller's data as the work buffer when it has its layout: the first pass reads it, the last writes it)
        const bool inputLike = current == nullptr && sameLayout(*p, p->inType, in, dims, WORK_TYPE<R>);
        const bool outputLike = lastStep && sameLayout(*p, p->outType, out, dims, WORK_TYPE<R>);
        if (!axis.bluestein && axis.t.mode == Mode::LARGE && lines.step == 1 && (inputLike || outputLike)) {
            const Cx<R> *src = inputLike ? static_cast<const Cx<R> *>(input) : current;
            if (src == nullptr) {
                gather<R><<<gridFor(elements), 256, 0, stream>>>(halfShape, p->batch, Kind::C2C, in, p->inType, input,
                                                                 a);
                src = a;
            }
            Cx<R> *result = largeLines<R>(stream, lines, axis.t, sign, src, src == a ? b : a, src == a ? a : b,
                                          outputLike ? static_cast<Cx<R> *>(output) : nullptr);
            if (result != nullptr) {
                if (outputLike) break;
                current = result;
                continue;
            }
            // (they can't run: the radix passes on the work buffers)
            if (src == a) current = a;
        }
        if (current == nullptr) {
            gather<R><<<gridFor(elements), 256, 0, stream>>>(halfShape, p->batch, Kind::C2C, in, p->inType, input, a);
            current = a;
        }
        current = transformAxis<R>(stream, axis, lines, sign, current, current == a ? b : a, extra);
        if (lastStep) {
            scatter<R><<<gridFor(elements), 256, 0, stream>>>(halfShape, p->batch, Kind::C2C, out, p->outType, current,
                                                              output);
        }
    }
    if (p->kind == Kind::C2R) {
        const Endpoint src = current != nullptr
                                 ? workEndpoint<R>(current, workLines(dims, rank, rank - 1, elements))
                                 : userEndpoint(input, p->inType, rank, dims, in, p->batch, rank - 1);
        const Endpoint dst = userEndpoint(output, p->outType, rank, p->n, out, p->batch, rank - 1);
        if (!runShared<R, Io::REAL_INVERSE>(stream, m, p->halfRadices, 1, last.t.twiddles, 2, src, dst)) {
            return CUFFT_EXEC_FAILED;
        }
    }
    return launched();
}

cufftResult exec(cufftHandle plan, const void *input, void *output, int direction) {
    Plan *p = planOf(plan);
    if (p == nullptr || !p->made) return CUFFT_INVALID_PLAN;
    if (p->workArea == nullptr || input == nullptr || output == nullptr) return CUFFT_INVALID_VALUE;
    return p->doubles ? execute<double>(p, input, output, direction) : execute<float>(p, input, output, direction);
}

void typesOf(cufftType type, cudaDataType &in, cudaDataType &out) {
    switch (type) {
        case CUFFT_R2C: in = CUDA_R_32F; out = CUDA_C_32F; break;
        case CUFFT_C2R: in = CUDA_C_32F; out = CUDA_R_32F; break;
        case CUFFT_D2Z: in = CUDA_R_64F; out = CUDA_C_64F; break;
        case CUFFT_Z2D: in = CUDA_C_64F; out = CUDA_R_64F; break;
        case CUFFT_Z2Z: in = CUDA_C_64F; out = CUDA_C_64F; break;
        default: in = CUDA_C_32F; out = CUDA_C_32F; break;
    }
}

cufftResult makeMany(cufftHandle plan, int rank, const int *n, const int *inembed, int istride, int idist,
                     const int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    if (rank < 1 || rank > 3 || n == nullptr) return CUFFT_INVALID_VALUE;
    long long n64[3], in64[3], on64[3];
    for (int d = 0; d < rank; d++) {
        n64[d] = n[d];
        in64[d] = inembed != nullptr ? inembed[d] : 0;
        on64[d] = onembed != nullptr ? onembed[d] : 0;
    }
    cudaDataType in, out;
    typesOf(type, in, out);
    return makePlan(p, rank, n64, inembed != nullptr ? in64 : nullptr, istride, idist, in,
                    onembed != nullptr ? on64 : nullptr, ostride, odist, out, batch, workSize);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the API

cufftResult cufftCreate(cufftHandle *handle) {
    if (handle == nullptr) return CUFFT_INVALID_VALUE;
    std::lock_guard lock(g_mutex);
    auto *p = new Plan{};
    p->autoAllocate = true;
    *handle = g_next++;
    g_plans[*handle] = p;
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
    Plan *p;
    {
        std::lock_guard lock(g_mutex);
        auto it = g_plans.find(plan);
        if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
        p = it->second;
        g_plans.erase(it);
    }
    for (int d = 0; d < 3; d++) freeAxis(p->axes[d]);
    if (p->ownWork != nullptr) cudaFree(p->ownWork);
    delete p;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    p->stream = stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    p->autoAllocate = autoAllocate != 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    p->workArea = workArea;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetSize(cufftHandle plan, size_t *workSize) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    if (workSize == nullptr) return CUFFT_INVALID_VALUE;
    *workSize = p->workSize;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetVersion(int *version) {
    if (version == nullptr) return CUFFT_INVALID_VALUE;
    *version = CUFFT_VERSION;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) return CUFFT_INVALID_VALUE;
    switch (type) {
        case MAJOR_VERSION: *value = CUFFT_VER_MAJOR; return CUFFT_SUCCESS;
        case MINOR_VERSION: *value = CUFFT_VER_MINOR; return CUFFT_SUCCESS;
        case PATCH_LEVEL: *value = CUFFT_VER_PATCH; return CUFFT_SUCCESS;
        default: return CUFFT_INVALID_TYPE;
    }
}

cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                                long long int istride, long long int idist, cudaDataType inputtype,
                                long long int *onembed, long long int ostride, long long int odist,
                                cudaDataType outputtype, long long int batch, size_t *workSize,
                                cudaDataType executiontype) {
    Plan *p = planOf(plan);
    if (p == nullptr) return CUFFT_INVALID_PLAN;
    (void) executiontype;
    return makePlan(p, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch,
                    workSize);
}

cufftResult cufftXtGetSizeMany(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                               long long int istride, long long int idist, cudaDataType inputtype,
                               long long int *onembed, long long int ostride, long long int odist,
                               cudaDataType outputtype, long long int batch, size_t *workSize,
                               cudaDataType) {
    if (planOf(plan) == nullptr) return CUFFT_INVALID_PLAN;
    // (sizes only: a temporary plan without allocation)
    Plan probe{};
    probe.autoAllocate = false;
    const cufftResult result = makePlan(&probe, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist,
                                        outputtype, batch, workSize);
    for (int d = 0; d < 3; d++) freeAxis(probe.axes[d]);
    return result;
}

cufftResult cufftXtExec(cufftHandle plan, void *input, void *output, int direction) {
    return exec(plan, input, output, direction);
}

cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed,
                              int ostride, int odist, cufftType type, int batch, size_t *workSize) {
    return makeMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize) {
    return makeMany(plan, 1, &nx, nullptr, 1, 0, nullptr, 1, 0, type, batch, workSize);
}

cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize) {
    const int n[2] = {nx, ny};
    return makeMany(plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}

cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize) {
    const int n[3] = {nx, ny, nz};
    return makeMany(plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1, workSize);
}

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed,
                          int ostride, int odist, cufftType type, int batch) {
    cufftResult result = cufftCreate(plan);
    if (result == CUFFT_SUCCESS) {
        result = makeMany(*plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
    }
    return result;
}

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) {
    return cufftPlanMany(plan, 1, &nx, nullptr, 1, 0, nullptr, 1, 0, type, batch);
}

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) {
    int n[2] = {nx, ny};
    return cufftPlanMany(plan, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}

cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) {
    int n[3] = {nx, ny, nz};
    return cufftPlanMany(plan, 3, n, nullptr, 1, 0, nullptr, 1, 0, type, 1);
}

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
    return exec(plan, idata, odata, direction);
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) {
    return exec(plan, idata, odata, CUFFT_FORWARD);
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) {
    return exec(plan, idata, odata, CUFFT_INVERSE);
}

cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction) {
    return exec(plan, idata, odata, direction);
}

cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata) {
    return exec(plan, idata, odata, CUFFT_FORWARD);
}

cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata) {
    return exec(plan, idata, odata, CUFFT_INVERSE);
}
