/**
 * LibreCUDA's cuFFT (libcufft.so.12): batched 1D / 2D / 3D complex-to-complex, real-to-complex and complex-to-real
 * transforms of half, float and double data, with cuFFT's advanced data layouts (embedding, stride, distance).
 *
 * A transform gathers its input into a complex work buffer (complex-to-real inputs are completed by Hermitian
 * symmetry), transforms each dimension with Stockham passes (self-sorting, mixed radix: 2 to 8 and generic small
 * primes up to 61; lengths with larger prime factors use Bluestein's algorithm with a power of two transform), then
 * scatters the result in the output's layout and type. Half data is computed in float. Twiddle factors come from tables
 * computed in double precision when the plan is made.
 */
#include <cufft.h>
#include <cufftXt.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr int MAX_RADIX = 61;

enum class Kind {
    C2C, R2C, C2R
};

struct Axis {
    long long n;
    std::vector<int> radices;
    // Bluestein: transform length m (power of two), its radices
    bool bluestein;
    long long m;
    std::vector<int> mRadices;
    // device tables (complex of the compute precision): e^{-2 pi i k / n}, e^{-2 pi i k / m}, chirp e^{-i pi k^2 / n},
    // FFT_m of the conjugate chirp
    void *twiddles, *mTwiddles, *chirp, *chirpSpectrum;
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

unsigned int gridFor(long long count) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + 255) / 256, 1 << 20)));
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

// runs the passes of an axis on the lines; returns the buffer holding the result (a or b)
template<typename R>
Cx<R> *transformLines(cudaStream_t stream, const Lines &lines, long long n, const std::vector<int> &radices, int sign,
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

void freeAxis(Axis &a) {
    for (void *p: {a.twiddles, a.mTwiddles, a.chirp, a.chirpSpectrum}) {
        if (p != nullptr) cudaFree(p);
    }
    a = Axis{};
}

cufftResult makeAxis(Axis &a, long long n, bool doubles) {
    a = Axis{};
    a.n = n;
    bool ok = false;
    a.radices = factorize(n, ok);
    a.twiddles = upload(rootsOfUnity(n), doubles);
    if (a.twiddles == nullptr) return CUFFT_ALLOC_FAILED;
    if (ok) return CUFFT_SUCCESS;
    // Bluestein: m >= 2n - 1, a power of two
    a.bluestein = true;
    a.m = 1;
    while (a.m < 2 * n - 1) a.m *= 2;
    a.mRadices = factorize(a.m, ok);
    a.mTwiddles = upload(rootsOfUnity(a.m), doubles);
    std::vector<cd> chirp(n), b(a.m, 0);
    for (long long k = 0; k < n; k++) {
        // e^{-i pi k^2 / n}, with k^2 reduced modulo 2n for accuracy
        const long long k2 = (k * k) % (2 * n);
        chirp[k] = std::polar(1.0, -M_PI * double(k2) / double(n));
    }
    for (long long k = 0; k < n; k++) {
        b[k] = std::conj(chirp[k]);
        if (k > 0) b[a.m - k] = std::conj(chirp[k]);
    }
    hostFft(b);
    a.chirp = upload(chirp, doubles);
    a.chirpSpectrum = upload(b, doubles);
    return a.mTwiddles && a.chirp && a.chirpSpectrum ? CUFFT_SUCCESS : CUFFT_ALLOC_FAILED;
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
            bluestein = std::max<size_t>(bluestein, size_t(elements / p.n[d]) * p.axes[d].m * 2 * element);
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

template<typename R>
cufftResult execute(Plan *p, const void *input, void *output, int direction) {
    cudaStream_t stream = p->stream;
    const long long total = totalOf(*p), elements = total * p->batch;
    auto *a = static_cast<Cx<R> *>(p->workArea);
    Cx<R> *b = a + elements;
    auto *extra = reinterpret_cast<Cx<R> *>(b + elements);
    Shape s{p->rank, {p->n[0], p->rank > 1 ? p->n[1] : 1, p->rank > 2 ? p->n[2] : 1}, total};
    const Layout in{{p->inEmbed[0], p->inEmbed[1], p->inEmbed[2]}, p->inStride, p->inDist};
    const Layout out{{p->outEmbed[0], p->outEmbed[1], p->outEmbed[2]}, p->outStride, p->outDist};
    gather<R><<<gridFor(elements), 256, 0, stream>>>(s, p->batch, p->kind, in, p->inType, input, a);
    const int sign = p->kind == Kind::R2C ? -1 : p->kind == Kind::C2R ? 1 : direction == CUFFT_INVERSE ? 1 : -1;
    Cx<R> *current = a, *other = b;
    for (int d = 0; d < p->rank; d++) {
        const Axis &axis = p->axes[d];
        long long inner = 1;
        for (int e = d + 1; e < p->rank; e++) inner *= p->n[e];
        const Lines lines{elements / axis.n, inner, axis.n * inner, inner};
        if (!axis.bluestein) {
            Cx<R> *result = transformLines<R>(stream, lines, axis.n, axis.radices, sign, axis.twiddles, current, other);
            if (result != current) std::swap(current, other);
            continue;
        }
        // Bluestein: chirp, convolution with the conjugate chirp (power of two transforms), chirp
        Cx<R> *padded = extra, *spare = extra + lines.count * axis.m;
        bluesteinIn<R><<<gridFor(lines.count * axis.m), 256, 0, stream>>>(
            lines, axis.n, axis.m, sign, static_cast<const Cx<R> *>(axis.chirp), current, padded);
        const Lines contiguous{lines.count, 1, axis.m, 1};
        Cx<R> *spectrum = transformLines<R>(stream, contiguous, axis.m, axis.mRadices, -1, axis.mTwiddles, padded,
                                            spare);
        bluesteinMultiply<R><<<gridFor(lines.count * axis.m), 256, 0, stream>>>(
            lines.count, axis.m, static_cast<const Cx<R> *>(axis.chirpSpectrum), spectrum);
        Cx<R> *conv = transformLines<R>(stream, contiguous, axis.m, axis.mRadices, 1, axis.mTwiddles, spectrum,
                                        spectrum == padded ? spare : padded);
        bluesteinOut<R><<<gridFor(lines.count * axis.n), 256, 0, stream>>>(
            lines, axis.n, axis.m, sign, static_cast<const Cx<R> *>(axis.chirp), conv, current);
    }
    scatter<R><<<gridFor(elements), 256, 0, stream>>>(s, p->batch, p->kind, out, p->outType, current, output);
    return cudaGetLastError() == cudaSuccess ? CUFFT_SUCCESS : CUFFT_EXEC_FAILED;
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
