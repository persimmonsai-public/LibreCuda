/**
 * LibreCUDA's cuDNN: elementwise operations, batch normalization, pooling and the spatial transformer, on tensors of
 * any strides. Elements are read and written as their descriptor's data type (half, bfloat16, float, double) and
 * computed in double (statistics) or float.
 */
#include "blas_internal.h"
#include "dnn_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <utility>

namespace librecuda_dnn {
namespace {

#define GRID_LOOP(i, total) \
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < (total); \
         i += static_cast<long long>(gridDim.x) * blockDim.x)

__device__ inline double load(const void *base, long long index, cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_HALF: return __half2float(static_cast<const __half *>(base)[index]);
        case CUDNN_DATA_BFLOAT16: return __bfloat162float(static_cast<const __nv_bfloat16 *>(base)[index]);
        case CUDNN_DATA_DOUBLE: return static_cast<const double *>(base)[index];
        default: return static_cast<const float *>(base)[index];
    }
}

__device__ inline void store(void *base, long long index, cudnnDataType_t type, double value) {
    switch (type) {
        case CUDNN_DATA_HALF: static_cast<__half *>(base)[index] = __float2half(float(value)); break;
        case CUDNN_DATA_BFLOAT16: static_cast<__nv_bfloat16 *>(base)[index] = __float2bfloat16(float(value)); break;
        case CUDNN_DATA_DOUBLE: static_cast<double *>(base)[index] = value; break;
        default: static_cast<float *>(base)[index] = float(value); break;
    }
}

__host__ __device__ inline long long elements(const Tensor &t) {
    long long n = 1;
    for (int i = 0; i < t.nbDims; i++) n *= t.dims[i];
    return n;
}

// the element offset of the e-th element (row-major order of the dimensions)
__device__ inline long long offsetOf(const Tensor &t, long long e) {
    long long offset = 0;
    for (int d = t.nbDims - 1; d >= 0; d--) {
        offset += (e % t.dims[d]) * t.strides[d];
        e /= t.dims[d];
    }
    return offset;
}

unsigned int gridFor(long long count) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + 255) / 256, 1 << 20)));
}

cudnnStatus_t launched() {
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

__device__ inline double activate(double x, cudnnActivationMode_t mode, double coef) {
    switch (mode) {
        case CUDNN_ACTIVATION_SIGMOID: return 1 / (1 + exp(-x));
        case CUDNN_ACTIVATION_RELU: return x > 0 ? x : 0;
        case CUDNN_ACTIVATION_TANH: return tanh(x);
        case CUDNN_ACTIVATION_CLIPPED_RELU: return fmin(fmax(x, 0.0), coef);
        case CUDNN_ACTIVATION_ELU: return x > 0 ? x : coef * (exp(x) - 1);
        case CUDNN_ACTIVATION_SWISH: return x / (1 + exp(-coef * x));
        default: return x;
    }
}

// derivative of the activation, from its output y (and input x where needed)
__device__ inline double activationGrad(double y, double x, cudnnActivationMode_t mode, double coef) {
    switch (mode) {
        case CUDNN_ACTIVATION_SIGMOID: return y * (1 - y);
        case CUDNN_ACTIVATION_RELU: return y > 0 ? 1 : 0;
        case CUDNN_ACTIVATION_TANH: return 1 - y * y;
        case CUDNN_ACTIVATION_CLIPPED_RELU: return y > 0 && y < coef ? 1 : 0;
        case CUDNN_ACTIVATION_ELU: return x > 0 ? 1 : y + coef;
        case CUDNN_ACTIVATION_SWISH: {
            const double s = 1 / (1 + exp(-coef * x));
            return s + coef * x * s * (1 - s);
        }
        default: return 1;
    }
}

__global__ void biasActivationKernel(Tensor y, void *data, const void *bias, cudnnActivationMode_t mode, double coef) {
    GRID_LOOP(e, elements(y)) {
        const long long offset = offsetOf(y, e);
        double v = load(data, offset, y.type);
        if (bias != nullptr) {
            // channel: dimension 1
            long long rest = e;
            for (int d = y.nbDims - 1; d > 1; d--) rest /= y.dims[d];
            v += load(bias, rest % y.dims[1], y.type);
        }
        store(data, offset, y.type, activate(v, mode, coef));
    }
}

__global__ void copyKernel(Tensor src, const void *srcData, Tensor dst, void *dstData) {
    GRID_LOOP(e, elements(dst)) {
        store(dstData, offsetOf(dst, e), dst.type, load(srcData, offsetOf(src, e), src.type));
    }
}

// ---- batch normalization

constexpr int THREADS = 256;

struct BnLayout {
    // features (C, or C * spatial for per-activation mode) and the positions reduced over per feature
    long long features, count;
    int spatial;  // product of the spatial dimensions
};

// offset of reduction position r of feature f
__device__ inline long long bnOffset(const Tensor &t, const BnLayout &l, bool perActivation, long long f,
                                     long long r) {
    // spatial mode: f = c, r = (n, spatial); per activation: f = (c, spatial), r = n
    long long n, c, s;
    if (perActivation) {
        n = r;
        c = f / l.spatial;
        s = f % l.spatial;
    } else {
        c = f;
        n = r / l.spatial;
        s = r % l.spatial;
    }
    long long offset = n * t.strides[0] + c * t.strides[1];
    for (int d = t.nbDims - 1; d >= 2; d--) {
        offset += (s % t.dims[d]) * t.strides[d];
        s /= t.dims[d];
    }
    return offset;
}

__device__ inline double blockSum(double value, double *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const double result = shared[0];
    __syncthreads();
    return result;
}

__global__ void bnForwardTraining(BatchNorm bn, BnLayout l, bool perActivation, double alpha, double beta, Tensor x,
                                  const void *xData, Tensor z, const void *zData, Tensor y, void *yData,
                                  cudnnDataType_t paramType, const void *scale, const void *bias, double factor,
                                  void *runningMean, void *runningVar, void *saveMean, void *saveInvStd) {
    __shared__ double shared[THREADS];
    for (long long f = blockIdx.x; f < l.features; f += gridDim.x) {
        double sum = 0;
        for (long long r = threadIdx.x; r < l.count; r += THREADS) sum += load(xData, bnOffset(x, l, perActivation, f, r), x.type);
        const double mean = blockSum(sum, shared) / l.count;
        double sq = 0;
        for (long long r = threadIdx.x; r < l.count; r += THREADS) {
            const double d = load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean;
            sq += d * d;
        }
        const double var = blockSum(sq, shared) / l.count;
        const double inv = 1 / sqrt(var + bn.epsilon);
        const double g = load(scale, f, paramType), b = load(bias, f, paramType);
        for (long long r = threadIdx.x; r < l.count; r += THREADS) {
            double v = (load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean) * inv * g + b;
            if (bn.ops == 2 && zData != nullptr) v += load(zData, bnOffset(z, l, perActivation, f, r), z.type);
            if (bn.ops >= 1) v = activate(v, bn.activation, bn.activationCoef);
            const long long out = bnOffset(y, l, perActivation, f, r);
            double result = alpha * v;
            if (beta != 0) result += beta * load(yData, out, y.type);
            store(yData, out, y.type, result);
        }
        if (threadIdx.x == 0) {
            if (runningMean != nullptr) {
                store(runningMean, f, paramType, (1 - factor) * load(runningMean, f, paramType) + factor * mean);
            }
            if (runningVar != nullptr) {
                const double unbiased = l.count > 1 ? var * l.count / (l.count - 1) : var;
                store(runningVar, f, paramType, (1 - factor) * load(runningVar, f, paramType) + factor * unbiased);
            }
            if (saveMean != nullptr) store(saveMean, f, paramType, mean);
            if (saveInvStd != nullptr) store(saveInvStd, f, paramType, inv);
        }
        __syncthreads();
    }
}

__global__ void bnForwardInference(BatchNorm bn, BnLayout l, bool perActivation, double alpha, double beta, Tensor x,
                                   const void *xData, Tensor y, void *yData, cudnnDataType_t paramType,
                                   const void *scale, const void *bias, const void *mean, const void *var) {
    GRID_LOOP(e, l.features * l.count) {
        const long long f = e / l.count, r = e % l.count;
        const double inv = 1 / sqrt(load(var, f, paramType) + bn.epsilon);
        const double v = (load(xData, bnOffset(x, l, perActivation, f, r), x.type) - load(mean, f, paramType)) * inv *
                         load(scale, f, paramType) + load(bias, f, paramType);
        const long long out = bnOffset(y, l, perActivation, f, r);
        double result = alpha * v;
        if (beta != 0) result += beta * load(yData, out, y.type);
        store(yData, out, y.type, result);
    }
}

__global__ void bnBackward(BatchNorm bn, BnLayout l, bool perActivation, double alphaData, double betaData,
                           double alphaParam, double betaParam, Tensor x, const void *xData, Tensor y,
                           const void *yData, Tensor dy, const void *dyData, Tensor dz, void *dzData, Tensor dx,
                           void *dxData, cudnnDataType_t paramType, const void *scale, const void *bias, void *dScale,
                           void *dBias, const void *savedMean, const void *savedInvStd) {
    __shared__ double shared[THREADS];
    for (long long f = blockIdx.x; f < l.features; f += gridDim.x) {
        double mean, inv;
        if (savedMean != nullptr && savedInvStd != nullptr) {
            mean = load(savedMean, f, paramType);
            inv = load(savedInvStd, f, paramType);
        } else {
            double sum = 0;
            for (long long r = threadIdx.x; r < l.count; r += THREADS) sum += load(xData, bnOffset(x, l, perActivation, f, r), x.type);
            mean = blockSum(sum, shared) / l.count;
            double sq = 0;
            for (long long r = threadIdx.x; r < l.count; r += THREADS) {
                const double d = load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean;
                sq += d * d;
            }
            inv = 1 / sqrt(blockSum(sq, shared) / l.count + bn.epsilon);
        }
        const double g = load(scale, f, paramType), b = bias != nullptr ? load(bias, f, paramType) : 0;
        // the gradient at the normalization's output (through the activation; the addend's gradient is the same)
        auto gradient = [&](long long r) {
            double grad = load(dyData, bnOffset(dy, l, perActivation, f, r), dy.type);
            if (bn.ops >= 1) {
                const double xhat = (load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean) * inv;
                const double pre = xhat * g + b;
                const double out = yData != nullptr ? load(yData, bnOffset(y, l, perActivation, f, r), y.type)
                                                    : activate(pre, bn.activation, bn.activationCoef);
                grad *= activationGrad(out, pre, bn.activation, bn.activationCoef);
            }
            return grad;
        };
        double sum_dy = 0, sum_dy_xhat = 0;
        for (long long r = threadIdx.x; r < l.count; r += THREADS) {
            const double grad = gradient(r);
            const double xhat = (load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean) * inv;
            sum_dy += grad;
            sum_dy_xhat += grad * xhat;
            if (bn.ops == 2 && dzData != nullptr) store(dzData, bnOffset(dz, l, perActivation, f, r), dz.type, grad);
        }
        sum_dy = blockSum(sum_dy, shared);
        sum_dy_xhat = blockSum(sum_dy_xhat, shared);
        for (long long r = threadIdx.x; r < l.count; r += THREADS) {
            const double grad = gradient(r);
            const double xhat = (load(xData, bnOffset(x, l, perActivation, f, r), x.type) - mean) * inv;
            const double value = g * inv / l.count * (l.count * grad - sum_dy - xhat * sum_dy_xhat);
            const long long out = bnOffset(dx, l, perActivation, f, r);
            double result = alphaData * value;
            if (betaData != 0) result += betaData * load(dxData, out, dx.type);
            store(dxData, out, dx.type, result);
        }
        if (threadIdx.x == 0) {
            if (dScale != nullptr) {
                double v = alphaParam * sum_dy_xhat;
                if (betaParam != 0) v += betaParam * load(dScale, f, paramType);
                store(dScale, f, paramType, v);
            }
            if (dBias != nullptr) {
                double v = alphaParam * sum_dy;
                if (betaParam != 0) v += betaParam * load(dBias, f, paramType);
                store(dBias, f, paramType, v);
            }
        }
        __syncthreads();
    }
}

BnLayout bnLayout(const Tensor &x, bool perActivation) {
    BnLayout l{};
    long long spatial = 1;
    for (int d = 2; d < x.nbDims; d++) spatial *= x.dims[d];
    l.spatial = static_cast<int>(spatial);
    l.features = perActivation ? x.dims[1] * spatial : x.dims[1];
    l.count = perActivation ? x.dims[0] : x.dims[0] * spatial;
    return l;
}

// ---- batch normalization over (N, spatial) of tensors whose spatial positions have one stride (NCHW, NHWC, ...) and
// one data type T: statistics as per-thread Welford triples (accumulation type A) merged per block, partials per
// (channel, image group) merged in double; then an elementwise pass

// strides of a tensor as (image, channel, position); false if its spatial positions don't have one stride
struct Planes {
    long long n, c, s;
};

bool planesOf(const Tensor &t, Planes &p) {
    p.n = t.strides[0];
    p.c = t.strides[1];
    p.s = 1;
    // the innermost dimension of size > 1 sets the stride; the outer ones must follow it
    long long expected = -1;
    for (int d = t.nbDims - 1; d >= 2; d--) {
        if (t.dims[d] == 1) continue;
        if (expected < 0) {
            p.s = t.strides[d];
            expected = t.strides[d];
        }
        if (t.strides[d] != expected) return false;
        expected *= t.dims[d];
    }
    return true;
}

struct BnFast {
    long long images, channels, positions;
    int groups;  // image groups per channel (reduction blocks); row groups per 32 channels when last
    // all tensors are channels-last (rows of the channels one after another: NHWC, ...): the passes run along the
    // rows (a (image, channel) plane of such a tensor has its elements C apart: the plane kernels read a sector per
    // element, and ResNet-50's batch normalizations took 37 times as long as cuDNN's)
    bool last;
    Planes x, y, z, dy, dx, dz;
};

template<typename T>
__device__ inline float loadF(const T *p) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(*p);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(*p);
    } else {
        return float(*p);
    }
}

template<typename T, typename A>
__device__ inline A loadA(const T *p) {
    if constexpr (std::is_same_v<T, double>) {
        return *p;
    } else {
        return A(loadF(p));
    }
}

template<typename T, typename A>
__device__ inline void storeA(T *p, A v) {
    if constexpr (std::is_same_v<T, __half>) {
        *p = __float2half(float(v));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        *p = __float2bfloat16(float(v));
    } else {
        *p = T(v);
    }
}

template<typename A>
__device__ inline A activateA(A x, cudnnActivationMode_t mode, A coef) {
    switch (mode) {
        case CUDNN_ACTIVATION_SIGMOID: return A(1) / (A(1) + exp(-x));
        case CUDNN_ACTIVATION_RELU: return x > A(0) ? x : A(0);
        case CUDNN_ACTIVATION_TANH: return tanh(x);
        case CUDNN_ACTIVATION_CLIPPED_RELU: return min(max(x, A(0)), coef);
        case CUDNN_ACTIVATION_ELU: return x > A(0) ? x : coef * (exp(x) - A(1));
        case CUDNN_ACTIVATION_SWISH: return x / (A(1) + exp(-coef * x));
        default: return x;
    }
}

template<typename A>
__device__ inline A activationGradA(A y, A x, cudnnActivationMode_t mode, A coef) {
    switch (mode) {
        case CUDNN_ACTIVATION_SIGMOID: return y * (A(1) - y);
        case CUDNN_ACTIVATION_RELU: return y > A(0) ? A(1) : A(0);
        case CUDNN_ACTIVATION_TANH: return A(1) - y * y;
        case CUDNN_ACTIVATION_CLIPPED_RELU: return y > A(0) && y < coef ? A(1) : A(0);
        case CUDNN_ACTIVATION_ELU: return x > A(0) ? A(1) : y + coef;
        case CUDNN_ACTIVATION_SWISH: {
            const A s = A(1) / (A(1) + exp(-coef * x));
            return s + coef * x * s * (A(1) - s);
        }
        default: return A(1);
    }
}

// a Welford triple: count, mean, sum of squared deviations
template<typename A>
struct Moments {
    A count, mean, m2;
};

template<typename A>
__device__ inline Moments<A> merge(Moments<A> a, Moments<A> b) {
    const A count = a.count + b.count;
    if (count == A(0)) return a;
    const A delta = b.mean - a.mean, wb = b.count / count;
    return {count, a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.count * wb};
}

template<typename A>
__device__ Moments<A> blockMerge(Moments<A> m) {
    __shared__ A shared[3][32];
    for (int offset = 16; offset > 0; offset /= 2) {
        const Moments<A> other{__shfl_down_sync(0xffffffffu, m.count, offset),
                               __shfl_down_sync(0xffffffffu, m.mean, offset),
                               __shfl_down_sync(0xffffffffu, m.m2, offset)};
        m = merge(m, other);
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        shared[0][warp] = m.count;
        shared[1][warp] = m.mean;
        shared[2][warp] = m.m2;
    }
    __syncthreads();
    if (warp == 0) {
        const int warps = blockDim.x / 32;
        m = lane < warps ? Moments<A>{shared[0][lane], shared[1][lane], shared[2][lane]} : Moments<A>{0, 0, 0};
        for (int offset = 16; offset > 0; offset /= 2) {
            const Moments<A> other{__shfl_down_sync(0xffffffffu, m.count, offset),
                                   __shfl_down_sync(0xffffffffu, m.mean, offset),
                                   __shfl_down_sync(0xffffffffu, m.m2, offset)};
            m = merge(m, other);
        }
    }
    __syncthreads();
    return m;
}

template<typename A>
__device__ inline A blockSumA(A v) {
    __shared__ A shared[32];
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset);
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : A(0);
        for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    __syncthreads();
    return v;
}

// the images of group g
__device__ inline void groupImages(const BnFast &f, int g, long long &first, long long &last) {
    const long long per = (f.images + f.groups - 1) / f.groups;
    first = min(f.images, g * per);
    last = min(f.images, first + per);
}

// partial moments of channel blockIdx.x over image group blockIdx.y (partials[c * groups + g])
template<typename T, typename A>
__global__ void bnFastMoments(BnFast f, const T *x, Moments<A> *partials) {
    const long long c = blockIdx.x;
    long long first, last;
    groupImages(f, blockIdx.y, first, last);
    Moments<A> m{0, 0, 0};
    for (long long n = first; n < last; n++) {
        const T *plane = x + n * f.x.n + c * f.x.c;
        for (long long s = threadIdx.x; s < f.positions; s += blockDim.x) {
            const A v = loadA<T, A>(plane + s * f.x.s);
            m.count += A(1);
            const A delta = v - m.mean;
            m.mean += delta / m.count;
            m.m2 += delta * (v - m.mean);
        }
    }
    m = blockMerge(m);
    if (threadIdx.x == 0) partials[c * f.groups + blockIdx.y] = m;
}

// per channel: [0] mean, [1] inverse standard deviation (from the partials, or the saved values)
template<typename A>
__global__ void bnFastStatistics(BnFast f, const Moments<A> *partials, BatchNorm bn, cudnnDataType_t paramType,
                                 double factor, void *runningMean, void *runningVar, void *saveMean, void *saveInvStd,
                                 A *stats) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < f.channels; c += gridDim.x * blockDim.x) {
        double count = 0, mean = 0, m2 = 0;
        for (int g = 0; g < f.groups; g++) {
            const Moments<A> p = partials[c * f.groups + g];
            const double total = count + double(p.count);
            if (total == 0) continue;
            const double delta = double(p.mean) - mean;
            mean += delta * double(p.count) / total;
            m2 += double(p.m2) + delta * delta * count * double(p.count) / total;
            count = total;
        }
        const double var = count > 0 ? m2 / count : 0;
        const double inv = 1 / sqrt(var + bn.epsilon);
        if (runningMean != nullptr) {
            store(runningMean, c, paramType, (1 - factor) * load(runningMean, c, paramType) + factor * mean);
        }
        if (runningVar != nullptr) {
            const double unbiased = count > 1 ? var * count / (count - 1) : var;
            store(runningVar, c, paramType, (1 - factor) * load(runningVar, c, paramType) + factor * unbiased);
        }
        if (saveMean != nullptr) store(saveMean, c, paramType, mean);
        if (saveInvStd != nullptr) store(saveInvStd, c, paramType, inv);
        stats[2 * c] = A(mean);
        stats[2 * c + 1] = A(inv);
    }
}

// planes (image, channel) in blockIdx.x (and gridDim.x * blockIdx.y), positions in blocks of the threads
__device__ inline bool planeOf(const BnFast &f, long long &n, long long &c) {
    const long long plane = blockIdx.x + static_cast<long long>(gridDim.x) * blockIdx.y;
    if (plane >= f.images * f.channels) return false;
    n = plane / f.channels;
    c = plane % f.channels;
    return true;
}

template<typename T, typename A>
__global__ void bnFastApply(BnFast f, BatchNorm bn, A alpha, A beta, const T *x, const T *z, T *y, const A *stats,
                            cudnnDataType_t paramType, const void *scale, const void *bias) {
    long long n, c;
    if (!planeOf(f, n, c)) return;
    const A mean = stats[2 * c], inv = stats[2 * c + 1];
    const A k = inv * A(load(scale, c, paramType)), b = A(load(bias, c, paramType)) - mean * k;
    const A coef = A(bn.activationCoef);
    const T *xp = x + n * f.x.n + c * f.x.c;
    T *yp = y + n * f.y.n + c * f.y.c;
    const T *zp = z != nullptr ? z + n * f.z.n + c * f.z.c : nullptr;
    for (long long s = threadIdx.x; s < f.positions; s += blockDim.x) {
        A v = loadA<T, A>(xp + s * f.x.s) * k + b;
        if (zp != nullptr) v += loadA<T, A>(zp + s * f.z.s);
        if (bn.ops >= 1) v = activateA(v, bn.activation, coef);
        T *out = yp + s * f.y.s;
        A result = alpha * v;
        if (beta != A(0)) result += beta * loadA<T, A>(out);
        storeA(out, result);
    }
}

// the gradient at the normalization's output: dy through the activation (y or recomputed)
template<typename T, typename A>
__device__ inline A bnGradient(const BatchNorm &bn, A dy, A xhat, A g, A b, const T *y, A coef) {
    if (bn.ops < 1) return dy;
    const A pre = xhat * g + b;
    const A out = y != nullptr ? loadA<T, A>(y) : activateA(pre, bn.activation, coef);
    return dy * activationGradA(out, pre, bn.activation, coef);
}

// per (channel, image group): sums of the gradient and of gradient * xhat (partials[2 * (c * groups + g)]); the
// addend's gradient
template<typename T, typename A>
__global__ void bnFastGradientSums(BnFast f, BatchNorm bn, const T *x, const T *y, const T *dy, T *dz,
                                   const A *stats, cudnnDataType_t paramType, const void *scale, const void *bias,
                                   A *partials) {
    const long long c = blockIdx.x;
    long long first, last;
    groupImages(f, blockIdx.y, first, last);
    const A mean = stats[2 * c], inv = stats[2 * c + 1];
    const A g = A(load(scale, c, paramType)), b = bias != nullptr ? A(load(bias, c, paramType)) : A(0);
    const A coef = A(bn.activationCoef);
    A sumDy = 0, sumDyXhat = 0;
    for (long long n = first; n < last; n++) {
        const T *xp = x + n * f.x.n + c * f.x.c, *dyp = dy + n * f.dy.n + c * f.dy.c;
        const T *yp = y != nullptr ? y + n * f.y.n + c * f.y.c : nullptr;
        T *dzp = dz != nullptr ? dz + n * f.dz.n + c * f.dz.c : nullptr;
        for (long long s = threadIdx.x; s < f.positions; s += blockDim.x) {
            const A xhat = (loadA<T, A>(xp + s * f.x.s) - mean) * inv;
            const A grad = bnGradient<T, A>(bn, loadA<T, A>(dyp + s * f.dy.s), xhat, g, b,
                                            yp != nullptr ? yp + s * f.y.s : nullptr, coef);
            sumDy += grad;
            sumDyXhat += grad * xhat;
            if (dzp != nullptr) storeA(dzp + s * f.dz.s, grad);
        }
    }
    sumDy = blockSumA(sumDy);
    sumDyXhat = blockSumA(sumDyXhat);
    if (threadIdx.x == 0) {
        partials[2 * (c * f.groups + blockIdx.y)] = sumDy;
        partials[2 * (c * f.groups + blockIdx.y) + 1] = sumDyXhat;
    }
}

// per channel: the parameter gradients; stats[2 * channels + 2c] = mean of the gradient, [.. + 1] mean of grad * xhat
template<typename A>
__global__ void bnFastParamGradients(BnFast f, const A *partials, double alphaParam, double betaParam,
                                     cudnnDataType_t paramType, void *dScale, void *dBias, A *stats) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < f.channels; c += gridDim.x * blockDim.x) {
        double sumDy = 0, sumDyXhat = 0;
        for (int g = 0; g < f.groups; g++) {
            sumDy += double(partials[2 * (c * f.groups + g)]);
            sumDyXhat += double(partials[2 * (c * f.groups + g) + 1]);
        }
        if (dScale != nullptr) {
            double v = alphaParam * sumDyXhat;
            if (betaParam != 0) v += betaParam * load(dScale, c, paramType);
            store(dScale, c, paramType, v);
        }
        if (dBias != nullptr) {
            double v = alphaParam * sumDy;
            if (betaParam != 0) v += betaParam * load(dBias, c, paramType);
            store(dBias, c, paramType, v);
        }
        const double count = double(f.images * f.positions);
        stats[2 * f.channels + 2 * c] = A(sumDy / count);
        stats[2 * f.channels + 2 * c + 1] = A(sumDyXhat / count);
    }
}

template<typename T, typename A>
__global__ void bnFastInputGradient(BnFast f, BatchNorm bn, A alpha, A beta, const T *x, const T *y, const T *dy,
                                    T *dx, const A *stats, cudnnDataType_t paramType, const void *scale,
                                    const void *bias) {
    long long n, c;
    if (!planeOf(f, n, c)) return;
    const A mean = stats[2 * c], inv = stats[2 * c + 1];
    const A meanDy = stats[2 * f.channels + 2 * c], meanDyXhat = stats[2 * f.channels + 2 * c + 1];
    const A g = A(load(scale, c, paramType)), b = bias != nullptr ? A(load(bias, c, paramType)) : A(0);
    const A coef = A(bn.activationCoef), k = g * inv;
    const T *xp = x + n * f.x.n + c * f.x.c, *dyp = dy + n * f.dy.n + c * f.dy.c;
    const T *yp = y != nullptr ? y + n * f.y.n + c * f.y.c : nullptr;
    T *dxp = dx + n * f.dx.n + c * f.dx.c;
    for (long long s = threadIdx.x; s < f.positions; s += blockDim.x) {
        const A xhat = (loadA<T, A>(xp + s * f.x.s) - mean) * inv;
        const A grad = bnGradient<T, A>(bn, loadA<T, A>(dyp + s * f.dy.s), xhat, g, b,
                                        yp != nullptr ? yp + s * f.y.s : nullptr, coef);
        T *out = dxp + s * f.dx.s;
        A result = alpha * k * (grad - meanDy - xhat * meanDyXhat);
        if (beta != A(0)) result += beta * loadA<T, A>(out);
        storeA(out, result);
    }
}

// ---- channels-last tensors: tiles of 32 channels x rows (coalesced), elementwise passes in memory order

constexpr int LAST_CHANNELS = 32, LAST_ROWS = 8;

// the rows (image, position) of group g
__device__ inline void groupRows(const BnFast &f, int g, long long &first, long long &last) {
    const long long rows = f.images * f.positions, per = (rows + f.groups - 1) / f.groups;
    first = min(rows, g * per);
    last = min(rows, first + per);
}

// partial moments of channels blockIdx.x * 32 .. over row group blockIdx.y (partials[c * groups + g])
template<typename T, typename A>
__global__ void bnLastMoments(BnFast f, const T *x, Moments<A> *partials) {
    __shared__ A shared[3][LAST_ROWS][LAST_CHANNELS + 1];
    const long long c = blockIdx.x * LAST_CHANNELS + threadIdx.x;
    long long first, last;
    groupRows(f, blockIdx.y, first, last);
    Moments<A> m{0, 0, 0};
    if (c < f.channels) {
        for (long long r = first + threadIdx.y; r < last; r += LAST_ROWS) {
            const A v = loadA<T, A>(x + r * f.channels + c);
            m.count += A(1);
            const A delta = v - m.mean;
            m.mean += delta / m.count;
            m.m2 += delta * (v - m.mean);
        }
    }
    shared[0][threadIdx.y][threadIdx.x] = m.count;
    shared[1][threadIdx.y][threadIdx.x] = m.mean;
    shared[2][threadIdx.y][threadIdx.x] = m.m2;
    __syncthreads();
    if (threadIdx.y == 0 && c < f.channels) {
        for (int row = 1; row < LAST_ROWS; row++) {
            m = merge(m, Moments<A>{shared[0][row][threadIdx.x], shared[1][row][threadIdx.x],
                                    shared[2][row][threadIdx.x]});
        }
        partials[c * f.groups + blockIdx.y] = m;
    }
}

// per channel: [0] scale, [1] shift of the normalization (scale * (x - mean) * inv + bias)
template<typename A>
__global__ void bnCoefficients(long long channels, const A *stats, cudnnDataType_t paramType, const void *scale,
                               const void *bias, A *coefficients) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        const A k = stats[2 * c + 1] * A(load(scale, c, paramType));
        coefficients[2 * c] = k;
        coefficients[2 * c + 1] = (bias != nullptr ? A(load(bias, c, paramType)) : A(0)) - stats[2 * c] * k;
    }
}

// (I: the index type, 32 bit where the tensor allows: a 64 bit remainder per element is a call of ~70 instructions)
template<typename T, typename A, typename I>
__global__ void bnLastApply(BnFast f, BatchNorm bn, A alpha, A beta, const T *x, const T *z, T *y,
                            const A *coefficients) {
    const I total = I(f.images * f.positions * f.channels), channels = I(f.channels);
    const A coef = A(bn.activationCoef);
    for (I e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        const I c = e % channels;
        A v = loadA<T, A>(x + e) * coefficients[2 * c] + coefficients[2 * c + 1];
        if (z != nullptr) v += loadA<T, A>(z + e);
        if (bn.ops >= 1) v = activateA(v, bn.activation, coef);
        A result = alpha * v;
        if (beta != A(0)) result += beta * loadA<T, A>(y + e);
        storeA(y + e, result);
    }
}

// sums of the gradient and of gradient * xhat per (channel, row group) as bnFastGradientSums
template<typename T, typename A>
__global__ void bnLastGradientSums(BnFast f, BatchNorm bn, const T *x, const T *y, const T *dy, T *dz,
                                   const A *stats, const A *params, A *partials) {
    __shared__ A shared[2][LAST_ROWS][LAST_CHANNELS + 1];
    const long long c = blockIdx.x * LAST_CHANNELS + threadIdx.x;
    long long first, last;
    groupRows(f, blockIdx.y, first, last);
    A sumDy = 0, sumDyXhat = 0;
    if (c < f.channels) {
        const A mean = stats[2 * c], inv = stats[2 * c + 1], g = params[2 * c], b = params[2 * c + 1];
        const A coef = A(bn.activationCoef);
        for (long long r = first + threadIdx.y; r < last; r += LAST_ROWS) {
            const long long e = r * f.channels + c;
            const A xhat = (loadA<T, A>(x + e) - mean) * inv;
            const A grad = bnGradient<T, A>(bn, loadA<T, A>(dy + e), xhat, g, b, y != nullptr ? y + e : nullptr, coef);
            sumDy += grad;
            sumDyXhat += grad * xhat;
            if (dz != nullptr) storeA(dz + e, grad);
        }
    }
    shared[0][threadIdx.y][threadIdx.x] = sumDy;
    shared[1][threadIdx.y][threadIdx.x] = sumDyXhat;
    __syncthreads();
    if (threadIdx.y == 0 && c < f.channels) {
        for (int row = 1; row < LAST_ROWS; row++) {
            sumDy += shared[0][row][threadIdx.x];
            sumDyXhat += shared[1][row][threadIdx.x];
        }
        partials[2 * (c * f.groups + blockIdx.y)] = sumDy;
        partials[2 * (c * f.groups + blockIdx.y) + 1] = sumDyXhat;
    }
}

// per channel: [0] scale, [1] bias (0 without)
template<typename A>
__global__ void bnParams(long long channels, cudnnDataType_t paramType, const void *scale, const void *bias,
                         A *params) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        params[2 * c] = A(load(scale, c, paramType));
        params[2 * c + 1] = bias != nullptr ? A(load(bias, c, paramType)) : A(0);
    }
}

template<typename T, typename A, typename I>
__global__ void bnLastInputGradient(BnFast f, BatchNorm bn, A alpha, A beta, const T *x, const T *y, const T *dy,
                                    T *dx, const A *stats, const A *params) {
    const I total = I(f.images * f.positions * f.channels), channels = I(f.channels);
    const A coef = A(bn.activationCoef);
    for (I e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        const I c = e % channels;
        const A mean = stats[2 * c], inv = stats[2 * c + 1];
        const A meanDy = stats[2 * f.channels + 2 * c], meanDyXhat = stats[2 * f.channels + 2 * c + 1];
        const A g = params[2 * c], b = params[2 * c + 1];
        const A xhat = (loadA<T, A>(x + e) - mean) * inv;
        const A grad = bnGradient<T, A>(bn, loadA<T, A>(dy + e), xhat, g, b, y != nullptr ? y + e : nullptr, coef);
        A result = alpha * g * inv * (grad - meanDy - xhat * meanDyXhat);
        if (beta != A(0)) result += beta * loadA<T, A>(dx + e);
        storeA(dx + e, result);
    }
}

bool smallIndex(const BnFast &f) {
    return f.images * f.positions * f.channels < (1ll << 32) - (1ll << 30);
}

unsigned int elementBlocks(const BnFast &f) {
    return unsigned(std::max<long long>(1, std::min<long long>((f.images * f.positions * f.channels + 255) / 256,
                                                              1 << 20)));
}

dim3 lastGrid(const BnFast &f) {
    return dim3(unsigned((f.channels + LAST_CHANNELS - 1) / LAST_CHANNELS), unsigned(f.groups));
}

// whether the fast path takes the problem: spatial mode, one data type, single-stride positions
bool bnFastOf(const BatchNorm &bn, const Tensor &x, std::initializer_list<std::pair<const Tensor *, Planes *>> others,
              BnFast &f) {
    if (bn.mode == CUDNN_BATCHNORM_PER_ACTIVATION || x.nbDims < 3 || !planesOf(x, f.x)) return false;
    for (auto [t, planes]: others) {
        if (t == nullptr) continue;
        if (t->type != x.type || t->nbDims != x.nbDims || !planesOf(*t, *planes)) return false;
        for (int d = 0; d < x.nbDims; d++) {
            if (t->dims[d] != x.dims[d]) return false;
        }
    }
    f.images = x.dims[0];
    f.channels = x.dims[1];
    f.positions = 1;
    for (int d = 2; d < x.nbDims; d++) f.positions *= x.dims[d];
    if (f.channels > 65535 || f.images * f.positions == 0) return false;
    int device = 0, sms = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    auto channelsLast = [&](const Planes &p) {
        return p.c == 1 && p.s == f.channels && p.n == f.channels * f.positions;
    };
    f.last = f.channels > 1 && channelsLast(f.x);
    for (auto [t, planes]: others) {
        if (t != nullptr) f.last = f.last && channelsLast(*planes);
    }
    if (f.last) {
        // about 4 blocks per SM, at least 64 rows per group
        const long long tiles = (f.channels + LAST_CHANNELS - 1) / LAST_CHANNELS, rows = f.images * f.positions;
        f.groups = int(std::clamp<long long>(4LL * sms / tiles, 1, std::min<long long>((rows + 63) / 64, 65535)));
    } else {
        // about 4 blocks per SM, at least one image per group
        f.groups = int(std::clamp<long long>(4LL * sms / f.channels, 1, std::min<long long>(f.images, 65535)));
    }
    return true;
}

dim3 planeGrid(const BnFast &f) {
    const long long planes = f.images * f.channels;
    const long long x = std::min<long long>(planes, 1 << 30);
    return dim3(unsigned(x), unsigned((planes + x - 1) / x));
}

unsigned int planeThreads(const BnFast &f) {
    return f.positions >= 256 ? 256 : f.positions > 64 ? 128 : 64;
}

template<typename T, typename A>
cudnnStatus_t bnFastForward(cudaStream_t stream, const BnFast &f, const BatchNorm &bn, double alpha, double beta,
                            const void *xData, const void *zData, void *yData, cudnnDataType_t paramType,
                            const void *scale, const void *bias, double factor, void *runningMean, void *runningVar,
                            void *saveMean, void *saveInvStd) {
    const size_t partialBytes = sizeof(Moments<A>) * f.channels * f.groups;
    char *scratch =
        static_cast<char *>(librecuda_blas_scratch(stream, partialBytes + 256 + sizeof(A) * 4 * f.channels));
    if (scratch == nullptr) return CUDNN_STATUS_ALLOC_FAILED;
    auto *partials = reinterpret_cast<Moments<A> *>(scratch);
    A *stats = reinterpret_cast<A *>(scratch + (partialBytes + 255) / 256 * 256);
    if (f.last) {
        bnLastMoments<T, A><<<lastGrid(f), dim3(LAST_CHANNELS, LAST_ROWS), 0, stream>>>(
            f, static_cast<const T *>(xData), partials);
    } else {
        bnFastMoments<T, A><<<dim3(unsigned(f.channels), unsigned(f.groups)), 256, 0, stream>>>(
            f, static_cast<const T *>(xData), partials);
    }
    bnFastStatistics<A><<<unsigned((f.channels + 127) / 128), 128, 0, stream>>>(
        f, partials, bn, paramType, factor, runningMean, runningVar, saveMean, saveInvStd, stats);
    if (f.last) {
        A *coefficients = stats + 2 * f.channels;
        bnCoefficients<A><<<unsigned((f.channels + 127) / 128), 128, 0, stream>>>(f.channels, stats, paramType,
                                                                                  scale, bias, coefficients);
        const auto apply = smallIndex(f) ? bnLastApply<T, A, unsigned> : bnLastApply<T, A, long long>;
        apply<<<elementBlocks(f), 256, 0, stream>>>(
            f, bn, A(alpha), A(beta), static_cast<const T *>(xData),
            bn.ops == 2 ? static_cast<const T *>(zData) : nullptr, static_cast<T *>(yData), coefficients);
        return launched();
    }
    bnFastApply<T, A><<<planeGrid(f), planeThreads(f), 0, stream>>>(
        f, bn, A(alpha), A(beta), static_cast<const T *>(xData), bn.ops == 2 ? static_cast<const T *>(zData) : nullptr,
        static_cast<T *>(yData), stats, paramType, scale, bias);
    return launched();
}

// saved statistics (paramType) -> stats
template<typename A>
__global__ void bnFastSaved(long long channels, cudnnDataType_t paramType, const void *mean, const void *invStd,
                            A *stats) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        stats[2 * c] = A(load(mean, c, paramType));
        stats[2 * c + 1] = A(load(invStd, c, paramType));
    }
}

template<typename T, typename A>
cudnnStatus_t bnFastBackward(cudaStream_t stream, const BnFast &f, const BatchNorm &bn, double alphaData,
                             double betaData, double alphaParam, double betaParam, const void *xData,
                             const void *yData, const void *dyData, void *dzData, void *dxData,
                             cudnnDataType_t paramType, const void *scale, const void *bias, void *dScale, void *dBias,
                             const void *savedMean, const void *savedInvStd) {
    const size_t partialBytes = (std::max(sizeof(Moments<A>), 2 * sizeof(A)) * f.channels * f.groups + 255) / 256 * 256;
    char *scratch = static_cast<char *>(librecuda_blas_scratch(stream, partialBytes + sizeof(A) * 6 * f.channels));
    if (scratch == nullptr) return CUDNN_STATUS_ALLOC_FAILED;
    A *stats = reinterpret_cast<A *>(scratch + partialBytes);
    const unsigned channelBlocks = unsigned((f.channels + 127) / 128);
    if (savedMean != nullptr && savedInvStd != nullptr) {
        bnFastSaved<A><<<channelBlocks, 128, 0, stream>>>(f.channels, paramType, savedMean, savedInvStd, stats);
    } else {
        auto *moments = reinterpret_cast<Moments<A> *>(scratch);
        if (f.last) {
            bnLastMoments<T, A><<<lastGrid(f), dim3(LAST_CHANNELS, LAST_ROWS), 0, stream>>>(
                f, static_cast<const T *>(xData), moments);
        } else {
            bnFastMoments<T, A><<<dim3(unsigned(f.channels), unsigned(f.groups)), 256, 0, stream>>>(
                f, static_cast<const T *>(xData), moments);
        }
        bnFastStatistics<A><<<channelBlocks, 128, 0, stream>>>(f, moments, bn, paramType, 0, nullptr, nullptr,
                                                               nullptr, nullptr, stats);
    }
    auto *sums = reinterpret_cast<A *>(scratch);
    const T *y = bn.ops >= 1 ? static_cast<const T *>(yData) : nullptr;
    if (f.last) {
        A *params = stats + 4 * f.channels;
        bnParams<A><<<channelBlocks, 128, 0, stream>>>(f.channels, paramType, scale, bias, params);
        bnLastGradientSums<T, A><<<lastGrid(f), dim3(LAST_CHANNELS, LAST_ROWS), 0, stream>>>(
            f, bn, static_cast<const T *>(xData), y, static_cast<const T *>(dyData),
            bn.ops == 2 ? static_cast<T *>(dzData) : nullptr, stats, params, sums);
        bnFastParamGradients<A><<<channelBlocks, 128, 0, stream>>>(f, sums, alphaParam, betaParam, paramType, dScale,
                                                                   dBias, stats);
        const auto inputGradient = smallIndex(f) ? bnLastInputGradient<T, A, unsigned>
                                                  : bnLastInputGradient<T, A, long long>;
        inputGradient<<<elementBlocks(f), 256, 0, stream>>>(
            f, bn, A(alphaData), A(betaData), static_cast<const T *>(xData), y, static_cast<const T *>(dyData),
            static_cast<T *>(dxData), stats, params);
        return launched();
    }
    bnFastGradientSums<T, A><<<dim3(unsigned(f.channels), unsigned(f.groups)), 256, 0, stream>>>(
        f, bn, static_cast<const T *>(xData), y, static_cast<const T *>(dyData),
        bn.ops == 2 ? static_cast<T *>(dzData) : nullptr, stats, paramType, scale, bias, sums);
    bnFastParamGradients<A><<<channelBlocks, 128, 0, stream>>>(f, sums, alphaParam, betaParam, paramType, dScale,
                                                               dBias, stats);
    bnFastInputGradient<T, A><<<planeGrid(f), planeThreads(f), 0, stream>>>(
        f, bn, A(alphaData), A(betaData), static_cast<const T *>(xData), y, static_cast<const T *>(dyData),
        static_cast<T *>(dxData), stats, paramType, scale, bias);
    return launched();
}

// running statistics -> stats (mean, inverse standard deviation)
template<typename A>
__global__ void bnFastRunning(long long channels, double epsilon, cudnnDataType_t paramType, const void *mean,
                              const void *var, A *stats) {
    for (long long c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        stats[2 * c] = A(load(mean, c, paramType));
        stats[2 * c + 1] = A(1 / sqrt(load(var, c, paramType) + epsilon));
    }
}

template<typename T, typename A>
cudnnStatus_t bnFastInference(cudaStream_t stream, const BnFast &f, const BatchNorm &bn, double alpha, double beta,
                              const void *xData, void *yData, cudnnDataType_t paramType, const void *scale,
                              const void *bias, const void *mean, const void *var) {
    A *stats = static_cast<A *>(librecuda_blas_scratch(stream, sizeof(A) * 4 * f.channels));
    if (stats == nullptr) return CUDNN_STATUS_ALLOC_FAILED;
    const unsigned channelBlocks = unsigned((f.channels + 127) / 128);
    bnFastRunning<A><<<channelBlocks, 128, 0, stream>>>(f.channels, bn.epsilon, paramType, mean, var, stats);
    // (inference applies no activation or addend)
    BatchNorm plain = bn;
    plain.ops = 0;
    if (f.last) {
        A *coefficients = stats + 2 * f.channels;
        bnCoefficients<A><<<channelBlocks, 128, 0, stream>>>(f.channels, stats, paramType, scale, bias, coefficients);
        const auto apply = smallIndex(f) ? bnLastApply<T, A, unsigned> : bnLastApply<T, A, long long>;
        apply<<<elementBlocks(f), 256, 0, stream>>>(f, plain, A(alpha), A(beta), static_cast<const T *>(xData),
                                                    nullptr, static_cast<T *>(yData), coefficients);
    } else {
        bnFastApply<T, A><<<planeGrid(f), planeThreads(f), 0, stream>>>(f, plain, A(alpha), A(beta),
                                                                        static_cast<const T *>(xData), nullptr,
                                                                        static_cast<T *>(yData), stats, paramType,
                                                                        scale, bias);
    }
    return launched();
}

// dispatch on the data type
#define BN_FAST_DISPATCH(type, call)                                                                                   \
    switch (type) {                                                                                                    \
        case CUDNN_DATA_HALF: { using T = __half; using A = float; return call; }                                     \
        case CUDNN_DATA_BFLOAT16: { using T = __nv_bfloat16; using A = float; return call; }                          \
        case CUDNN_DATA_FLOAT: { using T = float; using A = float; return call; }                                     \
        case CUDNN_DATA_DOUBLE: { using T = double; using A = double; return call; }                                  \
        default: break;                                                                                                \
    }


// ---- pooling

__global__ void poolingKernel(Pooling p, double alpha, Tensor x, const void *xData, double beta, Tensor y,
                              void *yData) {
    GRID_LOOP(e, elements(y)) {
        // y: N, C, H, W (2 spatial dimensions)
        const long long ow = e % y.dims[3], oh = (e / y.dims[3]) % y.dims[2];
        const long long c = (e / (y.dims[3] * y.dims[2])) % y.dims[1], n = e / (y.dims[3] * y.dims[2] * y.dims[1]);
        double acc = p.mode == CUDNN_POOLING_MAX || p.mode == CUDNN_POOLING_MAX_DETERMINISTIC ? -INFINITY : 0;
        int count = 0;
        for (int r = 0; r < p.window[0]; r++) {
            const long long ih = oh * p.stride[0] - p.pad[0] + r;
            for (int s = 0; s < p.window[1]; s++) {
                const long long iw = ow * p.stride[1] - p.pad[1] + s;
                if (ih < 0 || ih >= x.dims[2] || iw < 0 || iw >= x.dims[3]) continue;
                const double v = load(xData, n * x.strides[0] + c * x.strides[1] + ih * x.strides[2] +
                                             iw * x.strides[3], x.type);
                if (p.mode == CUDNN_POOLING_MAX || p.mode == CUDNN_POOLING_MAX_DETERMINISTIC) {
                    acc = fmax(acc, v);
                } else {
                    acc += v;
                }
                count++;
            }
        }
        if (p.mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
            acc /= p.window[0] * p.window[1];
        } else if (p.mode == CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
            acc = count > 0 ? acc / count : 0;
        }
        const long long out = offsetOf(y, e);
        double result = alpha * acc;
        if (beta != 0) result += beta * load(yData, out, y.type);
        store(yData, out, y.type, result);
    }
}

// ---- spatial transformer

__device__ inline double normalized(long long i, long long size) {
    return size > 1 ? -1 + 2.0 * i / (size - 1) : 0;
}

__global__ void gridForward(cudnnDataType_t type, int n, int h, int w, const void *theta, void *grid) {
    GRID_LOOP(e, static_cast<long long>(n) * h * w) {
        const long long b = e / (h * w), i = (e / w) % h, j = e % w;
        const double xt = normalized(j, w), yt = normalized(i, h);
        for (int k = 0; k < 2; k++) {
            const double v = load(theta, b * 6 + k * 3, type) * xt + load(theta, b * 6 + k * 3 + 1, type) * yt +
                             load(theta, b * 6 + k * 3 + 2, type);
            store(grid, e * 2 + k, type, v);
        }
    }
}

__global__ void gridBackward(cudnnDataType_t type, int n, int h, int w, const void *dgrid, void *dtheta) {
    __shared__ double shared[THREADS];
    // block b: dtheta[b] (2 x 3) = sum over positions of dgrid (2) * [xt, yt, 1]
    for (int b = blockIdx.x; b < n; b += gridDim.x) {
        double acc[6] = {};
        for (long long e = threadIdx.x; e < static_cast<long long>(h) * w; e += THREADS) {
            const double xt = normalized(e % w, w), yt = normalized(e / w, h);
            for (int k = 0; k < 2; k++) {
                const double g = load(dgrid, (b * static_cast<long long>(h) * w + e) * 2 + k, type);
                acc[k * 3] += g * xt;
                acc[k * 3 + 1] += g * yt;
                acc[k * 3 + 2] += g;
            }
        }
        for (int i = 0; i < 6; i++) {
            const double total = blockSum(acc[i], shared);
            if (threadIdx.x == 0) store(dtheta, b * 6 + i, type, total);
        }
    }
}

// bilinear sample weights of grid point (gx, gy) in an h x w image (corners aligned)
struct Bilinear {
    long long x0, y0;
    double fx, fy;
};

// (coordinates in the data's precision: on pixel boundaries, which neighbours count depends on rounding)
__device__ inline Bilinear bilinear(double gx, double gy, long long h, long long w, bool doubles) {
    double ix, iy;
    if (doubles) {
        ix = (gx + 1) * (w - 1) / 2;
        iy = (gy + 1) * (h - 1) / 2;
    } else {
        ix = (float(gx) + 1.0f) * float(w - 1) / 2.0f;
        iy = (float(gy) + 1.0f) * float(h - 1) / 2.0f;
    }
    Bilinear b{};
    b.x0 = static_cast<long long>(floor(ix));
    b.y0 = static_cast<long long>(floor(iy));
    b.fx = ix - b.x0;
    b.fy = iy - b.y0;
    return b;
}

__global__ void samplerForwardKernel(double alpha, Tensor x, const void *xData, const void *grid, double beta,
                                     Tensor y, void *yData) {
    GRID_LOOP(e, elements(y)) {
        const long long ow = e % y.dims[3], oh = (e / y.dims[3]) % y.dims[2];
        const long long c = (e / (y.dims[3] * y.dims[2])) % y.dims[1], n = e / (y.dims[3] * y.dims[2] * y.dims[1]);
        const long long g = ((n * y.dims[2] + oh) * y.dims[3] + ow) * 2;
        const Bilinear b = bilinear(load(grid, g, y.type), load(grid, g + 1, y.type), x.dims[2], x.dims[3],
                                    y.type == CUDNN_DATA_DOUBLE);
        double v = 0;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                const long long yy = b.y0 + dy, xx = b.x0 + dx;
                if (yy < 0 || yy >= x.dims[2] || xx < 0 || xx >= x.dims[3]) continue;
                const double weight = (dx ? b.fx : 1 - b.fx) * (dy ? b.fy : 1 - b.fy);
                v += weight * load(xData, n * x.strides[0] + c * x.strides[1] + yy * x.strides[2] + xx * x.strides[3],
                                   x.type);
            }
        }
        const long long out = offsetOf(y, e);
        double result = alpha * v;
        if (beta != 0) result += beta * load(yData, out, y.type);
        store(yData, out, y.type, result);
    }
}

__global__ void scaleKernel(Tensor t, void *data, double factor) {
    GRID_LOOP(e, elements(t)) {
        const long long offset = offsetOf(t, e);
        store(data, offset, t.type, factor == 0 ? 0 : factor * load(data, offset, t.type));
    }
}

__device__ inline void atomicAccumulate(void *base, long long index, cudnnDataType_t type, double value) {
    if (type == CUDNN_DATA_DOUBLE) {
        atomicAdd(static_cast<double *>(base) + index, value);
    } else {
        atomicAdd(static_cast<float *>(base) + index, float(value));
    }
}

// one thread per output position (n, h, w): dx contributions of all channels, and dgrid
__global__ void samplerBackwardKernel(double alpha, Tensor x, const void *xData, Tensor dx, void *dxData,
                                      double alphaGrid, Tensor dy, const void *dyData, const void *grid,
                                      double betaGrid, void *dgrid) {
    GRID_LOOP(e, dy.dims[0] * dy.dims[2] * dy.dims[3]) {
        const long long ow = e % dy.dims[3], oh = (e / dy.dims[3]) % dy.dims[2], n = e / (dy.dims[3] * dy.dims[2]);
        const long long g = e * 2;
        const long long h = x.dims[2], w = x.dims[3];
        const Bilinear b = bilinear(load(grid, g, dy.type), load(grid, g + 1, dy.type), h, w,
                                    dy.type == CUDNN_DATA_DOUBLE);
        double gx = 0, gy = 0;
        for (long long c = 0; c < x.dims[1]; c++) {
            const double grad = load(dyData, n * dy.strides[0] + c * dy.strides[1] + oh * dy.strides[2] +
                                             ow * dy.strides[3], dy.type);
            for (int sy = 0; sy < 2; sy++) {
                for (int sx = 0; sx < 2; sx++) {
                    const long long yy = b.y0 + sy, xx = b.x0 + sx;
                    if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                    const double wx = sx ? b.fx : 1 - b.fx, wy = sy ? b.fy : 1 - b.fy;
                    atomicAccumulate(dxData, n * dx.strides[0] + c * dx.strides[1] + yy * dx.strides[2] +
                                             xx * dx.strides[3], dx.type, alpha * grad * wx * wy);
                    const double v = load(xData, n * x.strides[0] + c * x.strides[1] + yy * x.strides[2] +
                                                 xx * x.strides[3], x.type);
                    gx += grad * v * (sx ? 1 : -1) * wy;
                    gy += grad * v * (sy ? 1 : -1) * wx;
                }
            }
        }
        // d(ix)/d(gx) = (w - 1) / 2
        const double values[2] = {gx * (w - 1) / 2, gy * (h - 1) / 2};
        for (int k = 0; k < 2; k++) {
            double result = alphaGrid * values[k];
            if (betaGrid != 0) result += betaGrid * load(dgrid, g + k, dy.type);
            store(dgrid, g + k, dy.type, result);
        }
    }
}

} // namespace

cudnnStatus_t biasActivation(cudaStream_t stream, const Tensor &y, void *yData, const void *bias,
                             cudnnActivationMode_t mode, double coef) {
    if (bias == nullptr && mode == CUDNN_ACTIVATION_IDENTITY) {
        return CUDNN_STATUS_SUCCESS;
    }
    biasActivationKernel<<<gridFor(elements(y)), 256, 0, stream>>>(y, yData, bias, mode, coef);
    return launched();
}

cudnnStatus_t copyTensor(cudaStream_t stream, const Tensor &src, const void *srcData, const Tensor &dst,
                         void *dstData) {
    if (elements(src) != elements(dst)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    copyKernel<<<gridFor(elements(dst)), 256, 0, stream>>>(src, srcData, dst, dstData);
    return launched();
}

namespace {

bool perActivationMode(cudnnBatchNormMode_t mode) {
    return mode == CUDNN_BATCHNORM_PER_ACTIVATION;
}

unsigned int bnBlocks(const BnLayout &l) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>(l.features, 65535)));
}

} // namespace

cudnnStatus_t batchNormForwardTraining(cudaStream_t stream, const BatchNorm &bn, double alpha, double beta,
                                       const Tensor &x, const void *xData, const Tensor &z, const void *zData,
                                       const Tensor &y, void *yData, const Tensor &param, const void *scale,
                                       const void *bias, double factor, void *runningMean, void *runningVar,
                                       void *saveMean, void *saveInvStd) {
    if (scale == nullptr || bias == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    BnFast f{};
    if (bnFastOf(bn, x, {{&y, &f.y}, {bn.ops == 2 && zData != nullptr ? &z : nullptr, &f.z}}, f)) {
        BN_FAST_DISPATCH(x.type, (bnFastForward<T, A>(stream, f, bn, alpha, beta, xData, zData, yData, param.type,
                                                      scale, bias, factor, runningMean, runningVar, saveMean,
                                                      saveInvStd)))
    }
    const bool per_activation = perActivationMode(bn.mode);
    const BnLayout l = bnLayout(x, per_activation);
    bnForwardTraining<<<bnBlocks(l), THREADS, 0, stream>>>(bn, l, per_activation, alpha, beta, x, xData, z, zData, y,
                                                           yData, param.type, scale, bias, factor, runningMean,
                                                           runningVar, saveMean, saveInvStd);
    return launched();
}

cudnnStatus_t batchNormForwardInference(cudaStream_t stream, const BatchNorm &bn, double alpha, double beta,
                                        const Tensor &x, const void *xData, const Tensor &y, void *yData,
                                        const Tensor &param, const void *scale, const void *bias, const void *mean,
                                        const void *var) {
    if (scale == nullptr || bias == nullptr || mean == nullptr || var == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    BnFast f{};
    if (bnFastOf(bn, x, {{&y, &f.y}}, f)) {
        BN_FAST_DISPATCH(x.type, (bnFastInference<T, A>(stream, f, bn, alpha, beta, xData, yData, param.type, scale,
                                                        bias, mean, var)))
    }
    const bool per_activation = perActivationMode(bn.mode);
    const BnLayout l = bnLayout(x, per_activation);
    bnForwardInference<<<gridFor(l.features * l.count), 256, 0, stream>>>(bn, l, per_activation, alpha, beta, x,
                                                                          xData, y, yData, param.type, scale, bias,
                                                                          mean, var);
    return launched();
}

cudnnStatus_t batchNormBackward(cudaStream_t stream, const BatchNorm &bn, double alphaData, double betaData,
                                double alphaParam, double betaParam, const Tensor &x, const void *xData,
                                const Tensor &y, const void *yData, const Tensor &dy, const void *dyData,
                                const Tensor &dz, void *dzData, const Tensor &dx, void *dxData, const Tensor &param,
                                const void *scale, const void *bias, void *dScale, void *dBias, const void *savedMean,
                                const void *savedInvStd) {
    if (scale == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    BnFast f{};
    if (bnFastOf(bn, x, {{&dy, &f.dy}, {&dx, &f.dx}, {bn.ops >= 1 && yData != nullptr ? &y : nullptr, &f.y},
                         {bn.ops == 2 && dzData != nullptr ? &dz : nullptr, &f.dz}}, f)) {
        BN_FAST_DISPATCH(x.type, (bnFastBackward<T, A>(stream, f, bn, alphaData, betaData, alphaParam, betaParam,
                                                       xData, yData, dyData, dzData, dxData, param.type, scale, bias,
                                                       dScale, dBias, savedMean, savedInvStd)))
    }
    const bool per_activation = perActivationMode(bn.mode);
    const BnLayout l = bnLayout(x, per_activation);
    bnBackward<<<bnBlocks(l), THREADS, 0, stream>>>(bn, l, per_activation, alphaData, betaData, alphaParam, betaParam,
                                                    x, xData, y, yData, dy, dyData, dz, dzData, dx, dxData,
                                                    param.type, scale, bias, dScale, dBias, savedMean, savedInvStd);
    return launched();
}

cudnnStatus_t poolingForward(cudaStream_t stream, const Pooling &pool, double alpha, const Tensor &x,
                             const void *xData, double beta, const Tensor &y, void *yData) {
    if (x.nbDims != 4 || y.nbDims != 4 || pool.spatial != 2) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    poolingKernel<<<gridFor(elements(y)), 256, 0, stream>>>(pool, alpha, x, xData, beta, y, yData);
    return launched();
}

cudnnStatus_t gridGeneratorForward(cudaStream_t stream, cudnnDataType_t type, int n, int h, int w, const void *theta,
                                   void *grid) {
    gridForward<<<gridFor(static_cast<long long>(n) * h * w), 256, 0, stream>>>(type, n, h, w, theta, grid);
    return launched();
}

cudnnStatus_t gridGeneratorBackward(cudaStream_t stream, cudnnDataType_t type, int n, int h, int w, const void *dgrid,
                                    void *dtheta) {
    gridBackward<<<std::min(n, 65535), THREADS, 0, stream>>>(type, n, h, w, dgrid, dtheta);
    return launched();
}

cudnnStatus_t samplerForward(cudaStream_t stream, double alpha, const Tensor &x, const void *xData, const void *grid,
                             double beta, const Tensor &y, void *yData) {
    if (x.nbDims != 4 || y.nbDims != 4) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    samplerForwardKernel<<<gridFor(elements(y)), 256, 0, stream>>>(alpha, x, xData, grid, beta, y, yData);
    return launched();
}

cudnnStatus_t samplerBackward(cudaStream_t stream, double alpha, const Tensor &x, const void *xData, double beta,
                              const Tensor &dx, void *dxData, double alphaGrid, const Tensor &dy, const void *dyData,
                              const void *grid, double betaGrid, void *dgrid) {
    // (dx is accumulated atomically: float or double)
    if (x.nbDims != 4 || dy.nbDims != 4 || (dx.type != CUDNN_DATA_FLOAT && dx.type != CUDNN_DATA_DOUBLE)) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    scaleKernel<<<gridFor(elements(dx)), 256, 0, stream>>>(dx, dxData, beta);
    samplerBackwardKernel<<<gridFor(dy.dims[0] * dy.dims[2] * dy.dims[3]), 256, 0, stream>>>(
        alpha, x, xData, dx, dxData, alphaGrid, dy, dyData, grid, betaGrid, dgrid);
    return launched();
}

} // namespace librecuda_dnn
