/**
 * LibreCUDA's cuDNN: elementwise operations, batch normalization, pooling and the spatial transformer, on tensors of
 * any strides. Elements are read and written as their descriptor's data type (half, bfloat16, float, double) and
 * computed in double (statistics) or float.
 */
#include "dnn_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>

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
