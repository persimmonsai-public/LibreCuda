/**
 * Convolutions as CUTLASS implicit GEMMs on tensor cores (float accumulation): half and bfloat16 data, and float data
 * as TF32 when allowed; 1D, 2D and 3D problems without groups. The operands are copied into channels-last layouts
 * with the channels padded for the kernels' 16 byte accesses (multiples of 8 of 16 bit data, 4 of float data): x / dx
 * as N(D)HWC, dy / y as N(Z)PQK, filters as K(T)RSC. A 3D problem's depth is handled as further rows of positions
 * (the layout changes and filter packing see D x H rows of W positions, and T x R rows of S taps). The kernels write results of the data type scaled by alpha (float results
 * when beta blends in the previous output, and for filter gradients), which are then copied into the caller's
 * layout. Problems the kernels can't run go to the other paths (cudnn_conv.cu).
 */
#include "dnn_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cutlass/bfloat16.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/conv3d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/kernel/default_conv3d_dgrad.h>
#include <cutlass/conv/kernel/default_conv3d_fprop.h>
#include <cutlass/conv/kernel/default_conv3d_wgrad.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/half.h>

#include <algorithm>
#include <type_traits>

namespace librecuda_dnn {

namespace {

using cutlass::conv::IteratorAlgorithm;
using cutlass::conv::StrideSupport;
using cutlass::layout::TensorNDHWC;
using cutlass::layout::TensorNHWC;
// results of type O: float, or the data type (alpha applied, no beta)
template<typename O>
using Epilogue = cutlass::epilogue::thread::LinearCombination<O, 128 / cutlass::sizeof_bits<O>::value, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// 16 bit data: m16n8k16 instructions, tiles 32 deep; float data (as TF32): m16n8k8, 16 deep
template<typename T>
constexpr bool IS_FLOAT = cutlass::sizeof_bits<T>::value == 32;
template<typename T>
using Instruction =
    std::conditional_t<IS_FLOAT<T>, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::gemm::GemmShape<16, 8, 16>>;
template<typename T>
constexpr int DEPTH = IS_FLOAT<T> ? 16 : 32;
// tiles of 128 x 128 (4 warps of 64 x 64), or 128 x 64 for GEMMs with at most 64 columns (output channels)
template<bool Narrow, int K>
using Block = std::conditional_t<Narrow, cutlass::gemm::GemmShape<128, 64, K>, cutlass::gemm::GemmShape<128, 128, K>>;
template<bool Narrow, int K>
using Warp = std::conditional_t<Narrow, cutlass::gemm::GemmShape<64, 32, K>, cutlass::gemm::GemmShape<64, 64, K>>;

template<typename T, bool Narrow, typename O>
using Fprop = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dFprop<
    T, TensorNHWC, T, TensorNHWC, O, TensorNHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Block<Narrow, DEPTH<T>>, Warp<Narrow, DEPTH<T>>, Instruction<T>, Epilogue<O>, Swizzle, 3,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm::kOptimized>::Kernel>;

// inputs of at most 4 channels (images): padded to 4 instead of 8, 8 byte accesses
template<typename T, bool Narrow, typename O>
using FpropFewChannels = cutlass::conv::device::ImplicitGemmConvolution<
    typename cutlass::conv::kernel::DefaultConv2dFprop<
        T, TensorNHWC, T, TensorNHWC, O, TensorNHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        Block<Narrow, 64>, Warp<Narrow, 64>, Instruction<T>, Epilogue<O>, Swizzle, 3, cutlass::arch::OpMultiplyAdd,
        IteratorAlgorithm::kFewChannels, StrideSupport::kStrided, 4, 4>::Kernel>;

template<typename T, bool Narrow, typename O>
using DgradUnity = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dDgrad<
    T, TensorNHWC, T, TensorNHWC, O, TensorNHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Block<Narrow, DEPTH<T>>, Warp<Narrow, DEPTH<T>>, Instruction<T>, Epilogue<O>, Swizzle, 3,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm::kOptimized, StrideSupport::kUnity>::Kernel>;

template<typename T, bool Narrow, typename O>
using DgradStrided = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dDgrad<
    T, TensorNHWC, T, TensorNHWC, O, TensorNHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Block<Narrow, DEPTH<T>>, Warp<Narrow, DEPTH<T>>, Instruction<T>, Epilogue<O>,
    cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>, 3, cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm::kOptimized, StrideSupport::kStrided>::Kernel>;

// weight gradients (float results): tiles of 64 x 128 for at most 64 output channels (the GEMM's rows)
template<typename T, bool Narrow, typename O>
using Wgrad = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dWgrad<
    T, TensorNHWC, T, TensorNHWC, O, TensorNHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    std::conditional_t<Narrow, cutlass::gemm::GemmShape<64, 128, DEPTH<T>>, Block<false, DEPTH<T>>>,
    std::conditional_t<Narrow, cutlass::gemm::GemmShape<32, 64, DEPTH<T>>, Warp<false, DEPTH<T>>>, Instruction<T>,
    Epilogue<O>, Swizzle, 3, cutlass::arch::OpMultiplyAdd, IteratorAlgorithm::kOptimized>::Kernel>;

/**
 * 3D kernels (R3D, R(2+1)D video networks): 3D convolutions went through im2col, which took 75 % of an R3D-18
 * training step (7.5 times NVIDIA's). Fprop and wgrad have optimized iterators for any stride; dgrad only for unit
 * strides (strided problems take the im2col path).
 */
template<typename T, bool Narrow, typename O>
using Fprop3 = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv3dFprop<
    T, TensorNDHWC, T, TensorNDHWC, O, TensorNDHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Block<Narrow, DEPTH<T>>, Warp<Narrow, DEPTH<T>>, Instruction<T>, Epilogue<O>, Swizzle, 3,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm::kOptimized>::Kernel>;

template<typename T, bool Narrow, typename O>
using DgradUnity3 = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv3dDgrad<
    T, TensorNDHWC, T, TensorNDHWC, O, TensorNDHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    Block<Narrow, DEPTH<T>>, Warp<Narrow, DEPTH<T>>, Instruction<T>, Epilogue<O>, Swizzle, 3,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm::kOptimized, StrideSupport::kUnity>::Kernel>;

template<typename T, bool Narrow, typename O>
using Wgrad3 = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv3dWgrad<
    T, TensorNDHWC, T, TensorNDHWC, O, TensorNDHWC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    std::conditional_t<Narrow, cutlass::gemm::GemmShape<64, 128, DEPTH<T>>, Block<false, DEPTH<T>>>,
    std::conditional_t<Narrow, cutlass::gemm::GemmShape<32, 64, DEPTH<T>>, Warp<false, DEPTH<T>>>, Instruction<T>,
    Epilogue<O>, Swizzle, 3, cutlass::arch::OpMultiplyAdd, IteratorAlgorithm::kOptimized>::Kernel>;

// the kernels of 1D / 2D problems (Dims 2) and of 3D problems
template<int Dims>
struct Kernels;

template<>
struct Kernels<2> {
    template<typename T, bool N, typename O>
    using Fprop = ::librecuda_dnn::Fprop<T, N, O>;
    template<typename T, bool N, typename O>
    using DgradUnity = ::librecuda_dnn::DgradUnity<T, N, O>;
    template<typename T, bool N, typename O>
    using DgradStrided = ::librecuda_dnn::DgradStrided<T, N, O>;
    template<typename T, bool N, typename O>
    using Wgrad = ::librecuda_dnn::Wgrad<T, N, O>;
};

template<>
struct Kernels<3> {
    template<typename T, bool N, typename O>
    using Fprop = Fprop3<T, N, O>;
    template<typename T, bool N, typename O>
    using DgradUnity = DgradUnity3<T, N, O>;
    // (not used: see layoutOf)
    template<typename T, bool N, typename O>
    using DgradStrided = DgradUnity3<T, N, O>;
    template<typename T, bool N, typename O>
    using Wgrad = Wgrad3<T, N, O>;
};

// a 3D problem (2D problems have D = Z = T = 1, 1D problems also H = P = R = 1); Cp / Kp: the channels padded to
// multiples of 8 (4 for the forward pass of at most 4 input channels)
struct Problem {
    int n, c, h, w, k, r, s, p, q, cp, kp;
    int d, t, z;  // depths of x, the filters and y
    bool three;
    bool fewChannels;  // forward pass of 16 bit data with at most 4 channels (padded to 4)
    int padH, padW, strideH, strideW, dilationH, dilationW, padD, strideD, dilationD;
    bool flip;
    // strides of x (n, c, h, w), w (k, c, r, s), y (n, k, p, q); the depth strides of x and y are H (P) times the
    // row strides; wt: the filters' depth stride
    int64_t xs[4], ws[4], ys[4], wt;
};

int roundUp(int64_t v, int multiple) {
    return int((v + multiple - 1) / multiple * multiple);
}

bool problemOf(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g, Problem &pr) {
    // (float data only as TF32)
    if (g.spatial < 1 || g.spatial > 3 || x.dims[1] != w.dims[1] ||
        (x.type != CUDNN_DATA_HALF && x.type != CUDNN_DATA_BFLOAT16 &&
         !(x.type == CUDNN_DATA_FLOAT && g.allowTF32))) {
        return false;
    }
    for (int i = 0; i < x.nbDims; i++) {
        if (x.dims[i] > INT32_MAX / 2 || w.dims[i] > INT32_MAX / 2 || y.dims[i] > INT32_MAX / 2) return false;
    }
    const bool three = g.spatial == 3, two = g.spatial >= 2;
    // (3D: the spatial dimensions after the depth)
    const int o = three ? 1 : 0;
    pr.three = three;
    pr.d = three ? int(x.dims[2]) : 1;
    pr.t = three ? int(w.dims[2]) : 1;
    pr.z = three ? int(y.dims[2]) : 1;
    pr.padD = three ? int(g.padBefore[0]) : 0;
    pr.strideD = three ? int(g.stride[0]) : 1;
    pr.dilationD = three ? int(g.dilation[0]) : 1;
    pr.n = int(x.dims[0]);
    pr.c = int(x.dims[1]);
    pr.k = int(w.dims[0]);
    pr.h = two ? int(x.dims[2 + o]) : 1;
    pr.w = int(x.dims[two ? 3 + o : 2]);
    pr.r = two ? int(w.dims[2 + o]) : 1;
    pr.s = int(w.dims[two ? 3 + o : 2]);
    pr.p = two ? int(y.dims[2 + o]) : 1;
    pr.q = int(y.dims[two ? 3 + o : 2]);
    // 16 byte accesses: multiples of 8 channels of 16 bit data, 4 of float data; the forward pass of 16 bit inputs with
    // few channels has its own kernel
    const int multiple = x.type == CUDNN_DATA_FLOAT ? 4 : 8;
    pr.fewChannels = x.type != CUDNN_DATA_FLOAT && kind == ConvKind::FORWARD && pr.c <= 4 && !three;
    pr.cp = pr.fewChannels ? 4 : roundUp(pr.c, multiple);
    pr.kp = roundUp(pr.k, multiple);
    pr.padH = two ? int(g.padBefore[o]) : 0;
    pr.padW = int(g.padBefore[two ? 1 + o : 0]);
    pr.strideH = two ? int(g.stride[o]) : 1;
    pr.strideW = int(g.stride[two ? 1 + o : 0]);
    pr.dilationH = two ? int(g.dilation[o]) : 1;
    pr.dilationW = int(g.dilation[two ? 1 + o : 0]);
    pr.flip = g.flip;
    auto strides = [&](const Tensor &t, int64_t *out) {
        out[0] = t.strides[0];
        out[1] = t.strides[1];
        out[2] = two ? t.strides[2 + o] : 0;
        out[3] = t.strides[two ? 3 + o : 2];
    };
    strides(x, pr.xs);
    strides(w, pr.ws);
    strides(y, pr.ys);
    pr.wt = three ? w.strides[2] : 0;
    // (depth as further rows: the depth stride must continue the rows')
    return !three || ((pr.d == 1 || x.strides[2] == pr.h * pr.xs[2]) && (pr.z == 1 || y.strides[2] == pr.p * pr.ys[2]));
}

template<typename Conv>
typename Conv::UnderlyingKernel::ConvProblemSize cutlassProblem(const Problem &pr, int splits) {
    if constexpr (Conv::kConvDim == 3) {
        return {pr.n, pr.d, pr.h, pr.w, pr.cp, pr.kp, pr.t, pr.r, pr.s, pr.z, pr.p, pr.q, pr.padD, pr.padH, pr.padW,
                pr.strideD, pr.strideH, pr.strideW, pr.dilationD, pr.dilationH, pr.dilationW,
                cutlass::conv::Mode::kCrossCorrelation, splits};
    } else {
        return {pr.n, pr.h, pr.w, pr.cp, pr.kp, pr.r, pr.s, pr.p, pr.q, pr.padH, pr.padW, pr.strideH, pr.strideW,
                pr.dilationH, pr.dilationW, cutlass::conv::Mode::kCrossCorrelation, splits};
    }
}

// positions of x and y, and filter taps
int64_t inPositionsOf(const Problem &pr) { return int64_t(pr.d) * pr.h * pr.w; }
int64_t outPositionsOf(const Problem &pr) { return int64_t(pr.z) * pr.p * pr.q; }
int64_t tapsOf(const Problem &pr) { return int64_t(pr.t) * pr.r * pr.s; }

int multiprocessors() {
    int device = 0, count = 0;
    cudaGetDevice(&device);
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return std::max(count, 1);
}

// weight gradients: GEMM-K is N*P*Q, so few output tiles each run long; split K serially to fill the GPU
int wgradSplits(const Problem &pr) {
    const int64_t tiles = int64_t((pr.kp + 127) / 128) * ((tapsOf(pr) * pr.cp + 127) / 128);
    const int64_t gemmK = int64_t(pr.n) * outPositionsOf(pr);
    const int64_t wanted = (2 * multiprocessors() + tiles - 1) / tiles;
    return int(std::max<int64_t>(1, std::min<int64_t>({wanted, gemmK / 256, 64})));
}

/**
 * forward: K splits for problems of fewer output tiles than SMs (Stable Diffusion's 8 x 8 and 16 x 16 layers: 10 and
 * 20 tiles of 128 x 128 on 30 SMs ran 1.3 to 2.4 times as long as cuDNN's kernels); the partial sums add up in the
 * float results
 */
int fpropSplits(const Problem &pr) {
    const int64_t columns = pr.kp <= 64 ? 64 : 128;
    const int64_t tiles = (int64_t(pr.n) * outPositionsOf(pr) + 127) / 128 * ((pr.kp + columns - 1) / columns);
    const int sms = multiprocessors();
    if (pr.fewChannels || tiles >= sms) return 1;
    const int64_t gemmK = tapsOf(pr) * pr.cp;
    const int64_t wanted = (2 * sms + tiles - 1) / tiles;
    return int(std::max<int64_t>(1, std::min<int64_t>({wanted, gemmK / 256, 16})));
}

template<typename Conv>
bool fitsOne(const Problem &pr, int splits) {
    typename Conv::Arguments args;
    args.problem_size = cutlassProblem<Conv>(pr, splits);
    args.split_k_mode = cutlass::conv::SplitKMode::kSerial;
    return Conv::can_implement(args) == cutlass::Status::kSuccess;
}

// (with float results, and for direct: also with results of the data type)
template<template<typename, bool, typename> class Conv, typename E>
bool fits(const Problem &pr, bool narrow, int splits, bool direct) {
    if (narrow) {
        return fitsOne<Conv<E, true, float>>(pr, splits) && (!direct || fitsOne<Conv<E, true, E>>(pr, splits));
    }
    return fitsOne<Conv<E, false, float>>(pr, splits) && (!direct || fitsOne<Conv<E, false, E>>(pr, splits));
}

// at most 64 of the GEMM's columns: output channels (forward), input channels (backward data); weight gradients: rows
bool narrowFor(ConvKind kind, const Problem &pr) {
    return (kind == ConvKind::BACKWARD_DATA ? pr.cp : pr.kp) <= 64;
}

bool unitStrides(const Problem &pr) {
    return pr.strideH == 1 && pr.strideW == 1 && pr.strideD == 1;
}

struct Layout {
    size_t a, b, out, semaphores, total;
};

size_t align256(size_t bytes) {
    return (bytes + 255) / 256 * 256;
}

/**
 * workspace: the packed operands a (x / dy) and b (filters / x), the float result and the split-K semaphores; total 0
 * when the kernels can't run the problem
 */
template<typename T, typename K>
Layout layoutOf(ConvKind kind, const Problem &pr) {
    const size_t xBytes = size_t(pr.n) * inPositionsOf(pr) * pr.cp, yBytes = size_t(pr.n) * outPositionsOf(pr) * pr.kp;
    const size_t wBytes = size_t(pr.kp) * tapsOf(pr) * pr.cp;
    Layout l{};
    switch (kind) {
        case ConvKind::FORWARD:
            if constexpr (!IS_FLOAT<T>) {
                if (pr.fewChannels && !fits<FpropFewChannels, T>(pr, narrowFor(kind, pr), 1, true)) return l;
            }
            if (!pr.fewChannels && !fits<K::template Fprop, T>(pr, narrowFor(kind, pr), 1, true)) return l;
            l.a = align256(xBytes * sizeof(T));
            l.b = align256(wBytes * sizeof(T));
            l.out = align256(yBytes * 4);
            if (const int splits = fpropSplits(pr); splits > 1) {
                if (!fits<K::template Fprop, T>(pr, narrowFor(kind, pr), splits, false)) return l;
                using Wide = typename K::template Fprop<T, false, float>;
                using Narrow = typename K::template Fprop<T, true, float>;
                typename Wide::Arguments args;
                typename Narrow::Arguments narrowArgs;
                args.problem_size = cutlassProblem<Wide>(pr, splits);
                narrowArgs.problem_size = cutlassProblem<Narrow>(pr, splits);
                args.split_k_mode = narrowArgs.split_k_mode = cutlass::conv::SplitKMode::kSerial;
                l.semaphores = align256(narrowFor(kind, pr) ? Narrow::get_workspace_size(narrowArgs)
                                                            : Wide::get_workspace_size(args));
            }
            break;
        case ConvKind::BACKWARD_DATA:
            // (3D strided: CUTLASS has only the analytic iterator, which visits every tap for every input position
            // and took as long as im2col for R3D's downsampling convolutions, 6 times as long for its stem's)
            if (pr.three && !unitStrides(pr)) return l;
            if (unitStrides(pr) ? !fits<K::template DgradUnity, T>(pr, narrowFor(kind, pr), 1, true)
                                : !fits<K::template DgradStrided, T>(pr, narrowFor(kind, pr), 1, true)) {
                return l;
            }
            l.a = align256(yBytes * sizeof(T));
            l.b = align256(wBytes * sizeof(T));
            l.out = align256(xBytes * 4);
            break;
        case ConvKind::BACKWARD_FILTER: {
            const int splits = wgradSplits(pr);
            if (!fits<K::template Wgrad, T>(pr, narrowFor(kind, pr), splits, false)) return l;
            using Wide = typename K::template Wgrad<T, false, float>;
            using Narrow = typename K::template Wgrad<T, true, float>;
            typename Wide::Arguments args;
            typename Narrow::Arguments narrowArgs;
            args.problem_size = cutlassProblem<Wide>(pr, splits);
            narrowArgs.problem_size = cutlassProblem<Narrow>(pr, splits);
            args.split_k_mode = narrowArgs.split_k_mode = cutlass::conv::SplitKMode::kSerial;
            l.a = align256(yBytes * sizeof(T));
            l.b = align256(xBytes * sizeof(T));
            l.out = align256(wBytes * 4);
            l.semaphores = align256(narrowFor(kind, pr) ? Narrow::get_workspace_size(narrowArgs)
                                                        : Wide::get_workspace_size(args));
            break;
        }
    }
    l.total = l.a + l.b + l.out + l.semaphores;
    return l;
}

// ---- layout changes

/**
 * dst(n, c, i) = alpha src(n, c, i) (+ beta dst) for c < channels (src channels >= srcChannels read as 0) and the
 * spatial positions i < positions (i = h * width + w); strides of n, c, h, w. Tiles of 32 channels x 32 positions
 * go through shared memory, so both sides are read / written along their contiguous dimension.
 */
struct Relayout {
    int channels, srcChannels, positions, width;
    int64_t src[4], dst[4];
    float alpha, beta;
};

template<typename T>
__device__ inline float load(const T *p) {
    if constexpr (std::is_same_v<T, float>) {
        return *p;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(*p);
    } else {
        return __bfloat162float(*p);
    }
}

template<typename T>
__device__ inline void store(T *p, float v) {
    if constexpr (std::is_same_v<T, float>) {
        *p = v;
    } else if constexpr (std::is_same_v<T, __half>) {
        *p = __float2half(v);
    } else {
        *p = __float2bfloat16(v);
    }
}

template<typename S, typename D>
__global__ void relayout(Relayout r, const S *src, D *dst) {
    __shared__ float tile[32][33];
    const int c0 = blockIdx.y * 32, i0 = blockIdx.x * 32;
    const int64_t n = blockIdx.z;
    // read along the dimension with the smaller stride
    const bool srcChannelMajor = r.src[1] < r.src[3];
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int cl = srcChannelMajor ? threadIdx.x : j, il = srcChannelMajor ? j : threadIdx.x;
        const int c = c0 + cl, i = i0 + il;
        float v = 0;
        if (c < r.srcChannels && i < r.positions) {
            v = load(src + n * r.src[0] + c * r.src[1] + (i / r.width) * r.src[2] + (i % r.width) * r.src[3]);
        }
        tile[cl][il] = v;
    }
    __syncthreads();
    const bool dstChannelMajor = r.dst[1] < r.dst[3];
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int cl = dstChannelMajor ? threadIdx.x : j, il = dstChannelMajor ? j : threadIdx.x;
        const int c = c0 + cl, i = i0 + il;
        if (c < r.channels && i < r.positions) {
            D *p = dst + n * r.dst[0] + c * r.dst[1] + (i / r.width) * r.dst[2] + (i % r.width) * r.dst[3];
            float v = r.alpha * tile[cl][il];
            if (r.beta != 0) v += r.beta * load(p);
            store(p, v);
        }
    }
}

/**
 * The same where both sides store the positions linearly (h stride = width x w stride: NCHW, packed NHWC), in tiles of
 * 64 x 64: no division of positions into rows, 16 elements per thread. relayout, a division and 64 bit arithmetic per
 * element and 4 elements per thread, moved 180 GB/s of 440 (Stable Diffusion's UNet spent 10 % of its time in it).
 */
constexpr int LINEAR_TILE = 64;

template<typename S, typename D>
__global__ void relayoutLinear(Relayout r, const S *src, D *dst) {
    __shared__ float tile[LINEAR_TILE][LINEAR_TILE + 1];
    const int c0 = blockIdx.y * LINEAR_TILE, i0 = blockIdx.x * LINEAR_TILE;
    const S *from = src + int64_t(blockIdx.z) * r.src[0];
    D *to = dst + int64_t(blockIdx.z) * r.dst[0];
    const bool srcChannelMajor = r.src[1] < r.src[3];
#pragma unroll 4
    for (int j = threadIdx.y; j < LINEAR_TILE; j += blockDim.y) {
        const int cl = srcChannelMajor ? threadIdx.x : j, il = srcChannelMajor ? j : threadIdx.x;
        const int c = c0 + cl, i = i0 + il;
        tile[cl][il] = c < r.srcChannels && i < r.positions ? load(from + c * r.src[1] + i * r.src[3]) : 0.f;
    }
    __syncthreads();
    const bool dstChannelMajor = r.dst[1] < r.dst[3];
#pragma unroll 4
    for (int j = threadIdx.y; j < LINEAR_TILE; j += blockDim.y) {
        const int cl = dstChannelMajor ? threadIdx.x : j, il = dstChannelMajor ? j : threadIdx.x;
        const int c = c0 + cl, i = i0 + il;
        if (c < r.channels && i < r.positions) {
            D *p = to + c * r.dst[1] + i * r.dst[3];
            float v = r.alpha * tile[cl][il];
            if (r.beta != 0) v += r.beta * load(p);
            store(p, v);
        }
    }
}

// the same for few channels (at most 8): a thread per position
template<typename S, typename D>
__global__ void relayoutFewChannels(Relayout r, int images, const S *src, D *dst) {
    const int64_t total = int64_t(images) * r.positions;
    for (int64_t e = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; e < total; e += int64_t(gridDim.x) * blockDim.x) {
        const int64_t n = e / r.positions;
        const int i = int(e % r.positions), h = i / r.width, w = i % r.width;
        const S *from = src + n * r.src[0] + h * r.src[2] + w * r.src[3];
        D *to = dst + n * r.dst[0] + h * r.dst[2] + w * r.dst[3];
        for (int c = 0; c < r.channels; c++) {
            float v = r.alpha * (c < r.srcChannels ? load(from + c * r.src[1]) : 0.f);
            if (r.beta != 0) v += r.beta * load(to + c * r.dst[1]);
            store(to + c * r.dst[1], v);
        }
    }
}

template<typename S, typename D>
cudnnStatus_t runRelayout(cudaStream_t stream, const Relayout &r, int images, const S *src, D *dst) {
    if (r.channels <= 8) {
        const int64_t total = int64_t(images) * r.positions;
        relayoutFewChannels<<<unsigned(std::max<int64_t>(1, std::min<int64_t>((total + 255) / 256, 1 << 20))), 256, 0,
                              stream>>>(r, images, src, dst);
        return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
    }
    if (r.src[2] == r.width * r.src[3] && r.dst[2] == r.width * r.dst[3]) {
        const dim3 grid((r.positions + LINEAR_TILE - 1) / LINEAR_TILE, (r.channels + LINEAR_TILE - 1) / LINEAR_TILE,
                        images);
        if (grid.y <= 65535 && grid.z <= 65535) {
            relayoutLinear<<<grid, dim3(LINEAR_TILE, 4), 0, stream>>>(r, src, dst);
            return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
        }
    }
    const dim3 grid((r.positions + 31) / 32, (r.channels + 31) / 32, images);
    if (grid.y > 65535 || grid.z > 65535) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    relayout<<<grid, dim3(32, 8), 0, stream>>>(r, src, dst);
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// channels-last strides of an [images][positions][channels] buffer, as (n, c, h, w)
void packedStrides(int positions, int width, int channels, int64_t *out) {
    out[0] = int64_t(positions) * channels;
    out[1] = 1;
    out[2] = int64_t(width) * channels;
    out[3] = channels;
}

/**
 * filters <-> packed [Kp][R][S][Cp] (flipped for CUDNN_CONVOLUTION); toPacked: w into packed (zero padding), else
 * w = alpha packed + beta w
 */
template<typename T, typename P>
__global__ void filterRelayout(Problem pr, const T *w, P *packed, T *wOut, float alpha, float beta, bool toPacked) {
    const int rows = pr.t * pr.r;  // (depth and rows of taps; flipping both flips this index)
    const int64_t total = int64_t(pr.kp) * rows * pr.s * pr.cp;
    for (int64_t e = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; e < total; e += int64_t(gridDim.x) * blockDim.x) {
        const int c = int(e % pr.cp);
        int64_t rest = e / pr.cp;
        int s = int(rest % pr.s);
        rest /= pr.s;
        int tr = int(rest % rows);
        const int k = int(rest / rows);
        if (pr.flip) {
            tr = rows - 1 - tr;
            s = pr.s - 1 - s;
        }
        const bool inside = c < pr.c && k < pr.k;
        const int64_t at = k * pr.ws[0] + c * pr.ws[1] + (tr / pr.r) * pr.wt + (tr % pr.r) * pr.ws[2] + s * pr.ws[3];
        if (toPacked) {
            store(&packed[e], inside ? load(w + at) : 0.f);
        } else if (inside) {
            float v = alpha * load(&packed[e]);
            if (beta != 0) v += beta * load(wOut + at);
            store(wOut + at, v);
        }
    }
}

unsigned int gridFor(int64_t count) {
    return unsigned(std::max<int64_t>(1, std::min<int64_t>((count + 255) / 256, 1 << 20)));
}

/**
 * The same into packed for filters whose rows (k) are contiguous [C][R][S]: a block per row of packed reads the
 * filter's row into shared memory (coalesced) and writes it as [R][S][Cp] (coalesced). filterRelayout, a thread per
 * element with 64 bit index arithmetic and loads R * S apart, took twice as long as the convolution's input relayout.
 */
template<typename T>
__global__ void filterPack(Problem pr, const T *w, T *packed) {
    extern __shared__ __align__(16) unsigned char shared[];
    T *row = reinterpret_cast<T *>(shared);
    const int k = blockIdx.x, rs = pr.t * pr.r * pr.s, length = pr.c * rs;
    T *to = packed + int64_t(k) * rs * pr.cp;
    if (k >= pr.k) {
        for (int e = threadIdx.x; e < rs * pr.cp; e += blockDim.x) to[e] = T(0.f);
        return;
    }
    const T *from = w + k * pr.ws[0];
    for (int e = threadIdx.x; e < length; e += blockDim.x) row[e] = from[e];
    __syncthreads();
    for (int e = threadIdx.x; e < rs * pr.cp; e += blockDim.x) {
        const int c = e % pr.cp, tap = e / pr.cp;
        to[e] = c < pr.c ? row[c * rs + (pr.flip ? rs - 1 - tap : tap)] : T(0.f);
    }
}

// shared memory filterPack takes for a row
constexpr size_t FILTER_ROW_LIMIT = 48 << 10;

/**
 * The filters as packed [Kp][R][S][Cp]: the caller's filters themselves where they are laid out so already (1 x 1
 * filters, or [K][R][S][C], without padding or flipping), else relaid out into packed
 */
template<typename T>
cudnnStatus_t packFilters(cudaStream_t stream, const Problem &pr, const T *w, T *packed, const T **out) {
    const int64_t rs = tapsOf(pr);
    // (the kernels load 16 bytes at a time)
    const bool exact = pr.c == pr.cp && pr.k == pr.kp && (!pr.flip || rs == 1) &&
                       reinterpret_cast<uintptr_t>(w) % 16 == 0;
    if (exact && pr.ws[1] == 1 && pr.ws[3] == pr.c && pr.ws[2] == pr.s * pr.c && pr.ws[0] == rs * pr.c &&
        (pr.t == 1 || pr.wt == int64_t(pr.r) * pr.s * pr.c)) {
        *out = w;  // [K][T][R][S][C]
        return CUDNN_STATUS_SUCCESS;
    }
    if (exact && rs == 1 && pr.ws[1] == 1 && pr.ws[0] == pr.c) {
        *out = w;  // [K][C][1][1]
        return CUDNN_STATUS_SUCCESS;
    }
    *out = packed;
    const size_t rowBytes = size_t(pr.c) * rs * sizeof(T);
    if (pr.ws[3] == 1 && pr.ws[2] == pr.s && (pr.t == 1 || pr.wt == int64_t(pr.r) * pr.s) && pr.ws[1] == rs &&
        rowBytes <= FILTER_ROW_LIMIT) {
        filterPack<T><<<unsigned(pr.kp), 256, rowBytes, stream>>>(pr, w, packed);
    } else {
        const int64_t filterCount = int64_t(pr.kp) * rs * pr.cp;
        filterRelayout<T, T><<<gridFor(filterCount), 256, 0, stream>>>(pr, w, packed, nullptr, 1, 0, true);
    }
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

template<typename Conv, typename E, typename O>
cudnnStatus_t runOne(cudaStream_t stream, const Problem &pr, int splits, const E *a, const E *b, O *out, float alpha,
                     void *semaphores) {
    const auto problem = cutlassProblem<Conv>(pr, splits);
    using Tensor = std::conditional_t<Conv::kConvDim == 3, TensorNDHWC, TensorNHWC>;
    Tensor x, y, w;
    if constexpr (Conv::kConvDim == 3) {
        x = TensorNDHWC::packed({pr.n, pr.d, pr.h, pr.w, pr.cp});
        y = TensorNDHWC::packed({pr.n, pr.z, pr.p, pr.q, pr.kp});
        w = TensorNDHWC::packed({pr.kp, pr.t, pr.r, pr.s, pr.cp});
    } else {
        x = TensorNHWC::packed({pr.n, pr.h, pr.w, pr.cp});
        y = TensorNHWC::packed({pr.n, pr.p, pr.q, pr.kp});
        w = TensorNHWC::packed({pr.kp, pr.r, pr.s, pr.cp});
    }
    constexpr auto op = Conv::kConvolutionalOperator;
    const Tensor la = op == cutlass::conv::Operator::kFprop ? x : y;
    const Tensor lb = op == cutlass::conv::Operator::kWgrad ? x : w;
    const Tensor lc = op == cutlass::conv::Operator::kFprop ? y : op == cutlass::conv::Operator::kDgrad ? x : w;
    typename Conv::Arguments args(problem, {const_cast<E *>(a), la}, {const_cast<E *>(b), lb}, {out, lc}, {out, lc},
                                  {alpha, 0.0f}, cutlass::conv::SplitKMode::kSerial);
    Conv conv;
    if (conv.can_implement(args) != cutlass::Status::kSuccess) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    if (conv.initialize(args, semaphores, stream) != cutlass::Status::kSuccess ||
        conv.run(stream) != cutlass::Status::kSuccess) {
        cudaGetLastError();
        return CUDNN_STATUS_EXECUTION_FAILED;
    }
    return CUDNN_STATUS_SUCCESS;
}

// direct: results of the data type E (scaled by alpha), else float results (alpha 1)
template<template<typename, bool, typename> class Conv, typename E>
cudnnStatus_t runConv(cudaStream_t stream, const Problem &pr, bool narrow, int splits, const E *a, const E *b,
                      void *out, bool direct, float alpha, void *semaphores) {
    if (direct) {
        E *o = static_cast<E *>(out);
        return narrow ? runOne<Conv<E, true, E>>(stream, pr, splits, a, b, o, alpha, semaphores)
                      : runOne<Conv<E, false, E>>(stream, pr, splits, a, b, o, alpha, semaphores);
    }
    float *o = static_cast<float *>(out);
    return narrow ? runOne<Conv<E, true, float>>(stream, pr, splits, a, b, o, 1.0f, semaphores)
                  : runOne<Conv<E, false, float>>(stream, pr, splits, a, b, o, 1.0f, semaphores);
}

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudnnStatus_t status_ = (call);                                                                          \
        if (status_ != CUDNN_STATUS_SUCCESS) return status_;                                                           \
    } while (0)

/**
 * T: the API's element type (__half / __nv_bfloat16), E: CUTLASS's (same bits)
 */
template<typename T, typename E, typename K>
cudnnStatus_t run(ConvKind kind, cudaStream_t stream, const Problem &pr, void *xData, void *wData, void *yData,
                  float alpha, float beta, char *workspace) {
    const Layout l = layoutOf<E, K>(kind, pr);
    T *a = reinterpret_cast<T *>(workspace);
    T *b = reinterpret_cast<T *>(workspace + l.a);
    void *out = workspace + l.a + l.b;
    void *semaphores = l.semaphores != 0 ? workspace + l.a + l.b + l.out : nullptr;
    const int inPositions = int(inPositionsOf(pr)), outPositions = int(outPositionsOf(pr));
    // x <-> packed NHWC, y / dy <-> packed NPQK
    auto xRelayout = [&](bool toPacked, int channels, float al, float be) {
        Relayout r{channels, toPacked ? pr.c : pr.cp, inPositions, pr.w, {}, {}, al, be};
        int64_t user[4] = {pr.xs[0], pr.xs[1], pr.xs[2], pr.xs[3]}, packed[4];
        packedStrides(inPositions, pr.w, pr.cp, packed);
        std::copy(toPacked ? user : packed, (toPacked ? user : packed) + 4, r.src);
        std::copy(toPacked ? packed : user, (toPacked ? packed : user) + 4, r.dst);
        return r;
    };
    auto yRelayout = [&](bool toPacked, int channels, float al, float be) {
        Relayout r{channels, toPacked ? pr.k : pr.kp, outPositions, pr.q, {}, {}, al, be};
        int64_t user[4] = {pr.ys[0], pr.ys[1], pr.ys[2], pr.ys[3]}, packed[4];
        packedStrides(outPositions, pr.q, pr.kp, packed);
        std::copy(toPacked ? user : packed, (toPacked ? user : packed) + 4, r.src);
        std::copy(toPacked ? packed : user, (toPacked ? packed : user) + 4, r.dst);
        return r;
    };
    /**
     * The caller's tensors themselves where they are packed channels-last already (NHWC without padded channels,
     * aligned for 16 byte accesses): no copy into or out of the workspace. With channels-last tensors (PyTorch's
     * channels_last), ResNet-50's training copied every activation and gradient twice (11 % of its time).
     */
    auto packedNhwc = [](const int64_t *st, int width, int positions, int channels, int padded, const void *p) {
        return channels == padded && st[1] == 1 && st[3] == channels && st[2] == int64_t(width) * channels &&
               st[0] == int64_t(positions) * channels && reinterpret_cast<uintptr_t>(p) % 16 == 0;
    };
    const bool xPacked = packedNhwc(pr.xs, pr.w, inPositions, pr.c, pr.cp, xData);
    const bool yPacked = packedNhwc(pr.ys, pr.q, outPositions, pr.k, pr.kp, yData);
    const int64_t filterCount = int64_t(pr.kp) * tapsOf(pr) * pr.cp;
    E *pa = reinterpret_cast<E *>(a), *pb = reinterpret_cast<E *>(b);
    const T *filters = nullptr;  // the packed filters: b or the caller's
    // without beta, the kernels write the data type (scaled) and the result is only reordered
    const bool direct = beta == 0;
    auto unpack = [&](const Relayout &r, T *to) {
        if (direct) {
            Relayout copy = r;
            copy.alpha = 1;
            return runRelayout(stream, copy, pr.n, static_cast<const T *>(out), to);
        }
        return runRelayout(stream, r, pr.n, static_cast<const float *>(out), to);
    };
    switch (kind) {
        case ConvKind::FORWARD: {
            if (xPacked) {
                pa = static_cast<E *>(xData);
            } else {
                CHECK(runRelayout(stream, xRelayout(true, pr.cp, 1, 0), pr.n, static_cast<const T *>(xData), a));
            }
            CHECK(packFilters(stream, pr, static_cast<const T *>(wData), b, &filters));
            pb = reinterpret_cast<E *>(const_cast<T *>(filters));
            // (results of the data type straight into the caller's packed output)
            void *target = direct && yPacked ? yData : out;
            if constexpr (!IS_FLOAT<E>) {
                if (pr.fewChannels) {
                    CHECK(runConv<FpropFewChannels>(stream, pr, narrowFor(kind, pr), 1, pa, pb, target, direct, alpha,
                                                    nullptr));
                    return target == yData ? CUDNN_STATUS_SUCCESS
                                           : unpack(yRelayout(false, pr.k, alpha, beta), static_cast<T *>(yData));
                }
            }
            if (const int splits = fpropSplits(pr); splits > 1) {
                // (float partial sums: the result is scaled when it is unpacked)
                CHECK(runConv<K::template Fprop>(stream, pr, narrowFor(kind, pr), splits, pa, pb, out, false, 1,
                                                 semaphores));
                return runRelayout(stream, yRelayout(false, pr.k, alpha, beta), pr.n, static_cast<const float *>(out),
                                   static_cast<T *>(yData));
            }
            CHECK(runConv<K::template Fprop>(stream, pr, narrowFor(kind, pr), 1, pa, pb, target, direct, alpha,
                                             nullptr));
            return target == yData ? CUDNN_STATUS_SUCCESS
                                   : unpack(yRelayout(false, pr.k, alpha, beta), static_cast<T *>(yData));
        }
        case ConvKind::BACKWARD_DATA: {
            if (yPacked) {
                pa = static_cast<E *>(yData);
            } else {
                CHECK(runRelayout(stream, yRelayout(true, pr.kp, 1, 0), pr.n, static_cast<const T *>(yData), a));
            }
            CHECK(packFilters(stream, pr, static_cast<const T *>(wData), b, &filters));
            pb = reinterpret_cast<E *>(const_cast<T *>(filters));
            void *target = direct && xPacked ? xData : out;
            if (unitStrides(pr)) {
                CHECK(runConv<K::template DgradUnity>(stream, pr, narrowFor(kind, pr), 1, pa, pb, target, direct,
                                                      alpha, nullptr));
            } else {
                // (positions no filter tap reaches stay 0)
                const size_t bytes = target == xData ? size_t(pr.n) * inPositions * pr.c * sizeof(T) : l.out;
                if (cudaMemsetAsync(target, 0, bytes, stream) != cudaSuccess) return CUDNN_STATUS_EXECUTION_FAILED;
                CHECK(runConv<K::template DgradStrided>(stream, pr, narrowFor(kind, pr), 1, pa, pb, target, direct,
                                                        alpha, nullptr));
            }
            return target == xData ? CUDNN_STATUS_SUCCESS
                                   : unpack(xRelayout(false, pr.c, alpha, beta), static_cast<T *>(xData));
        }
        case ConvKind::BACKWARD_FILTER:
            if (yPacked) {
                pa = static_cast<E *>(yData);
            } else {
                CHECK(runRelayout(stream, yRelayout(true, pr.kp, 1, 0), pr.n, static_cast<const T *>(yData), a));
            }
            if (xPacked) {
                pb = static_cast<E *>(xData);
            } else {
                CHECK(runRelayout(stream, xRelayout(true, pr.cp, 1, 0), pr.n, static_cast<const T *>(xData), b));
            }
            CHECK(runConv<K::template Wgrad>(stream, pr, narrowFor(kind, pr), wgradSplits(pr), pa, pb, out, false, 1,
                                             semaphores));
            filterRelayout<T, float><<<gridFor(filterCount), 256, 0, stream>>>(
                pr, nullptr, static_cast<float *>(out), static_cast<T *>(wData), alpha, beta, false);
            return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
    }
    return CUDNN_STATUS_NOT_SUPPORTED;
}

} // namespace

size_t implicitConvWorkspace(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g) {
    Problem pr{};
    if (!problemOf(kind, x, w, y, g, pr)) {
        return 0;
    }
    // (an empty workspace means unsupported: there is always a packed operand)
    if (pr.three) {
        switch (x.type) {
            case CUDNN_DATA_HALF: return layoutOf<cutlass::half_t, Kernels<3>>(kind, pr).total;
            case CUDNN_DATA_BFLOAT16: return layoutOf<cutlass::bfloat16_t, Kernels<3>>(kind, pr).total;
            default: return layoutOf<float, Kernels<3>>(kind, pr).total;
        }
    }
    switch (x.type) {
        case CUDNN_DATA_HALF: return layoutOf<cutlass::half_t, Kernels<2>>(kind, pr).total;
        case CUDNN_DATA_BFLOAT16: return layoutOf<cutlass::bfloat16_t, Kernels<2>>(kind, pr).total;
        default: return layoutOf<float, Kernels<2>>(kind, pr).total;
    }
}

cudnnStatus_t implicitConvRun(ConvKind kind, cudaStream_t stream, const Tensor &x, void *xData, const Tensor &w,
                              void *wData, const Tensor &y, void *yData, const ConvGeometry &g, double alpha,
                              double beta, void *workspace) {
    Problem pr{};
    if (!problemOf(kind, x, w, y, g, pr)) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    char *ws = static_cast<char *>(workspace);
    auto dispatch = [&](auto kernels) {
        using K = decltype(kernels);
        switch (x.type) {
            case CUDNN_DATA_HALF:
                return run<__half, cutlass::half_t, K>(kind, stream, pr, xData, wData, yData, float(alpha),
                                                       float(beta), ws);
            case CUDNN_DATA_BFLOAT16:
                return run<__nv_bfloat16, cutlass::bfloat16_t, K>(kind, stream, pr, xData, wData, yData, float(alpha),
                                                                  float(beta), ws);
            default:
                return run<float, float, K>(kind, stream, pr, xData, wData, yData, float(alpha), float(beta), ws);
        }
    };
    return pr.three ? dispatch(Kernels<3>{}) : dispatch(Kernels<2>{});
}

} // namespace librecuda_dnn
