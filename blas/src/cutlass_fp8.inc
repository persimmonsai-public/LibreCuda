// CUTLASS tensor core GEMMs of 8 bit float operands (ELEMENT_A, ELEMENT_B: float_e4m3_t or float_e5m2_t; float
// accumulation), included by cutlass_e*_e*.cu as FUNCTION. The sm_89 kernels (mma.sync of e4m3 / e5m2, no TMA) run on
// newer GPUs too. mma.sync takes 8 bit operands only as A row-major and B column-major (cuBLASLt's "TN" layout, the
// only one it supports for them too). 16 byte accesses; tiles of 128 x 128 (4 stages), or 64 x 64 (6 stages) for
// problems with few tiles. The epilogue applies cuBLASLt's scales and computes the absolute maximum; it stores 16 bytes
// per access, 8 bytes for 8 bit D (the epilogue's thread map has no layout for 16 of them).
#include "cutlass_gemm.cuh"

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/functional.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

namespace librecuda_blas {
namespace {

/**
 * D = scaleD * X, X = alpha scaleA scaleB accumulator + beta scaleC source, and amax = max |X| (as cuBLASLt's FP8
 * matmuls). The interface of CUTLASS's LinearCombination (what the 2.x epilogue uses of it).
 */
template<typename TD, int Count>
class ScaledCombination {
public:
    using ElementOutput = TD;
    using ElementSource = TD;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementC = TD;
    using ElementD = TD;
    static int const kCount = Count;
    static const cutlass::epilogue::thread::ScaleType::Kind kScale = cutlass::epilogue::thread::ScaleType::Default;
    static cutlass::FloatRoundStyle const kRound = cutlass::FloatRoundStyle::round_to_nearest;
    using FragmentOutput = cutlass::Array<TD, Count>;
    using FragmentSource = cutlass::Array<TD, Count>;
    using FragmentAccumulator = cutlass::Array<float, Count>;
    using FragmentCompute = cutlass::Array<float, Count>;
    using Params = Fp8Epilogue;

    CUTLASS_HOST_DEVICE
    explicit ScaledCombination(const Params &params) : amax_(params.amax) {
        alpha_ = params.alphaPtr != nullptr ? *params.alphaPtr : params.alpha;
        beta_ = params.betaPtr != nullptr ? *params.betaPtr : params.beta;
        if (params.scaleA != nullptr) alpha_ *= *params.scaleA;
        if (params.scaleB != nullptr) alpha_ *= *params.scaleB;
        if (params.scaleC != nullptr) beta_ *= *params.scaleC;
        scaleD_ = params.scaleD != nullptr ? *params.scaleD : 1.0f;
    }

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return beta_ != 0.0f; }

    // (no split k)
    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int) {
        if (k_partition) beta_ = 1.0f;
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(const FragmentAccumulator &accumulator, const FragmentSource &source) const {
        cutlass::NumericArrayConverter<float, TD, Count, kRound> source_converter;
        FragmentCompute x = cutlass::multiplies<FragmentCompute>()(beta_, source_converter(source));
        x = cutlass::multiply_add<FragmentCompute>()(alpha_, accumulator, x);
        return finish(x);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(const FragmentAccumulator &accumulator) const {
        return finish(cutlass::multiplies<FragmentCompute>()(alpha_, accumulator));
    }

private:
    CUTLASS_HOST_DEVICE
    FragmentOutput finish(const FragmentCompute &x) const {
#if defined(__CUDA_ARCH__)
        if (amax_ != nullptr) {
            float local = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Count; ++i) {
                local = fmaxf(local, fabsf(x[i]));
            }
            // non-negative floats order as their bits: one atomic per warp
            const unsigned mask = __activemask();
            const unsigned bits = __reduce_max_sync(mask, __float_as_uint(local));
            if ((threadIdx.x & 31) == __ffs(mask) - 1) {
                atomicMax(reinterpret_cast<unsigned *>(amax_), bits);
            }
        }
#endif
        cutlass::NumericArrayConverter<TD, float, Count, kRound> destination_converter;
        return destination_converter(cutlass::multiplies<FragmentCompute>()(scaleD_, x));
    }

    float alpha_, beta_, scaleD_;
    float *amax_;
};

template<typename TD, bool Small>
using Fp8Op = cutlass::gemm::device::GemmUniversal<
    ELEMENT_A, RowMajor, ELEMENT_B, ColumnMajor, TD, ColumnMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    std::conditional_t<Small, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<128, 128, 64>>,
    std::conditional_t<Small, cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<64, 64, 64>>,
    cutlass::gemm::GemmShape<16, 8, 32>, ScaledCombination<TD, fp8OutputAlignment(sizeof(TD))>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Small ? 6 : 4, 16, 16>;

template<bool Small>
cublasStatus_t withOutput(const GemmParams &p, const Fp8Epilogue &e) {
    switch (p.d.type) {
        case CUDA_R_16F: return runCutlass<Fp8Op<cutlass::half_t, Small>>(p, e);
        case CUDA_R_16BF: return runCutlass<Fp8Op<cutlass::bfloat16_t, Small>>(p, e);
        case CUDA_R_32F: return runCutlass<Fp8Op<float, Small>>(p, e);
        case CUDA_R_8F_E4M3: return runCutlass<Fp8Op<cutlass::float_e4m3_t, Small>>(p, e);
        case CUDA_R_8F_E5M2: return runCutlass<Fp8Op<cutlass::float_e5m2_t, Small>>(p, e);
        default: return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

} // namespace

cublasStatus_t FUNCTION(const GemmParams &p, const Fp8Epilogue &e, bool small) {
    if (p.transa != CUBLAS_OP_T || p.transb != CUBLAS_OP_N) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return small ? withOutput<true>(p, e) : withOutput<false>(p, e);
}

} // namespace librecuda_blas
