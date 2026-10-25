// CUTLASS tensor core GEMMs of 8 bit float operands (ELEMENT_A, ELEMENT_B: float_e4m3_t or float_e5m2_t; float
// accumulation), included by cutlass_e*_e*.cu as FUNCTION. CUTLASS's sm_89 kernels (no TMA), with another instruction
// on sm_120a: the e4m3 / e5m2 mma.sync of sm_89, like sm_120's f8f6f4 kind, runs at half the rate of the block scaled
// kind (mxf8f6f4; 85 against 170 TFLOPS on an RTX 5060), which with scale factors 1 computes the same. mma.sync takes 8
// bit operands only as A row-major and B column-major (cuBLASLt's "TN" layout, the only one it supports for them too).
// 16 byte accesses; tiles of 128 x 128 (4 stages; launched in groups of 8, which reuse the operands in L2: 8192^3 at 135
// instead of 100 TFLOPS), or 64 x 64 (6 stages) for problems with few tiles. The epilogue applies cuBLASLt's scales and
// computes the absolute maximum; it stores 16 bytes per access, 8 bytes for 8 bit D (the epilogue's thread map has no
// layout for 16 of them).
#include "cutlass_gemm.cuh"

#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/functional.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>

#include <type_traits>

namespace cutlass::arch {

/// multiply-add as mma.sync of the block scaled kind with scale factors 1 (2^0) where the GPU has it (sm_120a)
struct OpMultiplyAddUnitScale {};

template<typename TA, typename TB>
struct UnitScaleMma : Mma<gemm::GemmShape<16, 8, 32>, 32, TA, layout::RowMajor, TB, layout::ColumnMajor, float,
                          layout::RowMajor, OpMultiplyAdd> {
    using Base = Mma<gemm::GemmShape<16, 8, 32>, 32, TA, layout::RowMajor, TB, layout::ColumnMajor, float,
                     layout::RowMajor, OpMultiplyAdd>;
    using Operator = OpMultiplyAddUnitScale;

    CUTLASS_HOST_DEVICE
    void operator()(typename Base::FragmentC &d, typename Base::FragmentA const &a, typename Base::FragmentB const &b,
                    typename Base::FragmentC const &c) const {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
        const auto *A = reinterpret_cast<const uint32_t *>(&a);
        const auto *B = reinterpret_cast<const uint32_t *>(&b);
        const auto *C = reinterpret_cast<const float *>(&c);
        auto *D = reinterpret_cast<float *>(&d);
        // ue8m0 127 (2^0) in every byte, so the byte and thread selectors don't matter
#define LIBRECUDA_UNIT_SCALE_MMA(TYPES)                                                                                \
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32." TYPES ".f32.ue8m0 "         \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15,%15}, {%14}, {%15,%15};\n"              \
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])                                                               \
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]),           \
          "f"(C[3]), "r"(0x7f7f7f7fu), "h"(static_cast<uint16_t>(0)))
        constexpr bool a_e4m3 = std::is_same_v<TA, float_e4m3_t>, b_e4m3 = std::is_same_v<TB, float_e4m3_t>;
        if constexpr (a_e4m3 && b_e4m3) {
            LIBRECUDA_UNIT_SCALE_MMA("e4m3.e4m3");
        } else if constexpr (a_e4m3) {
            LIBRECUDA_UNIT_SCALE_MMA("e4m3.e5m2");
        } else if constexpr (b_e4m3) {
            LIBRECUDA_UNIT_SCALE_MMA("e5m2.e4m3");
        } else {
            LIBRECUDA_UNIT_SCALE_MMA("e5m2.e5m2");
        }
#undef LIBRECUDA_UNIT_SCALE_MMA
#else
        Base::operator()(d, a, b, c);
#endif
    }
};

// (both orders of the types: CUTLASS swaps A and B for a column-major D)
template<>
struct Mma<gemm::GemmShape<16, 8, 32>, 32, ELEMENT_A, layout::RowMajor, ELEMENT_B, layout::ColumnMajor, float,
           layout::RowMajor, OpMultiplyAddUnitScale> : UnitScaleMma<ELEMENT_A, ELEMENT_B> {};
#if !SAME_ELEMENTS
template<>
struct Mma<gemm::GemmShape<16, 8, 32>, 32, ELEMENT_B, layout::RowMajor, ELEMENT_A, layout::ColumnMajor, float,
           layout::RowMajor, OpMultiplyAddUnitScale> : UnitScaleMma<ELEMENT_B, ELEMENT_A> {};
#endif

} // namespace cutlass::arch

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
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Small ? 1 : 8>, Small ? 6 : 4, 16, 16,
    cutlass::arch::OpMultiplyAddUnitScale>;

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
