// CUTLASS tensor core GEMMs of 16 bit operands (ELEMENT: half or bfloat16; float accumulation, ELEMENT or float output),
// included by cutlass_half.cu / cutlass_bfloat16.cu as FUNCTION. The sm_80 kernels (mma.sync) run on newer GPUs too.
// Wide instances move 16 bytes per access; narrow ones 4 bytes. Tiles of 128 x 128 (4 stages), or 64 x 64 (6 stages)
// for problems with few tiles. Wide instances with ELEMENT output also apply ReLU or GELU (tanh approximation) in the
// epilogue.
#include "cutlass_gemm.cuh"

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/half.h>

#include <type_traits>

namespace librecuda_blas {
namespace {

// D = activation(alpha * accumulator + beta * C): ACT 0 none, 1 ReLU, 2 GELU
template<typename TD, int COUNT, int ACT>
using Epilogue = std::conditional_t<
    ACT == 1, cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::ReLu, TD, COUNT, float, float>,
    std::conditional_t<ACT == 2,
                       cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor, TD,
                                                                           COUNT, float, float>,
                       cutlass::epilogue::thread::LinearCombination<TD, COUNT, float, float>>>;

template<typename T, typename TD, typename LA, typename LB, bool Wide, bool Small, int Act>
using TensorOp = cutlass::gemm::device::GemmUniversal<
    T, LA, T, LB, TD, ColumnMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    std::conditional_t<Small, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<128, 128, 32>>,
    std::conditional_t<Small, cutlass::gemm::GemmShape<32, 32, 32>, cutlass::gemm::GemmShape<64, 64, 32>>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<TD, Wide ? 128 / cutlass::sizeof_bits<TD>::value : 4 / int(sizeof(TD)), Act>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Small ? 6 : 4, Wide ? 8 : 2, Wide ? 8 : 2>;

template<bool Small, int Act>
struct Instances {
    template<typename LA, typename LB, bool Wide>
    using Same = TensorOp<ELEMENT, ELEMENT, LA, LB, Wide, Small, Act>;
    template<typename LA, typename LB, bool Wide>
    using Float = TensorOp<ELEMENT, float, LA, LB, Wide, Small, 0>;
};

template<bool Small>
cublasStatus_t withActivation(const GemmParams &p, const CutlassScalars<float> &s, bool wide, Activation activation) {
    switch (activation) {
        case Activation::RELU:
            return wide ? dispatchWide<Instances<Small, 1>::template Same>(p, s) : CUBLAS_STATUS_NOT_SUPPORTED;
        case Activation::GELU:
            return wide ? dispatchWide<Instances<Small, 2>::template Same>(p, s) : CUBLAS_STATUS_NOT_SUPPORTED;
        default:
            return dispatchLayouts<Instances<Small, 0>::template Same>(p, s, wide);
    }
}

} // namespace

cublasStatus_t FUNCTION(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide, bool small,
                        Activation activation) {
    if (outputFloat) {
        if (activation != Activation::NONE) return CUBLAS_STATUS_NOT_SUPPORTED;
        return small ? dispatchLayouts<Instances<true, 0>::template Float>(p, s, wide)
                     : dispatchLayouts<Instances<false, 0>::template Float>(p, s, wide);
    }
    return small ? withActivation<true>(p, s, wide, activation) : withActivation<false>(p, s, wide, activation);
}

} // namespace librecuda_blas
