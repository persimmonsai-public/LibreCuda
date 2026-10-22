// CUTLASS tensor core GEMMs of 16 bit operands (ELEMENT: half or bfloat16; float accumulation, ELEMENT or float output),
// included by cutlass_half.cu / cutlass_bfloat16.cu as FUNCTION. The sm_80 kernels (mma.sync) run on newer GPUs too.
// Wide instances move 16 bytes per access; narrow ones 4 bytes.
#include "cutlass_gemm.cuh"

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/half.h>

namespace librecuda_blas {
namespace {

template<typename T, typename TD, typename LA, typename LB, bool Wide>
using TensorOp = cutlass::gemm::device::GemmUniversal<
    T, LA, T, LB, TD, ColumnMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<TD, Wide ? 128 / cutlass::sizeof_bits<TD>::value : 4 / int(sizeof(TD)),
                                                 float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, Wide ? 8 : 2, Wide ? 8 : 2>;

template<typename LA, typename LB, bool Wide>
using GemmT = TensorOp<ELEMENT, ELEMENT, LA, LB, Wide>;
template<typename LA, typename LB, bool Wide>
using GemmTFloat = TensorOp<ELEMENT, float, LA, LB, Wide>;

} // namespace

cublasStatus_t FUNCTION(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide) {
    return outputFloat ? dispatchLayouts<GemmTFloat>(p, s, wide) : dispatchLayouts<GemmT>(p, s, wide);
}

} // namespace librecuda_blas
