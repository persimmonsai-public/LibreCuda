// CUTLASS GEMMs of float operands: TF32 tensor cores (when allowed) and exact SIMT kernels; SIMT kernels for double
#include "cutlass_gemm.cuh"

#include <cutlass/epilogue/thread/linear_combination.h>

namespace librecuda_blas {
namespace {

template<typename LA, typename LB, bool Wide>
using TF32 = cutlass::gemm::device::GemmUniversal<
    float, LA, float, LB, float, ColumnMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, Wide ? 4 : 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, Wide ? 4 : 1, Wide ? 4 : 1>;

template<typename T, typename LA, typename LB>
using Simt = cutlass::gemm::device::GemmUniversal<T, LA, T, LB, T, ColumnMajor, T, cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm80>;
template<typename LA, typename LB, bool>
using SimtFloat = Simt<float, LA, LB>;
template<typename LA, typename LB, bool>
using SimtDouble = Simt<double, LA, LB>;

} // namespace

cublasStatus_t cutlassTF32(const GemmParams &p, const CutlassScalars<float> &s, bool wide) {
    return dispatchLayouts<TF32>(p, s, wide);
}

cublasStatus_t cutlassFloat(const GemmParams &p, const CutlassScalars<float> &s) {
    return dispatchLayouts<SimtFloat>(p, s, false);
}

cublasStatus_t cutlassDouble(const GemmParams &p, const CutlassScalars<double> &s) {
    return dispatchLayouts<SimtDouble>(p, s, false);
}

} // namespace librecuda_blas
