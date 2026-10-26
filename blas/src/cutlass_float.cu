// CUTLASS GEMMs of float operands: TF32 tensor cores (when allowed) and exact SIMT kernels; SIMT kernels for double
// and single precision complex operands
#include "cutlass_gemm.cuh"

#include <cutlass/complex.h>

#include <cutlass/epilogue/thread/linear_combination.h>

namespace librecuda_blas {
namespace {

// tiles of 128 x 128, or 64 x 64 (6 stages) for problems with few tiles (an RNN's per time step GEMMs: 32 columns)
template<typename LA, typename LB, bool Wide, bool Small>
using TF32Tiles = cutlass::gemm::device::GemmUniversal<
    float, LA, float, LB, float, ColumnMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    std::conditional_t<Small, cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<128, 128, 16>>,
    std::conditional_t<Small, cutlass::gemm::GemmShape<32, 32, 16>, cutlass::gemm::GemmShape<64, 64, 16>>,
    cutlass::gemm::GemmShape<16, 8, 8>, cutlass::epilogue::thread::LinearCombination<float, Wide ? 4 : 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Small ? 6 : 4, Wide ? 4 : 1, Wide ? 4 : 1>;
template<typename LA, typename LB, bool Wide>
using TF32 = TF32Tiles<LA, LB, Wide, false>;
template<typename LA, typename LB, bool Wide>
using TF32Small = TF32Tiles<LA, LB, Wide, true>;

template<typename T, typename LA, typename LB>
using Simt = cutlass::gemm::device::GemmUniversal<T, LA, T, LB, T, ColumnMajor, T, cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm80>;
template<typename LA, typename LB, bool>
using SimtFloat = Simt<float, LA, LB>;
template<typename LA, typename LB, bool>
using SimtDouble = Simt<double, LA, LB>;

/**
 * complex<float>: SIMT kernels (tiles of 64 x 32, 4 stages, as cuBLAS's cgemm uses for small problems: an FNO's
 * per-mode channel mixing, batches of hundreds of 32 x 64 x 64 products) for op N, T and C (conjugate transpose)
 */
template<typename LA, cutlass::ComplexTransform TA, typename LB, cutlass::ComplexTransform TB>
using ComplexSimt = cutlass::gemm::device::GemmUniversal<
    cutlass::complex<float>, LA, cutlass::complex<float>, LB, cutlass::complex<float>, ColumnMajor,
    cutlass::complex<float>, cutlass::arch::OpClassSimt, cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<32, 16, 16>, cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<cutlass::complex<float>, 1, cutlass::complex<float>,
                                                 cutlass::complex<float>>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4, 1, 1, cutlass::arch::OpMultiplyAddComplex, TA,
    TB>;

template<typename LA, cutlass::ComplexTransform TA>
cublasStatus_t complexB(const GemmParams &p, const CutlassScalars<cutlass::complex<float>> &s) {
    using cutlass::ComplexTransform;
    switch (p.transb) {
        case CUBLAS_OP_N: return runCutlass<ComplexSimt<LA, TA, ColumnMajor, ComplexTransform::kNone>>(p, s);
        case CUBLAS_OP_T: return runCutlass<ComplexSimt<LA, TA, RowMajor, ComplexTransform::kNone>>(p, s);
        default: return runCutlass<ComplexSimt<LA, TA, RowMajor, ComplexTransform::kConjugate>>(p, s);
    }
}

} // namespace

cublasStatus_t cutlassComplexFloat(const GemmParams &p, const CutlassScalars<cutlass::complex<float>> &s) {
    using cutlass::ComplexTransform;
    switch (p.transa) {
        case CUBLAS_OP_N: return complexB<ColumnMajor, ComplexTransform::kNone>(p, s);
        case CUBLAS_OP_T: return complexB<RowMajor, ComplexTransform::kNone>(p, s);
        default: return complexB<RowMajor, ComplexTransform::kConjugate>(p, s);
    }
}

cublasStatus_t cutlassTF32(const GemmParams &p, const CutlassScalars<float> &s, bool wide, bool small) {
    return small ? dispatchLayouts<TF32Small>(p, s, wide) : dispatchLayouts<TF32>(p, s, wide);
}

cublasStatus_t cutlassFloat(const GemmParams &p, const CutlassScalars<float> &s) {
    return dispatchLayouts<SimtFloat>(p, s, false);
}

cublasStatus_t cutlassDouble(const GemmParams &p, const CutlassScalars<double> &s) {
    return dispatchLayouts<SimtDouble>(p, s, false);
}

} // namespace librecuda_blas
