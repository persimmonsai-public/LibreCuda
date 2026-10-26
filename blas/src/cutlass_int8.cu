// CUTLASS tensor core GEMMs of 8 bit integers (cuBLASLt's int8 matmuls, as bitsandbytes' LLM.int8 layers run them):
// int8 A (along k: op(A) = A^T) and B (along k), int32 accumulation and output, int32 alpha and beta. The sm_80
// kernels (mma.sync m16n8k32) run on newer GPUs too. Tiles of 128 x 128, or 64 x 64 for problems with few tiles.
#include "cutlass_gemm.cuh"

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>

namespace librecuda_blas {
namespace {

template<bool Small>
using Int8Gemm = cutlass::gemm::device::GemmUniversal<
    int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, int32_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    std::conditional_t<Small, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<128, 128, 64>>,
    std::conditional_t<Small, cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<64, 64, 64>>,
    cutlass::gemm::GemmShape<16, 8, 32>, cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Small ? 4 : 3, 16, 16>;

template<bool Small>
cublasStatus_t run(const GemmParams &p, const CutlassScalars<int32_t> &s) {
    using Gemm = Int8Gemm<Small>;
    const auto epilogue = s.alphaPtr != nullptr ? typename Gemm::EpilogueOutputOp::Params(s.alphaPtr, s.betaPtr)
                                                : typename Gemm::EpilogueOutputOp::Params(s.alpha, s.beta);
    return runCutlass<Gemm>(p, epilogue);
}

} // namespace

cublasStatus_t cutlassInt8(const GemmParams &p, const CutlassScalars<int32_t> &s, bool small) {
    if (p.transa == CUBLAS_OP_N || p.transb != CUBLAS_OP_N || p.d.type != CUDA_R_32I ||
        (p.c.ptr != nullptr && p.c.type != CUDA_R_32I) || !aligned(p.a, 1, 16) || !aligned(p.b, 1, 16) ||
        !aligned(p.d, 4, 4) || ((p.c.ptr != nullptr || p.c.array != nullptr) && !aligned(p.c, 4, 4))) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return small ? run<true>(p, s) : run<false>(p, s);
}

} // namespace librecuda_blas
