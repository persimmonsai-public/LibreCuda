/**
 * Runs a GEMM with a CUTLASS kernel. The instances (cutlass_*.cu) cover the common cases; tryCutlass* return
 * CUBLAS_STATUS_NOT_SUPPORTED for problems they can't run (alignment, layouts), which the generic kernel then runs.
 */
#pragma once

#include "blas_internal.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cstdint>

namespace librecuda_blas {

// scalars converted to the kernel's compute type: host values, or device pointers of exactly that type
template<typename Compute>
struct CutlassScalars {
    Compute alpha, beta;
    const Compute *alphaPtr, *betaPtr;
};

// whether the operand's pointers (or pointer array, which can't be checked), leading dimension and batch stride are
// multiples of alignment elements
inline bool aligned(const Operand &operand, size_t elementSize, int alignment) {
    if (operand.inc != 1 || operand.ld % alignment != 0 || operand.stride % alignment != 0) {
        return false;
    }
    if (operand.array != nullptr) {
        // the per-batch pointers are on the device; CUDA allocations are aligned, but offsets might not be
        return alignment == 1;
    }
    return reinterpret_cast<uintptr_t>(operand.ptr) % (elementSize * alignment) == 0;
}

// the parameters of the 8 bit float GEMMs' epilogue: alpha / beta as in CutlassScalars (a pointer when not null) and
// the scales and amax of GemmParams (null: 1 / none)
struct Fp8Epilogue {
    float alpha, beta;
    const float *alphaPtr, *betaPtr;
    const float *scaleA, *scaleB, *scaleC, *scaleD;
    float *amax;
};

template<typename Gemm>
cublasStatus_t runCutlass(const GemmParams &p, const typename Gemm::EpilogueOutputOp::Params &epilogue) {
    const bool array = p.a.array != nullptr;
    // without C (beta == 0), the kernel reads nothing from it: point it at D
    const void *c = p.c.ptr != nullptr ? p.c.ptr : p.d.ptr;
    const void *c_array = p.c.array != nullptr ? static_cast<const void *>(p.c.array) : p.d.array;
    const long long ldc = p.c.ptr != nullptr || p.c.array != nullptr ? p.c.ld : p.d.ld;
    const long long stride_c = p.c.ptr != nullptr || p.c.array != nullptr ? p.c.stride : p.d.stride;
    // the grid's z dimension holds the batch entries: at most 65535 per launch
    for (int first = 0; first < p.batch; first += 65535) {
        const int count = std::min(p.batch - first, 65535);
        const auto mode = array ? cutlass::gemm::GemmUniversalMode::kArray
                                : count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                                            : cutlass::gemm::GemmUniversalMode::kGemm;
        auto offset = [&](const void *ptr, long long stride, size_t size) {
            return static_cast<const void *>(static_cast<const char *>(ptr) + size_t(first) * stride * size);
        };
        const size_t a_size = sizeof(typename Gemm::ElementA), b_size = sizeof(typename Gemm::ElementB);
        const size_t c_size = sizeof(typename Gemm::ElementC);
        typename Gemm::Arguments args(
            mode, {p.m, p.n, p.k}, count, epilogue,
            array ? static_cast<const void *>(p.a.array + first) : offset(p.a.ptr, p.a.stride, a_size),
            array ? static_cast<const void *>(p.b.array + first) : offset(p.b.ptr, p.b.stride, b_size),
            array ? static_cast<const void *>(static_cast<const void *const *>(c_array) + first)
                  : offset(c, stride_c, c_size),
            array ? const_cast<void *>(static_cast<const void *>(p.d.array + first))
                  : const_cast<void *>(offset(p.d.ptr, p.d.stride, c_size)),
            p.a.stride, p.b.stride, stride_c, p.d.stride, p.a.ld, p.b.ld, ldc, p.d.ld);
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        // no instance needs a workspace (no split-k)
        if (gemm.initialize(args, nullptr, p.stream) != cutlass::Status::kSuccess ||
            gemm.run(p.stream) != cutlass::Status::kSuccess) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template<typename Gemm>
cublasStatus_t runCutlass(const GemmParams &p, const CutlassScalars<typename Gemm::EpilogueOutputOp::ElementCompute> &s) {
    using EpilogueParams = typename Gemm::EpilogueOutputOp::Params;
    return runCutlass<Gemm>(p, s.alphaPtr != nullptr ? EpilogueParams(s.alphaPtr, s.betaPtr)
                                                     : EpilogueParams(s.alpha, s.beta));
}

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

/**
 * Picks the instance for the operands' layouts (op N: column-major, op T: row-major) and the widest alignment they
 * allow. Instance<LayoutA, LayoutB, Wide> is the GEMM type.
 */
template<template<typename, typename, bool> class Instance, typename Compute>
cublasStatus_t dispatchLayouts(const GemmParams &p, const CutlassScalars<Compute> &s, bool wide) {
    const bool ta = p.transa == CUBLAS_OP_T, tb = p.transb == CUBLAS_OP_T;
    if (wide) {
        if (!ta && !tb) return runCutlass<Instance<ColumnMajor, ColumnMajor, true>>(p, s);
        if (ta && !tb) return runCutlass<Instance<RowMajor, ColumnMajor, true>>(p, s);
        if (!ta && tb) return runCutlass<Instance<ColumnMajor, RowMajor, true>>(p, s);
        return runCutlass<Instance<RowMajor, RowMajor, true>>(p, s);
    }
    if (!ta && !tb) return runCutlass<Instance<ColumnMajor, ColumnMajor, false>>(p, s);
    if (ta && !tb) return runCutlass<Instance<RowMajor, ColumnMajor, false>>(p, s);
    if (!ta && tb) return runCutlass<Instance<ColumnMajor, RowMajor, false>>(p, s);
    return runCutlass<Instance<RowMajor, RowMajor, false>>(p, s);
}

// only the wide instances (activation epilogues)
template<template<typename, typename, bool> class Instance, typename Compute>
cublasStatus_t dispatchWide(const GemmParams &p, const CutlassScalars<Compute> &s) {
    const bool ta = p.transa == CUBLAS_OP_T, tb = p.transb == CUBLAS_OP_T;
    if (!ta && !tb) return runCutlass<Instance<ColumnMajor, ColumnMajor, true>>(p, s);
    if (ta && !tb) return runCutlass<Instance<RowMajor, ColumnMajor, true>>(p, s);
    if (!ta && tb) return runCutlass<Instance<ColumnMajor, RowMajor, true>>(p, s);
    return runCutlass<Instance<RowMajor, RowMajor, true>>(p, s);
}

// the instances (each in its own file, compiled in parallel); small: 64 x 64 tiles; activation: applied in the
// epilogue (wide instances with 16 bit output)
cublasStatus_t cutlassHalf(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide,
                           bool small, Activation activation);
cublasStatus_t cutlassBFloat16(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide,
                               bool small, Activation activation);
cublasStatus_t cutlassTF32(const GemmParams &p, const CutlassScalars<float> &s, bool wide);
cublasStatus_t cutlassFloat(const GemmParams &p, const CutlassScalars<float> &s);
cublasStatus_t cutlassDouble(const GemmParams &p, const CutlassScalars<double> &s);
// 8 bit float operands (A e4m3 / e5m2 times B e4m3 / e5m2): A transposed, B not (the layout cuBLASLt supports), 16 byte
// aligned; D of 16 bit floats, float or 8 bit floats, aligned to fp8OutputAlignment(element size) elements
constexpr int fp8OutputAlignment(size_t size) { return size == 1 ? 8 : 16 / static_cast<int>(size); }
cublasStatus_t cutlassE4M3E4M3(const GemmParams &p, const Fp8Epilogue &e, bool small);
cublasStatus_t cutlassE4M3E5M2(const GemmParams &p, const Fp8Epilogue &e, bool small);
cublasStatus_t cutlassE5M2E4M3(const GemmParams &p, const Fp8Epilogue &e, bool small);
cublasStatus_t cutlassE5M2E5M2(const GemmParams &p, const Fp8Epilogue &e, bool small);

} // namespace librecuda_blas
