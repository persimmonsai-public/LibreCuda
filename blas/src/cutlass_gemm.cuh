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

template<typename Gemm>
cublasStatus_t runCutlass(const GemmParams &p, const CutlassScalars<typename Gemm::EpilogueOutputOp::ElementCompute> &s) {
    using EpilogueParams = typename Gemm::EpilogueOutputOp::Params;
    const EpilogueParams epilogue = s.alphaPtr != nullptr ? EpilogueParams(s.alphaPtr, s.betaPtr)
                                                          : EpilogueParams(s.alpha, s.beta);
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

// the instances (each in its own file, compiled in parallel)
cublasStatus_t cutlassHalf(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide);
cublasStatus_t cutlassBFloat16(const GemmParams &p, const CutlassScalars<float> &s, bool outputFloat, bool wide);
cublasStatus_t cutlassTF32(const GemmParams &p, const CutlassScalars<float> &s, bool wide);
cublasStatus_t cutlassFloat(const GemmParams &p, const CutlassScalars<float> &s);
cublasStatus_t cutlassDouble(const GemmParams &p, const CutlassScalars<double> &s);

} // namespace librecuda_blas
