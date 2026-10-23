// LibreCUDA's NCCL: the kernels of the device transport (nccl_device.cu)
#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstddef>

// the reductions of the kernels (average and pre-multiplied sums are sums with a divisor / scale)
enum NcclDeviceOp {
    NCCL_DEVICE_SUM, NCCL_DEVICE_PROD, NCCL_DEVICE_MAX, NCCL_DEVICE_MIN
};

/**
 * out1 / out2 (either may be nullptr) [i < count] = combine(a[i] * scale (if scaled), b[i]) / divide (if > 1): with a
 * only, a copy (scaled); with b only, b's copy; with both, op(a, b)
 */
struct NcclCombine {
    ncclDataType_t type;
    int op;
    size_t count;
    const void *a, *b;
    void *out1, *out2;
    bool scaled;
    double scale;
    int divide;
};

cudaError_t ncclDeviceCombine(const NcclCombine &combine, cudaStream_t stream);
