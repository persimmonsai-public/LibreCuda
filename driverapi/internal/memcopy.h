#pragma once

#include <librecuda.h>
#include <cstddef>
#include <cstdint>

libreCudaStatus_t loadMemcpyKernelsIfNeeded(LibreCUdevice device);

/**
 * Forgets the kernels of the device; called when its context is destroyed (which frees the module memory)
 */
void releaseMemcpyKernels(LibreCUdevice device);

/**
 * @return true if the copy / fill kernels for device memory are available for the current context's device
 */
bool memcpyD2DKernelsAvailable();

bool memsetKernelsAvailable();

/**
 * Copies between device memory allocations with kernels (faster than the copy engine for device memory)
 */
libreCudaStatus_t memcpyD2D(void *dst, const void *src, size_t size, LibreCUstream stream, bool async);

/**
 * Fills device memory with a byte value using kernels
 */
libreCudaStatus_t memsetD8Kernel(void *dst, uint8_t value, size_t size, LibreCUstream stream, bool async);
