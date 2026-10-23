// CUDA prelude for clang's HIP mode targeting SPIR-V (-x hip --offload=spirv64 -nogpuinc -nogpulib): the
// qualifiers, threadIdx / blockIdx / blockDim / gridDim and __syncthreads, on the OpenCL built-in functions that
// LLVM's SPIR-V backend turns into SPIR-V built-in variables and OpControlBarrier (by their mangled names).
#pragma once

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))

extern "C" __device__ unsigned long _Z12get_local_idj(unsigned);
extern "C" __device__ unsigned long _Z12get_group_idj(unsigned);
extern "C" __device__ unsigned long _Z14get_local_sizej(unsigned);
extern "C" __device__ unsigned long _Z14get_num_groupsj(unsigned);
extern "C" __device__ void _Z7barrierj(unsigned);

#define LIBRECUDA_BUILTIN_VAR(type, var, fn)                                                  \
    struct type {                                                                             \
        static __device__ __forceinline__ unsigned get_x() { return unsigned(fn(0)); }        \
        static __device__ __forceinline__ unsigned get_y() { return unsigned(fn(1)); }        \
        static __device__ __forceinline__ unsigned get_z() { return unsigned(fn(2)); }        \
        __declspec(property(get = get_x)) unsigned x;                                         \
        __declspec(property(get = get_y)) unsigned y;                                         \
        __declspec(property(get = get_z)) unsigned z;                                         \
    };                                                                                        \
    extern const __device__ type var;

LIBRECUDA_BUILTIN_VAR(__librecuda_threadIdx_t, threadIdx, _Z12get_local_idj)
LIBRECUDA_BUILTIN_VAR(__librecuda_blockIdx_t, blockIdx, _Z12get_group_idj)
LIBRECUDA_BUILTIN_VAR(__librecuda_blockDim_t, blockDim, _Z14get_local_sizej)
LIBRECUDA_BUILTIN_VAR(__librecuda_gridDim_t, gridDim, _Z14get_num_groupsj)

// barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)
#define __syncthreads() _Z7barrierj(3)
