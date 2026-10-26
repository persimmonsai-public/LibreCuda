// NVCC_FLAGS: -arch=$ARCH -lcuda
// The driver API's libraries (cuLibrary*, cuKernel*), which CUDA Python's cuda.core (and so numba-cuda) loads code
// with: a library loaded from PTX before any context is current, its kernels enumerated, named, queried and launched
// through their CUkernel handles (which CUDA lets stand for functions), a global read back, the module and function of
// the current context. Checked against NVIDIA's driver too.
#include "check.h"
#include <cuda.h>
#include <cstring>

#define CU(call)                                                                                                       \
    do {                                                                                                               \
        CUresult r_ = (call);                                                                                          \
        EXPECT(r_ == CUDA_SUCCESS, "%s failed: %d", #call, int(r_));                                                   \
    } while (0)

static const char PTX[] = R"(
.version 8.0
.target sm_80
.address_size 64

.visible .global .align 4 .u32 launches;

.visible .entry add_one(.param .u64 p)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<5>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [%rd2];
    add.s32 %r2, %r1, 1;
    st.global.u32 [%rd2], %r2;
    ld.global.u32 %r3, [launches];
    add.s32 %r4, %r3, 1;
    st.global.u32 [launches], %r4;
    ret;
}

.visible .entry nothing()
{
    ret;
}
)";

int main() {
    CU(cuInit(0));
    // no context is current yet: libraries don't need one
    CUlibrary library = nullptr;
    CU(cuLibraryLoadData(&library, PTX, nullptr, nullptr, 0, nullptr, nullptr, 0));
    unsigned int count = 0;
    CU(cuLibraryGetKernelCount(&count, library));
    EXPECT(count == 2, "%u kernels", count);
    CUkernel kernels[2] = {};
    CU(cuLibraryEnumerateKernels(kernels, 2, library));
    CUkernel addOne = nullptr;
    CU(cuLibraryGetKernel(&addOne, library, "add_one"));
    EXPECT(addOne == kernels[0] || addOne == kernels[1], "cuLibraryGetKernel's kernel isn't enumerated");
    const char *name = nullptr;
    CU(cuKernelGetName(&name, addOne));
    EXPECT(name != nullptr && strcmp(name, "add_one") == 0, "kernel name %s", name ? name : "(null)");
    CUlibrary owner = nullptr;
    CU(cuKernelGetLibrary(&owner, addOne));
    EXPECT(owner == library, "cuKernelGetLibrary");
    int maxThreads = 0;
    CU(cuKernelGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, addOne, 0));
    EXPECT(maxThreads > 0, "max threads per block %d", maxThreads);

    // launches in a context: the kernel handle as a function, and the context's function
    CUdevice device;
    CU(cuDeviceGet(&device, 0));
    CUcontext ctx;
    CU(cuDevicePrimaryCtxRetain(&ctx, device));
    CU(cuCtxSetCurrent(ctx));
    // (NVIDIA's driver wants a context for parameter information)
    size_t offset = 1, size = 0;
    CU(cuKernelGetParamInfo(addOne, 0, &offset, &size));
    EXPECT(offset == 0 && size == 8, "parameter 0 at %zu, %zu bytes", offset, size);
    CUdeviceptr value;
    CU(cuMemAlloc(&value, 4));
    CU(cuMemsetD32(value, 41, 1));
    void *args[] = {&value};
    CU(cuLaunchKernel(reinterpret_cast<CUfunction>(addOne), 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CUfunction function = nullptr;
    CU(cuKernelGetFunction(&function, addOne));
    CU(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CU(cuCtxSynchronize());
    unsigned int result = 0;
    CU(cuMemcpyDtoH(&result, value, 4));
    EXPECT(result == 43, "value %u after two launches", result);
    CUdeviceptr launches;
    size_t bytes = 0;
    CU(cuLibraryGetGlobal(&launches, &bytes, library, "launches"));
    unsigned int counted = 0;
    CU(cuMemcpyDtoH(&counted, launches, 4));
    EXPECT(bytes == 4 && counted == 2, "global: %zu bytes, %u launches", bytes, counted);
    CUmodule module = nullptr;
    CU(cuLibraryGetModule(&module, library));
    CUfunction fromModule = nullptr;
    CU(cuModuleGetFunction(&fromModule, module, "nothing"));
    CU(cuLaunchKernel(fromModule, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
    CU(cuCtxSynchronize());

    // and through cuGetProcAddress, as CUDA Python finds them
    void *pfn = nullptr;
    CUdriverProcAddressQueryResult status{};
    CU(cuGetProcAddress("cuLibraryLoadData", &pfn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(pfn != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS, "cuLibraryLoadData not found");

    CU(cuMemFree(value));
    CU(cuLibraryUnload(library));
    CU(cuDevicePrimaryCtxRelease(device));
    PASS("driver_library");
    return 0;
}
