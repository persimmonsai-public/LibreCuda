// NVCC_FLAGS: -arch=$ARCH -lcuda
// Driver API functions whose signature changed between CUDA versions: cuGetProcAddress returns the version of the
// CUDA version asked for (a caller built with CUDA 12.0 gets cuMemAdvise taking a device, CUDA 12.2's takes a
// location), and the earlier versions are exported by their old names for programs linked with older headers.
// Checked against NVIDIA's driver too.
#include "check.h"
#include <cuda.h>
#include <dlfcn.h>

#define CU(call)                                                                                                       \
    do {                                                                                                               \
        CUresult r_ = (call);                                                                                          \
        EXPECT(r_ == CUDA_SUCCESS, "%s failed: %d", #call, int(r_));                                                   \
    } while (0)

typedef CUresult (*CtxCreate3020)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*CtxCreate11040)(CUcontext *, CUexecAffinityParam *, int, unsigned int, CUdevice);
typedef CUresult (*MemAdvise8000)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
typedef CUresult (*MemPrefetch8000)(CUdeviceptr, size_t, CUdevice, CUstream);
typedef CUresult (*StreamGetCtx9020)(CUstream, CUcontext *);
typedef CUresult (*CtxGetDevice13000)(CUdevice *, CUcontext);
typedef CUresult (*GraphInstantiate11000)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
typedef CUresult (*GetProcAddress11030)(const char *, void **, int, cuuint64_t);
typedef CUresult (*ElapsedTime2000)(float *, CUevent, CUevent);

template<typename F>
F lookup(const char *name, int version) {
    void *pfn = nullptr;
    CUdriverProcAddressQueryResult status{};
    CU(cuGetProcAddress(name, &pfn, version, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(pfn != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS, "%s for CUDA %d not found", name, version);
    return reinterpret_cast<F>(pfn);
}

int main() {
    CU(cuInit(0));
    CUdevice device;
    CU(cuDeviceGet(&device, 0));

    // contexts: CUDA 3.2's and 11.4's cuCtxCreate
    for (int version: {3020, 11040}) {
        CUcontext ctx = nullptr;
        if (version == 3020) {
            CU(lookup<CtxCreate3020>("cuCtxCreate", version)(&ctx, 0, device));
        } else {
            CU(lookup<CtxCreate11040>("cuCtxCreate", version)(&ctx, nullptr, 0, 0, device));
        }
        CUdevice current = -1;
        CU(lookup<CtxGetDevice13000>("cuCtxGetDevice", 13000)(&current, ctx));
        EXPECT(current == device, "cuCtxCreate for CUDA %d: context of device %d", version, current);
        CU(cuCtxDestroy(ctx));
    }
    CUcontext primary;
    CU(cuDevicePrimaryCtxRetain(&primary, device));
    CU(cuCtxSetCurrent(primary));

    // memory hints with a device (CUDA 12.0): managed memory prefetched to the device and to the host
    CUdeviceptr managed;
    CU(cuMemAllocManaged(&managed, 1 << 20, CU_MEM_ATTACH_GLOBAL));
    CU(lookup<MemAdvise8000>("cuMemAdvise", 12000)(managed, 1 << 20, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device));
    CU(lookup<MemPrefetch8000>("cuMemPrefetchAsync", 12000)(managed, 1 << 20, device, nullptr));
    CU(lookup<MemPrefetch8000>("cuMemPrefetchAsync", 12000)(managed, 1 << 20, CU_DEVICE_CPU, nullptr));
    CU(cuCtxSynchronize());
    int preferred = -7;
    CU(cuMemRangeGetAttribute(&preferred, sizeof(preferred), CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, managed,
                              1 << 20));
    EXPECT(preferred == device, "preferred location %d after cuMemAdvise for CUDA 12.0", preferred);
    // and the exported old names
    void *driver = dlopen("libcuda.so.1", RTLD_NOW);
    auto advise = reinterpret_cast<MemAdvise8000>(dlsym(driver, "cuMemAdvise"));
    EXPECT(advise != nullptr, "cuMemAdvise not exported");
    CU(advise(managed, 1 << 20, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, device));
    CU(cuMemFree(managed));

    // a stream's context (CUDA 9.2: without the green context)
    CUstream stream;
    CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUcontext streamCtx = nullptr;
    CU(lookup<StreamGetCtx9020>("cuStreamGetCtx", 12000)(stream, &streamCtx));
    EXPECT(streamCtx == primary, "cuStreamGetCtx for CUDA 12.0: %p", static_cast<void *>(streamCtx));

    // graphs instantiated as with CUDA 11 (an error node and a log buffer)
    CUgraph graph;
    CU(cuGraphCreate(&graph, 0));
    CUgraphExec exec = nullptr;
    CUgraphNode errorNode = reinterpret_cast<CUgraphNode>(1);
    char log[16] = "x";
    CU(lookup<GraphInstantiate11000>("cuGraphInstantiate", 11000)(&exec, graph, &errorNode, log, sizeof(log)));
    EXPECT(exec != nullptr && errorNode == nullptr && log[0] == 0, "cuGraphInstantiate for CUDA 11");
    CU(cuGraphLaunch(exec, stream));
    CU(cuStreamSynchronize(stream));
    CU(cuGraphExecDestroy(exec));
    CU(cuGraphDestroy(graph));

    // CUDA 11.3's cuGetProcAddress (no status) and CUDA 12's cuEventElapsedTime, by their exported names
    auto getProc = reinterpret_cast<GetProcAddress11030>(dlsym(driver, "cuGetProcAddress"));
    auto elapsed = reinterpret_cast<ElapsedTime2000>(dlsym(driver, "cuEventElapsedTime"));
    EXPECT(getProc != nullptr && elapsed != nullptr, "cuGetProcAddress / cuEventElapsedTime not exported");
    void *pfn = nullptr;
    CU(getProc("cuMemAdvise", &pfn, 12000, CU_GET_PROC_ADDRESS_DEFAULT));
    EXPECT(pfn == reinterpret_cast<void *>(advise) || pfn != nullptr, "cuGetProcAddress (CUDA 11.3) found nothing");
    CUevent start, end;
    CU(cuEventCreate(&start, 0));
    CU(cuEventCreate(&end, 0));
    CU(cuEventRecord(start, stream));
    CU(cuEventRecord(end, stream));
    CU(cuEventSynchronize(end));
    float ms = -1;
    CU(elapsed(&ms, start, end));
    EXPECT(ms >= 0, "cuEventElapsedTime: %f", ms);

    // a version older than a function's first one finds nothing (successfully, as NVIDIA's driver does)
    void *none = reinterpret_cast<void *>(1);
    CUdriverProcAddressQueryResult status{};
    const CUresult result = cuGetProcAddress("cuMemAdvise", &none, 7000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    EXPECT(result == CUDA_SUCCESS && none == nullptr && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT,
           "cuMemAdvise for CUDA 7.0: %d, status %d", int(result), int(status));

    CU(cuEventDestroy(start));
    CU(cuEventDestroy(end));
    CU(cuStreamDestroy(stream));
    CU(cuDevicePrimaryCtxRelease(device));
    PASS("driver_api_versions");
    return 0;
}
