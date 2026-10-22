// NVCC_FLAGS: -arch=$ARCH -lcuda
// The CUDA driver API (libcuda) together with the runtime: devices and their attributes, primary contexts (the
// runtime's current device), the context stack, memory, pointer attributes, fills, modules loaded from PTX and from a
// linked cubin, launches (kernel parameters and a parameter buffer, cuLaunchKernelEx), streams and events shared with
// the runtime, stream capture of driver launches, entry points (cuGetProcAddress, cudaGetDriverEntryPointByVersion),
// error names and the runtime's default memory pool. Driver API errors leave cudaGetLastError alone.
// Checked against NVIDIA's driver and runtime too.
#include "check.h"
#include <cuda.h>
#include <cstring>
#include <thread>
#include <vector>

#define DCHECK(call) do { \
    CUresult dcheck_result = (call); \
    if (dcheck_result != CUDA_SUCCESS) { \
        const char *dcheck_name = nullptr; \
        cuGetErrorName(dcheck_result, &dcheck_name); \
        fprintf(stderr, "driver error %d (%s) at %s:%d: %s\n", int(dcheck_result), dcheck_name, __FILE__, __LINE__, \
                #call); \
        exit(1); \
    } \
} while (0)

// out[i] = value + i for i < n; counts the threads that wrote in the global counter
static const char *PTX = R"(
.version 7.0
.target sm_75
.address_size 64

.visible .global .align 4 .u32 counter;

.visible .entry fill(
    .param .u64 out,
    .param .u32 value,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .b32 %r<9>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [value];
    ld.param.u32 %r2, [n];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.s32 %p, %r6, %r2;
    @%p bra DONE;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.s32 %rd3, %r6, 4;
    add.s64 %rd2, %rd2, %rd3;
    add.s32 %r7, %r1, %r6;
    st.global.u32 [%rd2], %r7;
    atom.global.add.u32 %r8, [counter], 1;
DONE:
    ret;
}
)";

static void expectFilled(CUdeviceptr ptr, int value, int n) {
    std::vector<int> host(n);
    DCHECK(cuMemcpyDtoH(host.data(), ptr, n * sizeof(int)));
    for (int i = 0; i < n; i++) {
        EXPECT(host[i] == value + i, "out[%d] = %d, expected %d", i, host[i], value + i);
    }
}

int main() {
    // ---- devices
    DCHECK(cuInit(0));
    int driver_version = 0, count = 0, runtime_count = 0;
    DCHECK(cuDriverGetVersion(&driver_version));
    DCHECK(cuDeviceGetCount(&count));
    CHECK(cudaGetDeviceCount(&runtime_count));
    EXPECT(driver_version >= 13000 && count == runtime_count && count > 0, "version %d, %d / %d devices",
           driver_version, count, runtime_count);
    CUdevice dev;
    DCHECK(cuDeviceGet(&dev, 0));
    EXPECT(cuDeviceGet(&dev, count) == CUDA_ERROR_INVALID_DEVICE, "invalid ordinal accepted");
    DCHECK(cuDeviceGet(&dev, 0));
    char name[256];
    DCHECK(cuDeviceGetName(name, sizeof(name), dev));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    EXPECT(strcmp(name, prop.name) == 0, "name %s / %s", name, prop.name);
    int sms = 0, major = 0, minor = 0;
    DCHECK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    DCHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    DCHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    EXPECT(sms == prop.multiProcessorCount && major == prop.major && minor == prop.minor, "attributes %d %d.%d",
           sms, major, minor);
    size_t total = 0;
    DCHECK(cuDeviceTotalMem(&total, dev));
    EXPECT(total == prop.totalGlobalMem, "total memory %zu / %zu", total, prop.totalGlobalMem);
    CUuuid uuid;
    DCHECK(cuDeviceGetUuid(&uuid, dev));
    EXPECT(memcmp(&uuid, &prop.uuid, sizeof(uuid)) == 0, "uuid differs");
    char bus_id[32];
    DCHECK(cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev));
    CUdevice by_bus = -1;
    DCHECK(cuDeviceGetByPCIBusId(&by_bus, bus_id));
    EXPECT(by_bus == dev, "device of %s: %d", bus_id, by_bus);
    PASS("devices");

    // ---- contexts: the runtime binds the primary context of its device; threads start without a context
    CUcontext current = reinterpret_cast<CUcontext>(1);
    DCHECK(cuCtxGetCurrent(&current));
    EXPECT(current == nullptr, "a context before the runtime was used");
    CUdeviceptr ptr;
    EXPECT(cuMemAlloc(&ptr, 64) == CUDA_ERROR_INVALID_CONTEXT, "allocation without a context");
    CHECK(cudaFree(nullptr));
    CUcontext primary;
    DCHECK(cuDevicePrimaryCtxRetain(&primary, dev));
    DCHECK(cuCtxGetCurrent(&current));
    EXPECT(current == primary, "the runtime's context %p is not the primary context %p", (void *) current,
           (void *) primary);
    unsigned int flags = 0;
    int active = 0;
    DCHECK(cuDevicePrimaryCtxGetState(dev, &flags, &active));
    EXPECT(active == 1, "primary context inactive");
    CUdevice ctx_device = -1;
    DCHECK(cuCtxGetDevice(&ctx_device));
    EXPECT(ctx_device == 0, "context device %d", ctx_device);
    std::thread([&] {
        CUcontext thread_ctx = reinterpret_cast<CUcontext>(1);
        DCHECK(cuCtxGetCurrent(&thread_ctx));
        EXPECT(thread_ctx == nullptr, "a new thread has a context");
        DCHECK(cuCtxSetCurrent(primary));
        int runtime_device = -1;
        CHECK(cudaGetDevice(&runtime_device));
        EXPECT(runtime_device == 0, "runtime device %d", runtime_device);
        // the runtime works in the thread's context
        void *p = nullptr;
        CHECK(cudaMalloc(&p, 256));
        CHECK(cudaFree(p));
    }).join();
    // a created context: current until popped
    CUcontext created;
    DCHECK(cuCtxCreate(&created, nullptr, 0, dev));
    DCHECK(cuCtxGetCurrent(&current));
    EXPECT(current == created, "created context not current");
    DCHECK(cuMemAlloc(&ptr, 64));
    DCHECK(cuMemFree(ptr));
    CUcontext popped;
    DCHECK(cuCtxPopCurrent(&popped));
    DCHECK(cuCtxGetCurrent(&current));
    EXPECT(popped == created && current == primary, "context stack: popped %p, current %p", (void *) popped,
           (void *) current);
    DCHECK(cuCtxDestroy(created));
    DCHECK(cuCtxPushCurrent(primary));
    DCHECK(cuCtxPopCurrent(&popped));
    EXPECT(popped == primary, "pushed primary context");
    size_t stack = 0;
    DCHECK(cuCtxGetLimit(&stack, CU_LIMIT_STACK_SIZE));
    size_t runtime_stack = 0;
    CHECK(cudaDeviceGetLimit(&runtime_stack, cudaLimitStackSize));
    EXPECT(stack == runtime_stack, "stack limit %zu / %zu", stack, runtime_stack);
    PASS("contexts");

    // ---- memory, fills and pointer attributes; driver errors don't reach cudaGetLastError
    const int n = 1000;
    DCHECK(cuMemAlloc(&ptr, n * sizeof(int)));
    EXPECT(cuMemAlloc(nullptr, 16) == CUDA_ERROR_INVALID_VALUE, "null allocation pointer");
    EXPECT(cuMemcpyHtoD(ptr, nullptr, 1ull << 40) != CUDA_SUCCESS, "oversized copy");
    CHECK(cudaGetLastError());
    DCHECK(cuMemsetD32(ptr, 0x12345678u, n));
    std::vector<unsigned int> words(n);
    CHECK(cudaMemcpy(words.data(), reinterpret_cast<void *>(ptr), n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        EXPECT(words[i] == 0x12345678u, "D32 fill [%d] = %x", i, words[i]);
    }
    DCHECK(cuMemsetD16(ptr + 4, 0xabcd, 3));
    DCHECK(cuMemsetD8(ptr + 12, 0x5a, 2));
    DCHECK(cuMemcpyDtoH(words.data(), ptr, 4 * sizeof(int)));
    EXPECT(words[0] == 0x12345678u && words[1] == 0xabcdabcdu && words[2] == 0x1234abcdu &&
           words[3] == 0x12345a5au, "D16 / D8 fills: %x %x %x %x", words[0], words[1], words[2], words[3]);
    CUdeviceptr base = 0;
    size_t size = 0;
    DCHECK(cuMemGetAddressRange(&base, &size, ptr + 100));
    EXPECT(base == ptr && size >= n * sizeof(int), "range %llx + %zu of %llx", base, size, ptr);
    unsigned int memory_type = 0;
    CUcontext owner = nullptr;
    int ordinal = -1;
    DCHECK(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr));
    DCHECK(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, ptr + 8));
    DCHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr));
    EXPECT(memory_type == CU_MEMORYTYPE_DEVICE && owner == primary && ordinal == 0, "pointer attributes %u %p %d",
           memory_type, (void *) owner, ordinal);
    int host_stack_value = 0;
    EXPECT(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                 reinterpret_cast<CUdeviceptr>(&host_stack_value)) == CUDA_ERROR_INVALID_VALUE,
           "attributes of pageable memory");
    CUpointer_attribute attributes[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_IS_MANAGED};
    unsigned int type_value = 7;
    int managed_value = 7;
    void *values[2] = {&type_value, &managed_value};
    DCHECK(cuPointerGetAttributes(2, attributes, values, reinterpret_cast<CUdeviceptr>(&host_stack_value)));
    EXPECT(type_value == 0 && managed_value == 0, "attributes of unknown memory %u %d", type_value, managed_value);
    void *pinned = nullptr;
    DCHECK(cuMemAllocHost(&pinned, 4096));
    DCHECK(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                 reinterpret_cast<CUdeviceptr>(pinned)));
    EXPECT(memory_type == CU_MEMORYTYPE_HOST, "pinned memory type %u", memory_type);
    DCHECK(cuMemFreeHost(pinned));
    CHECK(cudaGetLastError());
    PASS("memory");

    // ---- modules from PTX, launches
    CUmodule module;
    DCHECK(cuModuleLoadData(&module, PTX));
    CUfunction fill;
    DCHECK(cuModuleGetFunction(&fill, module, "fill"));
    CUfunction missing;
    EXPECT(cuModuleGetFunction(&missing, module, "nothing") == CUDA_ERROR_NOT_FOUND, "missing function found");
    CUdeviceptr counter = 0;
    size_t counter_size = 0;
    DCHECK(cuModuleGetGlobal(&counter, &counter_size, module, "counter"));
    EXPECT(counter != 0 && counter_size == 4, "global %llx size %zu", counter, counter_size);
    DCHECK(cuMemsetD32(counter, 0, 1));
    int threads = 0, regs = -1;
    DCHECK(cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fill));
    DCHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fill));
    EXPECT(threads > 0 && regs > 0, "function attributes: %d threads, %d registers", threads, regs);
    const char *function_name = nullptr;
    DCHECK(cuFuncGetName(&function_name, fill));
    EXPECT(function_name != nullptr && strcmp(function_name, "fill") == 0, "function name");
    int blocks_per_sm = 0;
    DCHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fill, 256, 0));
    int min_grid = 0, block_size = 0;
    DCHECK(cuOccupancyMaxPotentialBlockSize(&min_grid, &block_size, fill, nullptr, 0, 0));
    EXPECT(blocks_per_sm > 0 && block_size > 0 && min_grid > 0, "occupancy %d, block size %d, grid %d",
           blocks_per_sm, block_size, min_grid);

    int value = 100, count_param = n;
    void *params[] = {&ptr, &value, &count_param};
    DCHECK(cuLaunchKernel(fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
    DCHECK(cuCtxSynchronize());
    expectFilled(ptr, 100, n);
    // a parameter buffer
    struct {
        CUdeviceptr out;
        int value;
        int n;
    } buffer{ptr, 200, n};
    size_t buffer_size = sizeof(buffer);
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &buffer, CU_LAUNCH_PARAM_BUFFER_SIZE, &buffer_size,
                     CU_LAUNCH_PARAM_END};
    DCHECK(cuLaunchKernel(fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, nullptr, extra));
    expectFilled(ptr, 200, n);
    // cuLaunchKernelEx on a runtime stream
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUlaunchConfig config{};
    config.gridDimX = (n + 127) / 128;
    config.gridDimY = config.gridDimZ = 1;
    config.blockDimX = 128;
    config.blockDimY = config.blockDimZ = 1;
    config.hStream = stream;
    value = 300;
    DCHECK(cuLaunchKernelEx(&config, fill, params, nullptr));
    CHECK(cudaStreamSynchronize(stream));
    expectFilled(ptr, 300, n);
    unsigned int counted = 0;
    DCHECK(cuMemcpyDtoH(&counted, counter, sizeof(counted)));
    EXPECT(counted == 3u * n, "counter %u", counted);
    PASS("modules and launches");

    // ---- a cubin from the linker, loaded as a module
    CUlinkState link;
    DCHECK(cuLinkCreate(0, nullptr, nullptr, &link));
    DCHECK(cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char *>(PTX), strlen(PTX) + 1, "fill.ptx", 0, nullptr,
                         nullptr));
    void *cubin = nullptr;
    size_t cubin_size = 0;
    DCHECK(cuLinkComplete(link, &cubin, &cubin_size));
    EXPECT(cubin_size > 4 && memcmp(cubin, "\x7f" "ELF", 4) == 0, "linked image of %zu bytes", cubin_size);
    CUmodule linked;
    DCHECK(cuModuleLoadData(&linked, cubin));
    DCHECK(cuLinkDestroy(link));
    CUfunction linked_fill;
    DCHECK(cuModuleGetFunction(&linked_fill, linked, "fill"));
    value = 400;
    DCHECK(cuLaunchKernel(linked_fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
    expectFilled(ptr, 400, n);
    DCHECK(cuModuleUnload(linked));
    PASS("linked cubin");

    // ---- streams and events shared with the runtime
    CUstream driver_stream;
    DCHECK(cuStreamCreate(&driver_stream, CU_STREAM_NON_BLOCKING));
    CUevent start, stop;
    DCHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cudaEventCreate(&stop));
    DCHECK(cuEventRecord(start, driver_stream));
    value = 500;
    DCHECK(cuLaunchKernel(fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, driver_stream, params, nullptr));
    CHECK(cudaEventRecord(stop, driver_stream));
    // the runtime stream waits for the driver stream's work
    CHECK(cudaStreamWaitEvent(stream, stop, 0));
    std::vector<int> copy(n);
    CHECK(cudaMemcpyAsync(copy.data(), reinterpret_cast<void *>(ptr), n * sizeof(int), cudaMemcpyDeviceToHost,
                          stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(copy[n - 1] == 500 + n - 1, "copy after the event: %d", copy[n - 1]);
    DCHECK(cuEventSynchronize(stop));
    float ms = -1;
    DCHECK(cuEventElapsedTime(&ms, start, stop));
    EXPECT(ms >= 0, "elapsed %f", ms);
    unsigned int stream_flags = 0;
    DCHECK(cuStreamGetFlags(stream, &stream_flags));
    EXPECT(stream_flags == CU_STREAM_NON_BLOCKING, "runtime stream flags %u", stream_flags);
    CUcontext stream_ctx = nullptr;
    DCHECK(cuStreamGetCtx(driver_stream, &stream_ctx));
    EXPECT(stream_ctx == primary, "stream context");
    static int host_calls = 0;
    DCHECK(cuLaunchHostFunc(driver_stream, [](void *) { host_calls++; }, nullptr));
    CHECK(cudaStreamSynchronize(driver_stream));
    EXPECT(host_calls == 1, "host function calls %d", host_calls);
    DCHECK(cuEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    DCHECK(cuStreamDestroy(driver_stream));
    PASS("streams and events");

    // ---- stream capture of driver launches
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    value = 600;
    DCHECK(cuLaunchKernel(fill, (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream, params, nullptr));
    DCHECK(cuStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    DCHECK(cuGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    expectFilled(ptr, 600, n);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("captured driver launches");

    // ---- entry points and errors
    void *entry = nullptr;
    CUdriverProcAddressQueryResult status{};
    DCHECK(cuGetProcAddress("cuMemAlloc", &entry, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    EXPECT(entry != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS, "cuMemAlloc entry point");
    // (the entry points may be other functions than the exported ones: call them)
    CUdeviceptr entry_ptr = 0;
    DCHECK(reinterpret_cast<decltype(&cuMemAlloc_v2)>(entry)(&entry_ptr, 256));
    DCHECK(cuMemFree(entry_ptr));
    cudaDriverEntryPointQueryResult runtime_status{};
    CHECK(cudaGetDriverEntryPointByVersion("cuMemsetD32", &entry, 13000, cudaEnableDefault, &runtime_status));
    EXPECT(entry != nullptr && runtime_status == cudaDriverEntryPointSuccess, "cuMemsetD32 entry point");
    DCHECK(reinterpret_cast<decltype(&cuMemsetD32_v2)>(entry)(ptr, 7, n));
    DCHECK(cuMemcpyDtoH(words.data(), ptr, n * sizeof(int)));
    EXPECT(words[0] == 7 && words[n - 1] == 7, "fill through the entry point: %u", words[0]);
    entry = reinterpret_cast<void *>(1);
    cuGetProcAddress("cuNoSuchFunction", &entry, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    EXPECT(entry == nullptr && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, "missing entry point");
    const char *error_name = nullptr, *error_string = nullptr;
    DCHECK(cuGetErrorName(CUDA_ERROR_INVALID_CONTEXT, &error_name));
    DCHECK(cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &error_string));
    EXPECT(strcmp(error_name, "CUDA_ERROR_INVALID_CONTEXT") == 0 && error_string != nullptr, "error names");
    EXPECT(cuGetErrorName(static_cast<CUresult>(12345), &error_name) == CUDA_ERROR_INVALID_VALUE,
           "unknown error name");
    PASS("entry points and errors");

    // ---- the default memory pool
    cudaMemPool_t pool, driver_pool;
    CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
    DCHECK(cuDeviceGetDefaultMemPool(&driver_pool, dev));
    EXPECT(pool == driver_pool, "default pools differ");
    uint64_t threshold = 1 << 20, read_threshold = 0;
    CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &read_threshold));
    EXPECT(read_threshold == threshold, "release threshold %llu", (unsigned long long) read_threshold);
    void *async_ptr = nullptr;
    CHECK(cudaMallocAsync(&async_ptr, 1 << 20, stream));
    uint64_t used = 0;
    CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    EXPECT(used >= (1 << 20), "pool memory in use %llu", (unsigned long long) used);
    CHECK(cudaFreeAsync(async_ptr, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemPoolTrimTo(pool, 0));
    cudaMemAccessDesc access{};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = cudaMemAccessFlagsProtReadWrite;
    CHECK(cudaMemPoolSetAccess(pool, &access, 1));
    PASS("default memory pool");

    CHECK(cudaStreamDestroy(stream));
    DCHECK(cuModuleUnload(module));
    DCHECK(cuMemFree(ptr));
    DCHECK(cuDevicePrimaryCtxRelease(dev));
    CHECK(cudaGetLastError());
    return 0;
}
