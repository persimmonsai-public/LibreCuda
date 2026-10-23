// NVCC_FLAGS: -arch=$ARCH -lcuda
// Virtual memory management through the driver API: device attributes, allocation granularity, cuMemCreate of device
// memory exportable to a file descriptor, an address range reserved with cuMemAddressReserve, cuMemMap and
// cuMemSetAccess, kernels and copies using the mapping, cuMemRetainAllocationHandle and
// cuMemGetAllocationPropertiesFromHandle, the allocation exported to a file descriptor and imported again (mapped at
// a second address, the same memory), and, with two devices, device 1 given access to device 0's mapping. Unmapping
// and releasing in any order. Checked against NVIDIA's driver and runtime too.
#include "check.h"
#include <cuda.h>
#include <unistd.h>
#include <vector>

#define DCHECK(call) do { \
    CUresult dcheck_result = (call); \
    if (dcheck_result != CUDA_SUCCESS) { \
        const char *dcheck_name = nullptr; \
        cuGetErrorName(dcheck_result, &dcheck_name); \
        fprintf(stderr, "driver error %s at %s:%d: %s\n", dcheck_name, __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

__global__ void fill(int *p, int n, int base) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = base + i;
}

// counts the elements that aren't base + i, then adds one to each
__global__ void checkAndIncrement(int *p, int n, int base, int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (p[i] != base + i) atomicAdd(errors, 1);
        p[i] += 1;
    }
}

static int countErrors(int *p, int n, int base) {
    int *errors;
    CHECK(cudaMallocManaged(&errors, sizeof(int)));
    *errors = 0;
    checkAndIncrement<<<16, 256>>>(p, n, base, errors);
    CHECK(cudaDeviceSynchronize());
    const int result = *errors;
    CHECK(cudaFree(errors));
    return result;
}

int main() {
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(nullptr));
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    int vmm = 0, fdHandles = 0;
    DCHECK(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0));
    DCHECK(cuDeviceGetAttribute(&fdHandles, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, 0));
    EXPECT(vmm == 1 && fdHandles == 1, "virtual memory management %d, file descriptor handles %d", vmm, fdHandles);

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    DCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    EXPECT(granularity > 0 && (granularity & (granularity - 1)) == 0, "granularity %zu", granularity);
    const size_t size = 4 * granularity > (16 << 20) ? 4 * granularity : (16 << 20);
    const int n = int(size / sizeof(int));

    CUmemGenericAllocationHandle handle;
    DCHECK(cuMemCreate(&handle, size, &prop, 0));
    CUdeviceptr base = 0;
    DCHECK(cuMemAddressReserve(&base, 2 * size, 0, 0, 0));
    EXPECT(base % granularity == 0, "reserved address %llx", (unsigned long long) base);
    DCHECK(cuMemMap(base, size, 0, handle, 0));
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DCHECK(cuMemSetAccess(base, size, &access, 1));

    int *p = reinterpret_cast<int *>(base);
    fill<<<16, 256>>>(p, n, 7);
    CHECK(cudaDeviceSynchronize());
    EXPECT(countErrors(p, n, 7) == 0, "kernel through the mapping");
    std::vector<int> host(n);
    CHECK(cudaMemcpy(host.data(), p, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) EXPECT(host[i] == 8 + i, "copy from the mapping [%d] = %d", i, host[i]);
    CHECK(cudaMemset(p, 0, 4096));
    CHECK(cudaMemcpy(host.data(), p, 8192, cudaMemcpyDeviceToHost));
    EXPECT(host[0] == 0 && host[1023] == 0 && host[1024] == 8 + 1024, "memset of the mapping");
    printf("PASS create, reserve, map and use (granularity %zu)\n", granularity);

    // the allocation behind an address, and its properties
    CUmemGenericAllocationHandle retained;
    DCHECK(cuMemRetainAllocationHandle(&retained, p + 1000));
    CUmemAllocationProp got{};
    DCHECK(cuMemGetAllocationPropertiesFromHandle(&got, retained));
    EXPECT(got.type == CU_MEM_ALLOCATION_TYPE_PINNED && got.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
           got.location.id == 0, "properties %d %d %d", got.type, got.location.type, got.location.id);
    DCHECK(cuMemRelease(retained));
    printf("PASS retain allocation handle and properties\n");

    // exported to a file descriptor and imported: the same memory at a second address
    int fd = -1;
    DCHECK(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    EXPECT(fd >= 0, "file descriptor %d", fd);
    CUmemGenericAllocationHandle imported;
    DCHECK(cuMemImportFromShareableHandle(&imported, reinterpret_cast<void *>(intptr_t(fd)),
                                          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(fd);
    const CUdeviceptr second = base + size;
    DCHECK(cuMemMap(second, size, 0, imported, 0));
    DCHECK(cuMemSetAccess(second, size, &access, 1));
    DCHECK(cuMemRelease(imported));  // (the mapping keeps the memory)
    fill<<<16, 256>>>(reinterpret_cast<int *>(second), n, 100);
    CHECK(cudaDeviceSynchronize());
    EXPECT(countErrors(p, n, 100) == 0, "the imported allocation's writes through the first mapping");
    printf("PASS export to and import from a file descriptor\n");

    // device 1 reaches device 0's memory through the same addresses
    int peer = 0;
    if (deviceCount >= 2) CHECK(cudaDeviceCanAccessPeer(&peer, 1, 0));
    if (peer) {
        access.location.id = 1;
        DCHECK(cuMemSetAccess(base, size, &access, 1));
        CHECK(cudaSetDevice(1));
        EXPECT(countErrors(p, n, 101) == 0, "device 1 reading device 0's mapping");
        CHECK(cudaSetDevice(0));
        EXPECT(countErrors(p, n, 102) == 0, "device 1's writes seen by device 0");
        printf("PASS access from device 1\n");
    } else {
        printf("PASS access from device 1 (skipped: no two devices with peer access)\n");
    }

    DCHECK(cuMemUnmap(second, size));
    DCHECK(cuMemRelease(handle));  // (still mapped at base)
    EXPECT(countErrors(p, n, peer ? 103 : 101) == 0, "the mapping after its handle's release");
    DCHECK(cuMemUnmap(base, size));
    DCHECK(cuMemAddressFree(base, 2 * size));
    PASS("virtual memory management");
    return 0;
}
