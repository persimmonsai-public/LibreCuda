/**
 * LibreCUDA's NCCL: symmetric memory windows and device communicators (NCCL 2.28's device API, nccl_device.h).
 *
 * Kernels compiled against NCCL's public device headers reach every rank's window memory directly (load / store
 * accessible, "LSA") and synchronize with LSA barriers; the host side here sets up what those headers read:
 *  - ncclMemAlloc: physical memory (cuMemCreate, exportable as a file descriptor) mapped at a reserved range.
 *  - ncclCommWindowRegister (collective): every rank's allocation behind the buffer is mapped into one flat range per
 *    window, rank r's at lsaFlatBase + r * stride4G * 4 GiB (the layout ncclGetLsaPointer computes with); the
 *    allocations travel between processes as file descriptors over abstract Unix sockets. The window's struct
 *    ncclWindow_vidmem lives in device memory; the communicator's window table (ncclFindWindow) lists it.
 *  - ncclDevCommCreate: a resource window for the LSA barriers, LL all-to-all buffers and the caller's resource
 *    buffers (in units of 128 bytes), zeroed, and struct ncclDevComm describing it.
 * The LSA team is the whole communicator (one machine, peer access between all GPUs). Not supported: multicast
 * (multimem), the network (GIN) and its barriers, windows of several ranks driven by one thread, and buffers at
 * different offsets of their allocations on different ranks.
 *
 * Only the CUDA driver's public virtual memory API (cuMem*, looked up in libcuda.so.1: LibreCUDA's or NVIDIA's) and
 * NCCL's public API (for the communicator's rank, size and device, and all-gathers) are used.
 */
#include <nccl.h>
#include <nccl_device.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

// ---- the driver's virtual memory API

struct Driver {
    CUresult (*memCreate)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
    CUresult (*memRelease)(CUmemGenericAllocationHandle);
    CUresult (*memAddressReserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*memAddressFree)(CUdeviceptr, size_t);
    CUresult (*memMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*memUnmap)(CUdeviceptr, size_t);
    CUresult (*memSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    CUresult (*memGetAllocationGranularity)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
    CUresult (*memExportToShareableHandle)(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType,
                                           unsigned long long);
    CUresult (*memImportFromShareableHandle)(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
    CUresult (*memRetainAllocationHandle)(CUmemGenericAllocationHandle *, void *);
    CUresult (*memGetAddressRange)(CUdeviceptr *, size_t *, CUdeviceptr);
    bool ok;
};

const Driver &driver() {
    static const Driver d = [] {
        Driver d{};
        void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (cuda == nullptr) return d;
        auto find = [&](const char *name) {
            const std::string v2 = std::string(name) + "_v2";
            void *f = dlsym(cuda, v2.c_str());
            return f != nullptr ? f : dlsym(cuda, name);
        };
#define LOAD(field, name) d.field = reinterpret_cast<decltype(d.field)>(find(name))
        LOAD(memCreate, "cuMemCreate");
        LOAD(memRelease, "cuMemRelease");
        LOAD(memAddressReserve, "cuMemAddressReserve");
        LOAD(memAddressFree, "cuMemAddressFree");
        LOAD(memMap, "cuMemMap");
        LOAD(memUnmap, "cuMemUnmap");
        LOAD(memSetAccess, "cuMemSetAccess");
        LOAD(memGetAllocationGranularity, "cuMemGetAllocationGranularity");
        LOAD(memExportToShareableHandle, "cuMemExportToShareableHandle");
        LOAD(memImportFromShareableHandle, "cuMemImportFromShareableHandle");
        LOAD(memRetainAllocationHandle, "cuMemRetainAllocationHandle");
        LOAD(memGetAddressRange, "cuMemGetAddressRange");
#undef LOAD
        d.ok = d.memCreate && d.memRelease && d.memAddressReserve && d.memAddressFree && d.memMap && d.memUnmap &&
               d.memSetAccess && d.memGetAllocationGranularity && d.memExportToShareableHandle &&
               d.memImportFromShareableHandle && d.memRetainAllocationHandle && d.memGetAddressRange;
        return d;
    }();
    return d;
}

ncclResult_t report(ncclResult_t result, const std::string &what) {
    if (getenv("NCCL_DEBUG") != nullptr || getenv("LIBRECUDA_NCCL_DEBUG") != nullptr) {
        fprintf(stderr, "[nccl] %s\n", what.c_str());
    }
    return result;
}

#define DRV(call) do { \
    const CUresult drv_result = (call); \
    if (drv_result != CUDA_SUCCESS) return report(ncclUnhandledCudaError, std::string(#call) + " failed: " + std::to_string(int(drv_result))); \
} while (0)
#define RT(call) do { \
    const cudaError_t rt_result = (call); \
    if (rt_result != cudaSuccess) return report(ncclUnhandledCudaError, std::string(#call) + " failed: " + cudaGetErrorString(rt_result)); \
} while (0)
#define NC(call) do { const ncclResult_t nc_result = (call); if (nc_result != ncclSuccess) return nc_result; } while (0)

CUmemAllocationProp allocationProp(int device) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return prop;
}

ncclResult_t granularity(int device, size_t *out) {
    const CUmemAllocationProp prop = allocationProp(device);
    DRV(driver().memGetAllocationGranularity(out, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return ncclSuccess;
}

// read / write access for the current device (and the others with peer access) to a mapped range
ncclResult_t allowAccess(CUdeviceptr va, size_t size) {
    int devices = 0, current = 0;
    RT(cudaGetDeviceCount(&devices));
    RT(cudaGetDevice(&current));
    std::vector<CUmemAccessDesc> access;
    for (int d = 0; d < devices; d++) {
        int peer = d == current;
        if (!peer) cudaDeviceCanAccessPeer(&peer, current, d);
        if (!peer) continue;
        CUmemAccessDesc desc{};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = d;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access.push_back(desc);
    }
    cudaGetLastError();
    DRV(driver().memSetAccess(va, size, access.data(), access.size()));
    return ncclSuccess;
}

// ---- ncclMemAlloc'd memory

struct Allocation {
    CUmemGenericAllocationHandle handle;
    size_t size;
};
std::mutex g_mutex;
std::map<CUdeviceptr, Allocation> g_allocations;

// ---- collective helpers on the communicator (public API): all-gather of fixed size records, barrier

template<typename T>
ncclResult_t allGather(ncclComm_t comm, const T &mine, std::vector<T> &all) {
    int n = 0;
    NC(ncclCommCount(comm, &n));
    const size_t bytes = (sizeof(T) + 7) / 8 * 8;
    char *buffer = nullptr;
    cudaStream_t stream;
    RT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    RT(cudaMalloc(&buffer, bytes * (n + 1)));
    RT(cudaMemcpy(buffer + bytes * n, &mine, sizeof(T), cudaMemcpyHostToDevice));
    NC(ncclAllGather(buffer + bytes * n, buffer, bytes, ncclUint8, comm, stream));
    RT(cudaStreamSynchronize(stream));
    std::vector<char> host(bytes * n);
    RT(cudaMemcpy(host.data(), buffer, bytes * n, cudaMemcpyDeviceToHost));
    RT(cudaFree(buffer));
    RT(cudaStreamDestroy(stream));
    all.resize(n);
    for (int r = 0; r < n; r++) memcpy(&all[r], host.data() + bytes * r, sizeof(T));
    return ncclSuccess;
}

ncclResult_t barrier(ncclComm_t comm) {
    std::vector<int> ignored;
    return allGather(comm, 0, ignored);
}

// ---- file descriptors between processes: every rank listens on an abstract Unix socket and sends its descriptor
// to every peer (the connections queue in the listening sockets until the peers accept them)

std::string socketName(int pid, uint64_t serial, int rank) {
    return "librecuda-nccl-window-" + std::to_string(pid) + "-" + std::to_string(serial) + "-" +
           std::to_string(rank);
}

sockaddr_un socketAddress(const std::string &name, socklen_t *len) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    // abstract namespace: leading NUL
    memcpy(addr.sun_path + 1, name.data(), std::min(name.size(), sizeof(addr.sun_path) - 2));
    *len = socklen_t(offsetof(sockaddr_un, sun_path) + 1 + std::min(name.size(), sizeof(addr.sun_path) - 2));
    return addr;
}

ncclResult_t sendFd(const std::string &peerSocket, int rank, int fd) {
    const int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return report(ncclSystemError, "socket failed");
    socklen_t len;
    sockaddr_un addr = socketAddress(peerSocket, &len);
    if (connect(s, reinterpret_cast<sockaddr *>(&addr), len) != 0) {
        close(s);
        return report(ncclSystemError, "connect to " + peerSocket + " failed: " + strerror(errno));
    }
    char control[CMSG_SPACE(sizeof(int))]{};
    iovec iov{&rank, sizeof(rank)};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
    const bool ok = sendmsg(s, &msg, 0) == ssize_t(sizeof(rank));
    close(s);
    return ok ? ncclSuccess : report(ncclSystemError, "sendmsg failed");
}

ncclResult_t receiveFd(int listener, int *rank, int *fd) {
    const int s = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (s < 0) return report(ncclSystemError, "accept failed");
    char control[CMSG_SPACE(sizeof(int))]{};
    iovec iov{rank, sizeof(*rank)};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    const ssize_t got = recvmsg(s, &msg, MSG_CMSG_CLOEXEC);
    close(s);
    cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (got != ssize_t(sizeof(*rank)) || cmsg == nullptr || cmsg->cmsg_type != SCM_RIGHTS) {
        return report(ncclSystemError, "recvmsg failed");
    }
    memcpy(fd, CMSG_DATA(cmsg), sizeof(int));
    return ncclSuccess;
}

// ---- windows

struct Window {
    ncclComm_t comm;
    void *buffer;
    size_t size;
    CUdeviceptr flat;       // the reserved range: nRanks slots of stride bytes
    size_t stride;
    std::vector<CUmemGenericAllocationHandle> handles;  // every rank's allocation (imported or retained)
    std::vector<size_t> mapped;                         // bytes mapped in each slot
    ncclWindow_vidmem *device;                          // the window's struct in device memory (ncclWindow_t)
    int tableIndex;
};

// per communicator: device window tables (32 entries each, chained) and a serial number for socket names
struct CommState {
    std::vector<ncclDevCommWindowTable *> tables;   // device memory
    std::vector<Window *> entries;                  // host copy of the slots (nullptr: free)
    uint64_t serial = 0;
};
std::map<ncclComm_t, CommState> g_comms;

ncclResult_t tableInsert(CommState &state, Window *w) {
    int index = -1;
    for (size_t i = 0; i < state.entries.size(); i++) {
        if (state.entries[i] == nullptr) {
            index = int(i);
            break;
        }
    }
    if (index < 0) {
        index = int(state.entries.size());
        state.entries.resize(state.entries.size() + 32, nullptr);
        ncclDevCommWindowTable *table = nullptr;
        RT(cudaMalloc(&table, sizeof(ncclDevCommWindowTable)));
        RT(cudaMemset(table, 0, sizeof(ncclDevCommWindowTable)));
        if (!state.tables.empty()) {
            ncclDevCommWindowTable *previous = state.tables.back();
            RT(cudaMemcpy(&previous->next, &table, sizeof(table), cudaMemcpyHostToDevice));
        }
        state.tables.push_back(table);
    }
    state.entries[index] = w;
    w->tableIndex = index;
    ncclDevCommWindowTable::Entry e{reinterpret_cast<uintptr_t>(w->buffer), w->size, w->device};
    RT(cudaMemcpy(&state.tables[index / 32]->entries[index % 32], &e, sizeof(e), cudaMemcpyHostToDevice));
    return ncclSuccess;
}

ncclResult_t tableRemove(CommState &state, Window *w) {
    ncclDevCommWindowTable::Entry e{};
    RT(cudaMemcpy(&state.tables[w->tableIndex / 32]->entries[w->tableIndex % 32], &e, sizeof(e),
                  cudaMemcpyHostToDevice));
    state.entries[w->tableIndex] = nullptr;
    return ncclSuccess;
}

ncclResult_t releaseWindow(Window *w) {
    for (size_t r = 0; r < w->mapped.size(); r++) {
        if (w->mapped[r] != 0) driver().memUnmap(w->flat + r * w->stride, w->mapped[r]);
    }
    for (CUmemGenericAllocationHandle h: w->handles) {
        if (h != 0) driver().memRelease(h);
    }
    if (w->flat != 0) driver().memAddressFree(w->flat, w->stride * w->mapped.size());
    if (w->device != nullptr) cudaFree(w->device);
    delete w;
    return ncclSuccess;
}

ncclResult_t registerWindow(ncclComm_t comm, void *buffer, size_t size, ncclWindow_t *out) {
    if (!driver().ok) return report(ncclInvalidUsage, "the CUDA driver has no virtual memory API");
    int n = 0, rank = 0, device = 0;
    NC(ncclCommCount(comm, &n));
    NC(ncclCommUserRank(comm, &rank));
    NC(ncclCommCuDevice(comm, &device));
    RT(cudaSetDevice(device));

    // the allocation behind the buffer (ncclMemAlloc'd, or other cuMemCreate'd memory)
    struct Info {
        uint64_t allocationSize, offset, granularity;
        int pid;
        int peerAccess;   // this rank's GPU reaches every rank's GPU
        unsigned char uuid[16];
    } mine{};
    CUdeviceptr base = 0;
    size_t allocationSize = 0;
    DRV(driver().memGetAddressRange(&base, &allocationSize, reinterpret_cast<CUdeviceptr>(buffer)));
    mine.allocationSize = allocationSize;
    mine.offset = reinterpret_cast<CUdeviceptr>(buffer) - base;
    NC(granularity(device, &mine.granularity));
    mine.pid = getpid();
    cudaDeviceProp prop{};
    RT(cudaGetDeviceProperties(&prop, device));
    memcpy(mine.uuid, prop.uuid.bytes, 16);

    CommState &state = g_comms[comm];
    const uint64_t serial = state.serial++;
    // listen before the peers learn the socket's name
    const std::string name = socketName(mine.pid, serial, rank);
    const int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    socklen_t len;
    sockaddr_un addr = socketAddress(name, &len);
    if (listener < 0 || bind(listener, reinterpret_cast<sockaddr *>(&addr), len) != 0 || listen(listener, 64) != 0) {
        if (listener >= 0) close(listener);
        return report(ncclSystemError, "window socket " + name + ": " + strerror(errno));
    }
    // peer access to every rank's GPU (by UUID, as this process numbers the GPUs)
    std::vector<Info> all;
    {
        std::vector<Info> uuids;
        NC(allGather(comm, mine, uuids));
        int devices = 0;
        cudaGetDeviceCount(&devices);
        mine.peerAccess = 1;
        for (const Info &other: uuids) {
            int local = -1;
            for (int d = 0; d < devices; d++) {
                cudaDeviceProp p{};
                if (cudaGetDeviceProperties(&p, d) == cudaSuccess && memcmp(p.uuid.bytes, other.uuid, 16) == 0) local = d;
            }
            int access = local == device;
            if (!access && local >= 0) cudaDeviceCanAccessPeer(&access, device, local);
            if (!access) mine.peerAccess = 0;
        }
        cudaGetLastError();
    }
    NC(allGather(comm, mine, all));
    auto fail = [&](ncclResult_t result, const std::string &what) {
        close(listener);
        return report(result, what);
    };
    for (const Info &other: all) {
        if (!other.peerAccess) return fail(ncclInvalidUsage, "windows need peer access between all GPUs");
        if (other.offset != mine.offset) {
            return fail(ncclInvalidUsage, "window buffers at different offsets of their allocations");
        }
    }

    auto *w = new Window{};
    w->comm = comm;
    w->buffer = buffer;
    w->size = size;
    w->handles.assign(n, 0);
    w->mapped.assign(n, 0);
    // slots of whole 4 GiB units, as large as the largest allocation
    size_t largest = 0;
    for (const Info &other: all) largest = std::max<size_t>(largest, other.allocationSize);
    const size_t unit = size_t(1) << 32;
    w->stride = (largest + unit - 1) / unit * unit;
    if (CUresult e = driver().memAddressReserve(&w->flat, w->stride * n, unit, 0, 0); e != CUDA_SUCCESS) {
        releaseWindow(w);
        return fail(ncclUnhandledCudaError, "cuMemAddressReserve failed: " + std::to_string(int(e)));
    }

    // this rank's allocation: retained, exported, sent to every peer
    ncclResult_t result = ncclSuccess;
    int fd = -1;
    if (driver().memRetainAllocationHandle(&w->handles[rank], buffer) != CUDA_SUCCESS ||
        driver().memExportToShareableHandle(&fd, w->handles[rank], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) !=
        CUDA_SUCCESS) {
        result = report(ncclInvalidArgument, "the window buffer isn't exportable cuMemCreate'd memory (ncclMemAlloc)");
    }
    for (int r = 0; r < n && result == ncclSuccess; r++) {
        if (r != rank) result = sendFd(socketName(all[r].pid, serial, r), rank, fd);
    }
    for (int i = 0; i < n - 1 && result == ncclSuccess; i++) {
        int from = -1, peerFd = -1;
        result = receiveFd(listener, &from, &peerFd);
        if (result != ncclSuccess) break;
        if (from < 0 || from >= n || from == rank ||
            driver().memImportFromShareableHandle(&w->handles[from], reinterpret_cast<void *>(intptr_t(peerFd)),
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS) {
            result = report(ncclUnhandledCudaError, "importing a peer's window memory failed");
        }
        close(peerFd);
    }
    if (fd >= 0) close(fd);
    close(listener);
    // every rank's allocation in its slot
    for (int r = 0; r < n && result == ncclSuccess; r++) {
        const CUdeviceptr slot = w->flat + r * w->stride;
        if (driver().memMap(slot, all[r].allocationSize, 0, w->handles[r], 0) != CUDA_SUCCESS) {
            result = report(ncclUnhandledCudaError, "mapping a window slot failed");
            break;
        }
        w->mapped[r] = all[r].allocationSize;
        result = allowAccess(slot, all[r].allocationSize);
    }
    if (result == ncclSuccess) {
        ncclWindow_vidmem v{};
        v.winHost = w;
        v.lsaFlatBase = reinterpret_cast<char *>(w->flat + mine.offset);
        v.lsaRank = rank;
        v.worldRank = rank;
        v.stride4G = uint32_t(w->stride >> 32);
        if (cudaMalloc(&w->device, sizeof(v)) != cudaSuccess ||
            cudaMemcpy(w->device, &v, sizeof(v), cudaMemcpyHostToDevice) != cudaSuccess) {
            result = report(ncclUnhandledCudaError, "window struct");
        }
    }
    if (result == ncclSuccess) result = tableInsert(state, w);
    // (all ranks mapped everything before anyone uses the window)
    if (result == ncclSuccess) result = barrier(comm);
    if (result != ncclSuccess) {
        releaseWindow(w);
        return result;
    }
    *out = w->device;
    return ncclSuccess;
}

ncclResult_t deregisterWindow(ncclComm_t comm, ncclWindow_t win) {
    if (win == nullptr) return ncclInvalidArgument;
    ncclWindow_vidmem v{};
    RT(cudaMemcpy(&v, win, sizeof(v), cudaMemcpyDeviceToHost));
    auto *w = static_cast<Window *>(v.winHost);
    if (w == nullptr || w->comm != comm || w->device != win) return ncclInvalidArgument;
    // nobody may still access the window: the ranks' earlier work completed (as NCCL, the caller ensures that)
    RT(cudaDeviceSynchronize());
    NC(barrier(comm));
    NC(tableRemove(g_comms[comm], w));
    return releaseWindow(w);
}

// ---- device communicators

struct DevCommState {
    void *resource;        // ncclMemAlloc'd
    ncclWindow_t resourceWindow;
};
std::map<ncclWindow_t, DevCommState> g_devComms;

size_t alignUp(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

} // namespace

extern "C" {

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
    if (ptr == nullptr) return ncclInvalidArgument;
    if (!driver().ok) {
        // no windows without the virtual memory API: ordinary device memory
        RT(cudaMalloc(ptr, size));
        return ncclSuccess;
    }
    int device = 0;
    RT(cudaGetDevice(&device));
    RT(cudaFree(nullptr));  // (the device's context current for the driver API)
    size_t g = 0;
    NC(granularity(device, &g));
    const size_t bytes = alignUp(std::max<size_t>(size, 1), g);
    const CUmemAllocationProp prop = allocationProp(device);
    CUmemGenericAllocationHandle handle;
    DRV(driver().memCreate(&handle, bytes, &prop, 0));
    CUdeviceptr va = 0;
    if (driver().memAddressReserve(&va, bytes, g, 0, 0) != CUDA_SUCCESS ||
        driver().memMap(va, bytes, 0, handle, 0) != CUDA_SUCCESS || allowAccess(va, bytes) != ncclSuccess) {
        if (va != 0) {
            driver().memUnmap(va, bytes);
            driver().memAddressFree(va, bytes);
        }
        driver().memRelease(handle);
        return report(ncclUnhandledCudaError, "ncclMemAlloc: mapping failed");
    }
    std::lock_guard lock(g_mutex);
    g_allocations[va] = {handle, bytes};
    *ptr = reinterpret_cast<void *>(va);
    return ncclSuccess;
}

ncclResult_t ncclMemFree(void *ptr) {
    if (ptr == nullptr) return ncclSuccess;
    Allocation a;
    {
        std::lock_guard lock(g_mutex);
        auto it = g_allocations.find(reinterpret_cast<CUdeviceptr>(ptr));
        if (it == g_allocations.end()) {
            RT(cudaFree(ptr));
            return ncclSuccess;
        }
        a = it->second;
        g_allocations.erase(it);
    }
    cudaDeviceSynchronize();
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(ptr);
    DRV(driver().memUnmap(va, a.size));
    DRV(driver().memAddressFree(va, a.size));
    DRV(driver().memRelease(a.handle));
    return ncclSuccess;
}

ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size, ncclWindow_t *win, int) {
    if (comm == nullptr || buff == nullptr || win == nullptr) return ncclInvalidArgument;
    std::lock_guard lock(g_mutex);
    return registerWindow(comm, buff, size, win);
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
    std::lock_guard lock(g_mutex);
    return deregisterWindow(comm, win);
}

// (NCCL 2.29) host side LSA pointers into a window: rank r's buffer is at lsaFlatBase + r * stride4G * 4 GiB
ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void **outPtr) {
    if (window == nullptr || outPtr == nullptr) return ncclInvalidArgument;
    ncclWindow_vidmem v{};
    RT(cudaMemcpy(&v, window, sizeof(v), cudaMemcpyDeviceToHost));
    auto *w = static_cast<Window *>(v.winHost);
    int n = 0;
    if (w == nullptr || ncclCommCount(w->comm, &n) != ncclSuccess || lsaRank < 0 || lsaRank >= n ||
        offset >= w->size) {
        return ncclInvalidArgument;
    }
    *outPtr = v.lsaFlatBase + (size_t(lsaRank) * v.stride4G << 32) + offset;
    return ncclSuccess;
}

// (the LSA team is the world)
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void **outPtr) {
    return ncclGetLsaDevicePointer(window, offset, peer, outPtr);
}

ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t, size_t, void **) {
    return report(ncclInvalidUsage, "multimem (multicast) pointers are not supported");
}

ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
    ncclTeam_t t{1, 0, 1};
    ncclCommCount(comm, &t.nRanks);
    ncclCommUserRank(comm, &t.rank);
    return t;
}

// one machine with peer access between all GPUs: the LSA team is the world
ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
    return ncclTeamWorld(comm);
}

ncclTeam_t ncclTeamRail(ncclComm_t comm) {
    ncclTeam_t t{1, 0, 1};
    ncclCommCount(comm, &t.stride);
    return t;
}

int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
    int me = 0;
    ncclCommUserRank(comm, &me);
    return me + (rank - team.rank) * team.stride;
}

int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) {
    return ncclTeamRankToWorld(comm, team, rank);
}

// layout of nccl_device/impl/lsa_barrier__types.h: per barrier 3 words, plus one inbox word per team member
ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t *outHandle,
                                             ncclDevResourceRequirements_t *outReq) {
    if (outHandle == nullptr || outReq == nullptr || nBarriers < 0) return ncclInvalidArgument;
    *outReq = {};
    outReq->bufferSize = size_t(3 + team.nRanks) * nBarriers * sizeof(uint32_t);
    outReq->bufferAlign = 128;
    outReq->outBufferHandle = &outHandle->bufHandle;
    outHandle->nBarriers = nBarriers;
    return ncclSuccess;
}

// layout of nccl_device/impl/ll_a2a__funcs.h: per block a header line and two sets of slots, 16 bytes each;
// an element takes one slot per 8 bytes
int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) {
    return maxElts * ((maxEltSize + 7) / 8);
}

ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t *outHandle,
                                        ncclDevResourceRequirements_t *outReq) {
    if (outHandle == nullptr || outReq == nullptr || nBlocks < 0 || nSlots < 0) return ncclInvalidArgument;
    *outReq = {};
    outReq->bufferSize = size_t(nBlocks) * (1 + 2 * size_t(nSlots)) * 16;
    outReq->bufferAlign = 128;
    outReq->outBufferHandle = &outHandle->bufHandle;
    outHandle->nSlots = uint32_t(nSlots);
    return ncclSuccess;
}

ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int, ncclGinBarrierHandle_t *,
                                             ncclDevResourceRequirements_t *) {
    return report(ncclInvalidUsage, "GIN (network) barriers are not supported");
}

ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const *reqs, ncclDevComm_t *out) {
    if (comm == nullptr || reqs == nullptr || out == nullptr) return ncclInvalidArgument;
    if (reqs->lsaMultimem || reqs->ginForceEnable || reqs->ginSignalCount > 0 || reqs->ginCounterCount > 0 ||
        reqs->railGinBarrierCount > 0) {
        return report(ncclInvalidUsage, "multimem and GIN (network) resources are not supported");
    }
    int n = 0, rank = 0, device = 0;
    NC(ncclCommCount(comm, &n));
    NC(ncclCommUserRank(comm, &rank));
    NC(ncclCommCuDevice(comm, &device));
    RT(cudaSetDevice(device));

    // the resource buffers: the communicator's barriers, LL all-to-all, then the caller's, in 128 byte units
    ncclLsaBarrierHandle_t lsaBarrier{};
    ncclDevResourceRequirements_t barrierReq{};
    // (world barriers are LSA barriers: the LSA team is the world)
    NC(ncclLsaBarrierCreateRequirement(ncclTeamLsa(comm), std::max(reqs->lsaBarrierCount, reqs->barrierCount),
                                       &lsaBarrier, &barrierReq));
    ncclLLA2AHandle_t lla2a{};
    ncclDevResourceRequirements_t lla2aReq{};
    NC(ncclLLA2ACreateRequirement(reqs->lsaLLA2ABlockCount, reqs->lsaLLA2ASlotCount, &lla2a, &lla2aReq));
    std::vector<ncclDevResourceRequirements_t *> list{&barrierReq, &lla2aReq};
    for (ncclDevResourceRequirements_t *r = reqs->resourceRequirementsList; r != nullptr; r = r->next) {
        if (r->ginSignalCount > 0 || r->ginCounterCount > 0) {
            return report(ncclInvalidUsage, "GIN (network) signals and counters are not supported");
        }
        list.push_back(r);
    }
    size_t bytes = 0;
    for (ncclDevResourceRequirements_t *r: list) {
        bytes = alignUp(bytes, std::max<size_t>(128, r->bufferAlign));
        if (r->outBufferHandle != nullptr) *r->outBufferHandle = ncclDevResourceHandle_t(bytes / 128);
        bytes += alignUp(r->bufferSize, 128);
    }
    void *resource = nullptr;
    NC(ncclMemAlloc(&resource, std::max<size_t>(bytes, 128)));
    RT(cudaMemset(resource, 0, std::max<size_t>(bytes, 128)));
    RT(cudaDeviceSynchronize());
    ncclWindow_t window = nullptr;
    if (ncclResult_t result = ncclCommWindowRegister(comm, resource, std::max<size_t>(bytes, 128), &window, 0);
        result != ncclSuccess) {
        ncclMemFree(resource);
        return result;
    }

    ncclDevComm d{};
    d.rank = rank;
    d.nRanks = n;
    d.nRanks_rcp32 = nccl::utility::idivRcp32(uint32_t(n));
    d.lsaRank = rank;
    d.lsaSize = n;
    d.lsaSize_rcp32 = d.nRanks_rcp32;
    {
        std::lock_guard lock(g_mutex);
        d.windowTable = g_comms[comm].tables.empty() ? nullptr : g_comms[comm].tables.front();
        g_devComms[window] = {resource, window};
    }
    d.resourceWindow = window;
    RT(cudaMemcpy(&d.resourceWindow_inlined, window, sizeof(d.resourceWindow_inlined), cudaMemcpyDeviceToHost));
    d.lsaBarrier = lsaBarrier;
    *out = d;
    return ncclSuccess;
}

ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const *devComm) {
    if (comm == nullptr || devComm == nullptr) return ncclInvalidArgument;
    DevCommState s;
    {
        std::lock_guard lock(g_mutex);
        auto it = g_devComms.find(devComm->resourceWindow);
        if (it == g_devComms.end()) return ncclInvalidArgument;
        s = it->second;
        g_devComms.erase(it);
    }
    NC(ncclCommWindowDeregister(comm, s.resourceWindow));
    return ncclMemFree(s.resource);
}

} // extern "C"
