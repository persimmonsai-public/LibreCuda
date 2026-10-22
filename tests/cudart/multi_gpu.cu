// Several devices: pointer attributes, copies between devices without peer access (cudaMemcpy, cudaMemcpyPeer,
// cudaMemcpyPeerAsync, host copies to and from another device's memory; fills of it are refused), peer access
// (kernels reading and writing a peer's memory, allocations made after enabling it, error codes, disabling),
// page-locked memory used on every device, freeing another device's memory, cross-device event waits, one thread per
// device and peer access after a device reset. Needs 2 devices (passes as skipped with fewer). Checked against
// NVIDIA's runtime too.
#include "check.h"
#include <thread>
#include <vector>

__global__ void fill(int *p, int n, int base) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = base + i;
    }
}

// out[i] = in[i] * 2 (in and out may be on different devices)
__global__ void twice(const int *in, int *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = in[i] * 2;
    }
}

// advances a token shared by both devices from expected to expected + 1, counting out of order steps
__global__ void pass(volatile int *token, int expected, int *errors) {
    if (*token != expected) {
        atomicAdd(errors, 1);
    }
    *token = expected + 1;
}

static void expectSequence(const std::vector<int> &h, int base, int factor, const char *what) {
    for (size_t i = 0; i < h.size(); i++) {
        EXPECT(h[i] == (base + int(i)) * factor, "%s: [%zu] = %d, expected %d", what, i, h[i], (base + int(i)) * factor);
    }
}

static std::vector<int> download(const int *p, int n) {
    std::vector<int> h(n);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(int), cudaMemcpyDefault));
    return h;
}

static int *allocFilled(int device, int n, int base) {
    int *p;
    CHECK(cudaSetDevice(device));
    CHECK(cudaMalloc(&p, n * sizeof(int)));
    fill<<<64, 256>>>(p, n, base);
    CHECK(cudaDeviceSynchronize());
    return p;
}

static void testWithoutPeerAccess(int n) {
    int *d0 = allocFilled(0, n, 0), *d1 = allocFilled(1, n, 1000);

    for (int device = 0; device < 2; device++) {
        cudaPointerAttributes attributes{};
        CHECK(cudaSetDevice(1 - device));
        CHECK(cudaPointerGetAttributes(&attributes, device == 0 ? d0 : d1));
        EXPECT(attributes.type == cudaMemoryTypeDevice && attributes.device == device, "attributes of device %d memory: "
               "type %d device %d", device, int(attributes.type), attributes.device);
    }

    // device to device with either device current, through cudaMemcpy and the peer functions
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpy(d0, d1, n * sizeof(int), cudaMemcpyDeviceToDevice));
    expectSequence(download(d0, n), 1000, 1, "cudaMemcpy device 1 -> 0 from device 0");
    CHECK(cudaSetDevice(1));
    fill<<<64, 256>>>(d1, n, 7);
    CHECK(cudaMemcpyPeer(d0, 0, d1, 1, n * sizeof(int)));
    expectSequence(download(d0, n), 7, 1, "cudaMemcpyPeer from device 1");
    EXPECT(cudaMemcpyPeer(d0, 0, d1, 5, 4) == cudaErrorInvalidDevice, "cudaMemcpyPeer with an invalid device");
    cudaGetLastError();

    for (int device = 0; device < 2; device++) {
        CHECK(cudaSetDevice(device));
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        int *src = device == 0 ? d0 : d1, *dst = device == 0 ? d1 : d0;
        // queued work on the stream comes first
        fill<<<64, 256, 0, stream>>>(src, n, 100 + device);
        CHECK(cudaMemcpyPeerAsync(dst, 1 - device, src, device, n * sizeof(int), stream));
        CHECK(cudaStreamSynchronize(stream));
        expectSequence(download(dst, n), 100 + device, 1, "cudaMemcpyPeerAsync");
        CHECK(cudaStreamDestroy(stream));
    }

    // host copies and fills of the other device's memory
    CHECK(cudaSetDevice(0));
    std::vector<int> h(n);
    for (int i = 0; i < n; i++) {
        h[i] = 50 + i;
    }
    CHECK(cudaMemcpy(d1, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    expectSequence(download(d1, n), 50, 1, "host to device 1 from device 0");
    // fills need the device to access the memory
    EXPECT(cudaMemset(d1, 0x01, n * sizeof(int)) == cudaErrorInvalidValue, "cudaMemset of device 1 memory");
    cudaGetLastError();

    // a copy larger than the bounce buffer
    const size_t big = 20u << 20;
    int *b0, *b1;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&b1, big * sizeof(int)));
    fill<<<256, 256>>>(b1, int(big), 3);
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&b0, big * sizeof(int)));
    CHECK(cudaMemcpy(b0, b1, big * sizeof(int), cudaMemcpyDefault));
    std::vector<int> hb = download(b0, int(big));
    EXPECT(hb[0] == 3 && hb[big / 2] == 3 + int(big / 2) && hb[big - 1] == 3 + int(big - 1), "large copy: %d %d %d",
           hb[0], hb[big / 2], hb[big - 1]);

    // freeing memory of another device
    CHECK(cudaFree(b1));
    CHECK(cudaFree(b0));
    CHECK(cudaFree(d1));
    CHECK(cudaSetDevice(1));
    CHECK(cudaFree(d0));
    int *small;
    CHECK(cudaMalloc(&small, 16));
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(small));
    PASS("multi_gpu without peer access");
}

static void testPeerAccess(int n) {
    int can = -1;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 0));
    EXPECT(can == 0, "a device is not its own peer");
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) {
        printf("PASS multi_gpu peer access (skipped: devices 0 and 1 can't access each other)\n");
        return;
    }
    int value = -1;
    CHECK(cudaDeviceGetP2PAttribute(&value, cudaDevP2PAttrAccessSupported, 0, 1));
    EXPECT(value == 1, "cudaDevP2PAttrAccessSupported %d", value);
    EXPECT(cudaDeviceGetP2PAttribute(&value, cudaDevP2PAttrAccessSupported, 1, 1) == cudaErrorInvalidDevice,
           "P2P attribute of a device with itself");
    cudaGetLastError();

    int *d0 = allocFilled(0, n, 0), *d1 = allocFilled(1, n, 1000);
    CHECK(cudaSetDevice(0));
    EXPECT(cudaDeviceDisablePeerAccess(1) == cudaErrorPeerAccessNotEnabled, "disable before enable");
    EXPECT(cudaDeviceEnablePeerAccess(0, 0) == cudaErrorPeerAccessUnsupported, "enable for the device itself");
    EXPECT(cudaDeviceEnablePeerAccess(1, 1) == cudaErrorInvalidValue, "enable with flags");
    cudaGetLastError();
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    EXPECT(cudaDeviceEnablePeerAccess(1, 0) == cudaErrorPeerAccessAlreadyEnabled, "enable twice");
    cudaGetLastError();

    // device 0 reads device 1 memory and writes it back
    twice<<<64, 256>>>(d1, d0, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(download(d0, n), 1000, 2, "device 0 reading device 1");
    twice<<<64, 256>>>(d0, d1, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(download(d1, n), 1000, 4, "device 0 writing device 1");
    // copies on device 0 between both devices' memory
    fill<<<64, 256>>>(d0, n, 5);
    CHECK(cudaMemcpy(d1, d0, n * sizeof(int), cudaMemcpyDefault));
    expectSequence(download(d1, n), 5, 1, "peer copy");

    // memory allocated on device 1 after enabling access
    int *late = allocFilled(1, n, 9);
    CHECK(cudaSetDevice(0));
    twice<<<64, 256>>>(late, d0, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(download(d0, n), 9, 2, "device 0 reading a later device 1 allocation");

    // page-locked memory of device 0 used on device 1
    int *pinned;
    CHECK(cudaMallocHost(&pinned, n * sizeof(int)));
    CHECK(cudaSetDevice(1));
    fill<<<64, 256>>>(pinned, n, 11);
    CHECK(cudaDeviceSynchronize());
    expectSequence(std::vector<int>(pinned, pinned + n), 11, 1, "device 1 writing device 0 page-locked memory");
    CHECK(cudaFreeHost(pinned));

    // after disabling, copies still work (without the GPU accessing the peer)
    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceDisablePeerAccess(1));
    EXPECT(cudaDeviceDisablePeerAccess(1) == cudaErrorPeerAccessNotEnabled, "disable twice");
    cudaGetLastError();
    CHECK(cudaMemcpy(d0, late, n * sizeof(int), cudaMemcpyDefault));
    expectSequence(download(d0, n), 9, 1, "copy after disabling peer access");
    CHECK(cudaFree(late));
    CHECK(cudaFree(d1));
    CHECK(cudaFree(d0));

    // a reset of the peer device ends access; it can be enabled again for its new context
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceReset());
    d1 = allocFilled(1, n, 20);
    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    d0 = allocFilled(0, n, 0);
    twice<<<64, 256>>>(d1, d0, n);
    CHECK(cudaDeviceSynchronize());
    expectSequence(download(d0, n), 20, 2, "peer access after a reset of the peer");
    CHECK(cudaDeviceDisablePeerAccess(1));
    CHECK(cudaFree(d0));
    CHECK(cudaFree(d1));
    PASS("multi_gpu peer access");
}

// streams of one device waiting on the GPU for events of the other device (ping-pong), the host waiting for an
// event of a device that isn't current
static void testCrossDeviceEvents() {
    cudaStream_t stream[2];
    cudaEvent_t done[2];
    for (int device = 0; device < 2; device++) {
        CHECK(cudaSetDevice(device));
        CHECK(cudaStreamCreateWithFlags(&stream[device], cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&done[device], cudaEventDisableTiming));
    }
    int *token, *errors;
    CHECK(cudaMallocHost(&token, sizeof(int)));
    CHECK(cudaMallocHost(&errors, sizeof(int)));
    *token = 0;
    *errors = 0;
    const int rounds = 200;
    for (int i = 0; i < rounds; i++) {
        CHECK(cudaSetDevice(0));
        if (i > 0) {
            CHECK(cudaStreamWaitEvent(stream[0], done[1], 0));
        }
        pass<<<1, 1, 0, stream[0]>>>(token, 2 * i, errors);
        CHECK(cudaEventRecord(done[0], stream[0]));
        CHECK(cudaSetDevice(1));
        CHECK(cudaStreamWaitEvent(stream[1], done[0], 0));
        pass<<<1, 1, 0, stream[1]>>>(token, 2 * i + 1, errors);
        CHECK(cudaEventRecord(done[1], stream[1]));
    }
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventSynchronize(done[1]));
    EXPECT(*token == 2 * rounds && *errors == 0, "ping-pong: token %d (expected %d), %d out of order", *token,
           2 * rounds, *errors);
    for (int device = 0; device < 2; device++) {
        CHECK(cudaSetDevice(device));
        CHECK(cudaStreamDestroy(stream[device]));
        CHECK(cudaEventDestroy(done[device]));
    }
    CHECK(cudaFreeHost(token));
    CHECK(cudaFreeHost(errors));
    PASS("multi_gpu cross-device events");
}

static void testThreadPerDevice(int n) {
    bool ok[2] = {};
    std::thread threads[2];
    for (int device = 0; device < 2; device++) {
        threads[device] = std::thread([&, device] {
            if (cudaSetDevice(device) != cudaSuccess) {
                return;
            }
            int *p;
            if (cudaMalloc(&p, n * sizeof(int)) != cudaSuccess) {
                return;
            }
            for (int round = 0; round < 50; round++) {
                fill<<<64, 256>>>(p, n, round + device * 100);
            }
            std::vector<int> h(n);
            ok[device] = cudaMemcpy(h.data(), p, n * sizeof(int), cudaMemcpyDefault) == cudaSuccess &&
                         h[5] == 49 + device * 100 + 5 && cudaFree(p) == cudaSuccess;
        });
    }
    for (auto &thread: threads) {
        thread.join();
    }
    EXPECT(ok[0] && ok[1], "thread per device: %d %d", ok[0], ok[1]);
    PASS("multi_gpu thread per device");
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        printf("PASS multi_gpu (skipped: needs 2 devices, found %d)\n", count);
        return 0;
    }
    const int n = 1 << 20;
    testWithoutPeerAccess(n);
    testPeerAccess(n);
    testCrossDeviceEvents();
    testThreadPerDevice(n);
    CHECK(cudaGetLastError());
    return 0;
}
