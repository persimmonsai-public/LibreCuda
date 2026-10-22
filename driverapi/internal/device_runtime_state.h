#ifndef LIBRECUDA_DEVICE_RUNTIME_STATE_H
#define LIBRECUDA_DEVICE_RUNTIME_STATE_H

#include "nvidia/nvtypes.h"

// matches DeviceRuntimeState in driverapi/kernels/device_runtime/device_runtime.cu
struct DeviceRuntimeState {
    NvU64 buffer;
    NvU32 capacity;
    NvU32 offset;
    NvU32 dropped;
    NvU32 padding;
    NvU64 heap_base;
    NvU64 heap_size;
    NvU64 heap_top;
    NvU64 free_lists[32][32];
    NvU64 launch_queue;
    NvU32 launch_capacity;
    NvU32 launch_count;
    NvU32 last_error;
    NvS32 device;
    NvS32 device_count;
    NvU32 padding2;
    NvU64 error_slots;
    NvU32 error_slot_sms;
    NvU32 error_slot_warps;
};

// per thread error entry (ErrorSlot in device_runtime.cu)
struct DeviceErrorSlot {
    NvU64 grid;
    NvU32 thread;
    NvU32 error;
};

#endif //LIBRECUDA_DEVICE_RUNTIME_STATE_H
