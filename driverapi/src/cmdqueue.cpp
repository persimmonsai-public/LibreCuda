#include "cmdqueue.h"

#include <algorithm>
#include <thread>
#include <cstring>

#include "nvidia/clc6c0.h"
#include "nvidia/clc6b5.h"
#include "nvidia/g_allclasses.h"

#include "librecuda_internal.h"

#include "nvidia/clc6c0qmd.h"
#include "nvidia/clcec0qmd.h"
#include "nvidia/nvmisc.h"

#define ENSURE_QUEUE_INITIALIZED() LIBRECUDA_VALIDATE(initialized, LIBRECUDA_ERROR_NOT_INITIALIZED)
// the context of a queue may have been destroyed before the queue
#define ENSURE_QUEUE_ATTACHED() LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT)
#define QUEUE_LOCK() std::lock_guard<std::recursive_mutex> queue_lock(mutex)


#define MAX_CMD_QUEUE_PAGE_SIZE 0x200000
#define MAX_SIGNAL_POOL_SIZE 64

#define KERNARGS_PAGE_SIZE (16 << 20)

// size of each of the two pinned buffers used to stage copies from/to pageable host memory
#define STAGING_BUFFER_SIZE (8 << 20)

// maximum number of bytes / elements the copy engine transfers with one LAUNCH_DMA
#define MAX_DMA_LINE_LENGTH 0x80000000ull

// bytes of constant bank 0 written by the driver before the kernel parameters on Blackwell
#define BLACKWELL_CBUF0_DRIVER_SIZE 0x380

static inline NvU32 hi_32(NvU64 value) {
    return static_cast<NvV32>(value >> 32);
}

static inline NvU32 lo_32(NvU64 value) {
    return static_cast<NvV32>(value & 0xFFFFFFFF);
}

#define U64_HI_32_BITS(value) hi_32(reinterpret_cast<NvU64>(value))
#define U64_LO_32_BITS(value) lo_32(reinterpret_cast<NvU64>(value))

struct LibreCUEvent_ {
    LibreCUcontext ctx;
    LibreCUstream_ *stream;
    NvSignal *signal{};

    /**
     * The value the signal is released with when the event is reached; incremented on every record
     */
    NvU64 target{};
};

LibreCUEvent_ *NewEvent() {
    return new LibreCUEvent_{.ctx = current_ctx};
}

LibreCUstream_ *EventGetStream(LibreCUEvent_ *pEvent) {
    return pEvent->stream;
}

void DeleteEvent(LibreCUEvent_ *pEvent) {
    if (pEvent->signal != nullptr) {
        ctxReleaseEventSignal(pEvent->ctx, pEvent->signal);
    }
    delete pEvent;
}

static NvMethod makeNvMethod(int subcommand, int method, int size, int typ) {
    return (typ << 28) | (size << 16) | (subcommand << 13) | (method >> 2);
}

static inline NvU64 ceilDiv(NvU64 a, NvU64 b) {
    return (a + b - 1) / b;
}

static inline NvU64 roundUp(NvU64 a, NvU64 b) {
    return ceilDiv(a, b) * b;
}

template<typename T>
static inline T maxOf(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
static inline T minOf(T a, T b) {
    return (a < b) ? a : b;
}

NvCommandQueue::NvCommandQueue(LibreCUcontext ctx) : ctx(ctx) {
    if (ctx->device->compute_class >= BLACKWELL_COMPUTE_A) {
        // Blackwell rejects the small legacy window addresses (Xid 69 on SET_SHADER_SHARED_MEMORY_WINDOW);
        // use 4GB-aligned windows far away from the UVM heap instead
        shared_mem_window = 0x729400000000;
        local_mem_window = 0x729300000000;
    }
}

libreCudaStatus_t NvCommandQueue::submitToFifo(QueueType queueType) {
    auto &commandBuffer = commandBuffers[queueType];

    if (commandBuffer.empty()) {
        return LIBRECUDA_SUCCESS; // don't do anything when command buffer is emtpy
    }
    LIBRECUDA_ERR_PROPAGATE(ctxCheckChannelError(ctx));

    auto &page = queuePages[queueType];
    auto &gpfifo = fifoOf(queueType);

    // room for the local memory methods that may be put in front (see below)
    constexpr size_t SLM_METHODS_SIZE = 7;
    LIBRECUDA_VALIDATE(commandBuffer.size() + SLM_METHODS_SIZE <= MAX_CMD_QUEUE_PAGE_SIZE,
                       LIBRECUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);

    // the page is used as a ring buffer; before wrapping around, all commands previously submitted from it
    // must have been consumed by the GPU
    if ((page.commandWriteIdx + commandBuffer.size() + SLM_METHODS_SIZE) > MAX_CMD_QUEUE_PAGE_SIZE) {
        LIBRECUDA_ERR_PROPAGATE(signalWaitCpu(timelineSignal, page.lastSubmittedTimelineValue));
        page.commandWriteIdx = 0;
    }

    // the gpfifo ring is shared by all queues of the context; before reusing an entry, wait until the commands it
    // referenced last time have executed
    std::lock_guard submit_lock(ctx->submit_mutex);

    // point the channel at the local memory buffer the kernels need, unless commands submitted earlier (possibly by
    // another queue) already pointed it at that or a larger one
    if (queueType == COMPUTE && pendingSlmPerThread > gpfifo.slm_per_thread) {
        const NvU32 slm_methods[SLM_METHODS_SIZE] = {
            static_cast<NvU32>(makeNvMethod(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_A, 2)),
            U64_HI_32_BITS(pendingSlmVa), U64_LO_32_BITS(pendingSlmVa),
            static_cast<NvU32>(makeNvMethod(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 3)),
            hi_32(pendingSlmBytesPerTpc), lo_32(pendingSlmBytesPerTpc), 0xff // max sm count
        };
        commandBuffer.insert(commandBuffer.begin(), std::begin(slm_methods), std::end(slm_methods));
        gpfifo.slm_va = pendingSlmVa;
        gpfifo.slm_bytes_per_tpc = pendingSlmBytesPerTpc;
        gpfifo.slm_per_thread = pendingSlmPerThread;
    }
    if (queueType == COMPUTE) {
        pendingSlmPerThread = 0;
    }

    const NvU32 ring_idx = gpfifo.put_value % gpfifo.entries_count;
    {
        auto [done_signal, done_value] = gpfifo.entry_done[ring_idx];
        if (done_signal != nullptr) {
            LIBRECUDA_ERR_PROPAGATE(ctxWaitForValue(ctx, &done_signal->value, done_value));
        }
    }

    memcpy(page.commandQueueSpace + page.commandWriteIdx, commandBuffer.data(), commandBuffer.size() * sizeof(NvU32));

    const NvU64 entry = ((NvU64(page.commandQueueSpace + page.commandWriteIdx) / sizeof(NvU32)) << 2) |
                        (commandBuffer.size() << 42) | (1l << 41);
    page.commandWriteIdx += commandBuffer.size();
    page.lastSubmittedTimelineValue = timelineCtr;
    kickRing(gpfifo, entry, timelineCtr);

    commandBuffer.clear();

    LIBRECUDA_SUCCEED();
}

void NvCommandQueue::kickRing(GPFifo &gpfifo, NvU64 entry, NvU32 doneValue) {
    const NvU32 ring_idx = gpfifo.put_value % gpfifo.entries_count;
    gpfifo.ring[ring_idx] = entry;
    gpfifo.entry_done[ring_idx] = {timelineSignal, doneValue};
    gpfifo.put_value++;
    ctxRingDoorbell(ctx, gpfifo, false);
}

// how long after a doorbell submissions are only collected (LIBRECUDA_DOORBELL_DELAY_US, 0: ring every time)
static std::chrono::microseconds doorbellDelay() {
    static const auto delay = [] {
        const char *env = getenv("LIBRECUDA_DOORBELL_DELAY_US");
        return std::chrono::microseconds(env != nullptr ? strtol(env, nullptr, 10) : 20);
    }();
    return delay;
}

void ctxRingDoorbell(LibreCUcontext_ *ctx, GPFifo &gpfifo, bool force) {
    const auto delay = doorbellDelay();
    constexpr NvU32 MAX_DEFERRED_ENTRIES = 16;
    const auto now = std::chrono::steady_clock::now();
    if (!force && delay.count() > 0 && gpfifo.deferred_entries + 1 < MAX_DEFERRED_ENTRIES &&
        now - gpfifo.last_doorbell < delay) {
        // a doorbell was rung just now: more submissions are likely to follow
        gpfifo.deferred_entries++;
        if (!ctx->doorbell_deferred.exchange(true)) {
            ctx->doorbell_cv.notify_one();
        }
        return;
    }
    gpfifo.deferred_entries = 0;
    gpfifo.last_doorbell = now;

    // command page and ring are write-combined; make the writes visible before ringing the doorbell
    __builtin_ia32_sfence();
    gpfifo.controls->GPPut = gpfifo.put_value % gpfifo.entries_count;
    __builtin_ia32_sfence();
    // command page, ring and USERD are written through BAR1, the doorbell through BAR0. The sfence only pushes the
    // writes out of the CPU; reading back through BAR1 guarantees they have reached video memory before the doorbell
    // makes the GPU fetch GPPut. Otherwise the GPU may see the old GPPut and never look at the new entry (like UVM's
    // dummy BAR1 read in uvm_hal_turing_host_write_gpu_put).
    (void) *static_cast<volatile NvU32 *>(&gpfifo.controls->GPPut);
    ctx->device->gpu_mmio[0x90 / 4] = gpfifo.token;
}

void ctxFlushDoorbells(LibreCUcontext_ *ctx) {
    if (!ctx->doorbell_deferred.load()) {
        return;
    }
    std::lock_guard submit_lock(ctx->submit_mutex);
    ctx->doorbell_deferred = false;
    for (GPFifo *fifo: ctxChannels(ctx)) {
        if (fifo->deferred_entries > 0) {
            ctxRingDoorbell(ctx, *fifo, true);
        }
    }
}

void ctxStartDoorbellThread(LibreCUcontext_ *ctx) {
    ctx->doorbell_thread = std::thread([ctx] {
        std::unique_lock submit_lock(ctx->submit_mutex);
        while (!ctx->doorbell_stop) {
            if (!ctx->doorbell_deferred) {
                ctx->doorbell_cv.wait(submit_lock);
                continue;
            }
            // let the submitting thread add more entries, then ring
            submit_lock.unlock();
            std::this_thread::sleep_for(doorbellDelay());
            submit_lock.lock();
            ctx->doorbell_deferred = false;
            for (GPFifo *fifo: ctxChannels(ctx)) {
                if (fifo->deferred_entries > 0) {
                    ctxRingDoorbell(ctx, *fifo, true);
                }
            }
        }
    });
}

void ctxStopDoorbellThread(LibreCUcontext_ *ctx) {
    if (!ctx->doorbell_thread.joinable()) {
        return;
    }
    ctxFlushDoorbells(ctx);
    {
        std::lock_guard submit_lock(ctx->submit_mutex);
        ctx->doorbell_stop = true;
    }
    ctx->doorbell_cv.notify_all();
    ctx->doorbell_thread.join();
}

libreCudaStatus_t NvCommandQueue::initializeQueue() {
    QUEUE_LOCK();
    {
        std::lock_guard ctx_lock(ctx->mutex);
        // registered first so a partially initialized queue is still detached when the context is destroyed
        ctx->queues.insert(this);
        // spread queues over the compute channels so that kernels of different streams run concurrently: the
        // channel with the fewest live queues (channels of destroyed queues are reused), round robin among equals
        const size_t channels = ctx->compute_gpfifos.size();
        std::vector<size_t> load(channels, 0);
        for (const NvCommandQueue *queue: ctx->queues) {
            if (queue != this) {
                load[queue->computeChannel]++;
            }
        }
        size_t best = ctx->next_compute_channel % channels;
        for (size_t i = 1; i < channels; i++) {
            const size_t c = (ctx->next_compute_channel + i) % channels;
            if (load[c] < load[best]) {
                best = c;
            }
        }
        computeChannel = static_cast<NvU32>(best);
        ctx->next_compute_channel = computeChannel + 1;
    }
    LIBRECUDA_ERR_PROPAGATE(ctxEnsureComputeChannel(ctx, computeChannel));

    // allocate a command queue page per channel
    for (int type = 0; type < NUM_QUEUE_TYPES; type++) {
        if (type == DMA_D2H && !ctx->has_d2h_channel) {
            continue;
        }
        NvU64 cmdq_va{};
        LIBRECUDA_ERR_PROPAGATE(
                gpuAlloc(ctx, MAX_CMD_QUEUE_PAGE_SIZE * sizeof(NvU32), false, false, true, 0, &cmdq_va));
        queuePages[type].commandQueueSpace = reinterpret_cast<NvU32 *>(cmdq_va);
    }

    // allocate signal pool
    {
        NvU64 signal_pool_va{};
        LIBRECUDA_ERR_PROPAGATE(
                gpuSystemAlloc(ctx, sizeof(NvSignal) * MAX_SIGNAL_POOL_SIZE, true, 0, &signal_pool_va)
        );
        signalPool = reinterpret_cast<NvSignal *>(signal_pool_va);

        // consider all signals free
        {
            for (int i = 0; i < MAX_SIGNAL_POOL_SIZE; i++) {
                freeSignals.push_back(i);
            }
        }
    }

    // create timeline signal
    {
        LIBRECUDA_ERR_PROPAGATE(obtainSignal(&timelineSignal));
    }

    // setup compute queue
    {
        LIBRECUDA_ERR_PROPAGATE(
                enqueue(makeNvMethod(1, NVC6C0_SET_OBJECT, 1), {ctx->device->compute_class}, COMPUTE)
        );
        LIBRECUDA_ERR_PROPAGATE(
                enqueue(
                        makeNvMethod(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, 2),
                        {U64_HI_32_BITS(local_mem_window), U64_LO_32_BITS(local_mem_window)},
                        COMPUTE
                )
        );
        LIBRECUDA_ERR_PROPAGATE(
                enqueue(
                        makeNvMethod(1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2),
                        {U64_HI_32_BITS(shared_mem_window), U64_LO_32_BITS(shared_mem_window)},
                        COMPUTE
                )
        );
        LIBRECUDA_ERR_PROPAGATE(completeOperation(COMPUTE, false));
    }
    LIBRECUDA_ERR_PROPAGATE(startExecution());
    LIBRECUDA_ERR_PROPAGATE(awaitExecution());

    // setup copy queue(s)
    for (QueueType type: {DMA, DMA_D2H}) {
        if (type == DMA_D2H && !ctx->has_d2h_channel) {
            continue;
        }
        LIBRECUDA_ERR_PROPAGATE(switchQueueType(type));
        LIBRECUDA_ERR_PROPAGATE(
                enqueue(makeNvMethod(4, NVC6C0_SET_OBJECT, 1), {ctx->device->dma_class}, type)
        );
        LIBRECUDA_ERR_PROPAGATE(completeOperation(type, false));
    }

    LIBRECUDA_ERR_PROPAGATE(startExecution());
    LIBRECUDA_ERR_PROPAGATE(awaitExecution());

    // allocate kernargs page
    {
        LIBRECUDA_ERR_PROPAGATE(
                gpuAlloc(
                        ctx,
                        KERNARGS_PAGE_SIZE,
                        false,
                        true,
                        true,
                        0,
                        reinterpret_cast<NvU64 *>(&kernArgsPageVa)
                )
        )
    }

    initialized = true;

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::allocKernArgs(NvU64 *pMemOut, size_t size) {
    LIBRECUDA_VALIDATE(pMemOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(size <= KERNARGS_PAGE_SIZE, LIBRECUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    if ((kernArgsWriteIdx + size) > KERNARGS_PAGE_SIZE) {
        // the page is full: wait until all kernels referencing it have completed, then start over
        LIBRECUDA_ERR_PROPAGATE(awaitExecution());
    }
    NvU8 *ptr = kernArgsPageVa + kernArgsWriteIdx;
    kernArgsWriteIdx += size;
    *pMemOut = reinterpret_cast<NvU64>(ptr);

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::enqueue(NvMethod method, std::initializer_list<NvU32> arguments, QueueType type) {
    ENSURE_QUEUE_ATTACHED();
    auto &commandBuffer = commandBuffers[type];
    size_t n_arguments = method >> 16 & 0xFFF;
    LIBRECUDA_VALIDATE(n_arguments == arguments.size(), LIBRECUDA_ERROR_INVALID_VALUE);

    commandBuffer.push_back(method);
    commandBuffer.insert(commandBuffer.end(), arguments.begin(), arguments.end());

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::obtainSignal(NvSignal **pSignalPtr) {
    LIBRECUDA_VALIDATE(pSignalPtr != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(signalPool != nullptr, LIBRECUDA_ERROR_NOT_INITIALIZED);

    if (freeSignals.empty()) {
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);
    }

    NvU32 signalIdx = freeSignals[0];
    freeSignals.pop_front();

    NvSignal *signalPointer = signalPool + signalIdx;
    signalPointer->value = 0;
    signalPointer->time_stamp = 0;
    *pSignalPtr = signalPointer;

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::releaseSignal(NvSignal *signal) {
    LIBRECUDA_VALIDATE(signal != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    // compute signal index from pointer
    NvU32 signalIdx = signal - signalPool;

    // ensure pointer was within signal pool and thus valid
    if (signalIdx >= MAX_SIGNAL_POOL_SIZE) {
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
    }

    // check if the signal has already been released
    if (std::find(freeSignals.begin(), freeSignals.end(), signalIdx) != freeSignals.end()) {
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
    }

    freeSignals.push_back(signalIdx);
    LIBRECUDA_SUCCEED();
}

void NvCommandQueue::detachFromContext() {
    QUEUE_LOCK();
    initialized = false;
    ctx = nullptr;
}

NvCommandQueue::~NvCommandQueue() {
    QUEUE_LOCK();
    if (ctx != nullptr) {
        std::lock_guard ctx_lock(ctx->mutex);
        ctx->queues.erase(this);
    }
    if (!initialized) {
        return;
    }
    // we unfortunately can't validate errors in the destructor; make sure the GPU no longer uses the queue's memory
    awaitExecution();
    // the timeline signal is about to be released; ring entries completed by it no longer need tracking (a
    // submission of another queue waiting for one of them holds the lock until the wait is over)
    std::lock_guard submit_lock(ctx->submit_mutex);
    for (GPFifo *fifo: ctxChannels(ctx)) {
        for (auto &entry: fifo->entry_done) {
            if (entry.first == timelineSignal) {
                entry = {nullptr, 0};
            }
        }
    }
    releaseSignal(timelineSignal);

    gpuFree(ctx, reinterpret_cast<NvU64>(signalPool));
    for (auto &page: queuePages) {
        if (page.commandQueueSpace != nullptr) {
            gpuFree(ctx, reinterpret_cast<NvU64>(page.commandQueueSpace));
        }
    }
    gpuFree(ctx, reinterpret_cast<NvU64>(kernArgsPageVa));
    for (auto staging_buffer: stagingBuffers) {
        if (staging_buffer != nullptr) {
            gpuFree(ctx, reinterpret_cast<NvU64>(staging_buffer));
        }
    }
}

libreCudaStatus_t NvCommandQueue::awaitExecution() {
    std::unique_lock<std::recursive_mutex> queue_lock(mutex);
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(timelineSignal != nullptr, LIBRECUDA_ERROR_NOT_INITIALIZED);
    LIBRECUDA_ERR_PROPAGATE(startExecution());
    const NvU32 target = timelineCtr;
    LibreCUcontext queue_ctx = ctx;
    NvSignal *signal = timelineSignal;
    // other threads may use the stream while this one waits (unless this thread holds the lock further up)
    queue_lock.unlock();
    LIBRECUDA_ERR_PROPAGATE(ctxWaitForValue(queue_ctx, &signal->value, target));
    queue_lock.lock();
    if (timelineCtr == target) {
        // nothing was enqueued meanwhile: no kernel uses the kernel argument page anymore
        kernArgsWriteIdx = 0;
    }
    // the work completing doesn't mean it succeeded: kernels that hit an SM exception complete and RM reports the
    // error only later (sm_exceptions.cpp), so check the error state even if no wait was necessary
    return ctxCheckChannelError(ctx);
}

bool NvCommandQueue::tryIsIdle() {
    std::unique_lock<std::recursive_mutex> queue_lock(mutex, std::try_to_lock);
    if (!queue_lock.owns_lock() || ctx == nullptr || timelineSignal == nullptr) {
        return false;
    }
    ctxFlushDoorbells(ctx);
    return std::all_of(std::begin(commandBuffers), std::end(commandBuffers),
                       [](const std::vector<NvU32> &buffer) { return buffer.empty(); }) &&
           timelineSignal->value >= timelineCtr;
}

libreCudaStatus_t NvCommandQueue::queryExecution() {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    ctxFlushDoorbells(ctx);
    LIBRECUDA_VALIDATE(timelineSignal != nullptr, LIBRECUDA_ERROR_NOT_INITIALIZED);
    if (std::any_of(std::begin(commandBuffers), std::end(commandBuffers),
                    [](const std::vector<NvU32> &buffer) { return !buffer.empty(); })) {
        return LIBRECUDA_ERROR_NOT_READY;
    }
    return timelineSignal->value >= timelineCtr ? ctxCheckChannelError(ctx) : LIBRECUDA_ERROR_NOT_READY;
}

libreCudaStatus_t NvCommandQueue::signalNotify(NvSignal *pSignal, NvU32 signalTarget, QueueType type) {
    LIBRECUDA_VALIDATE(pSignal != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    switch (type) {
        case COMPUTE: {
            // semaphore release after the engine is idle, i.e. after all preceding kernels completed
            LIBRECUDA_ERR_PROPAGATE(enqueue(
                    makeNvMethod(0, NVC56F_SEM_ADDR_LO, 5),
                    {
                            // little endian
                            U64_LO_32_BITS(pSignal),
                            U64_HI_32_BITS(pSignal),

                            // little endian
                            signalTarget,
                            0,

                            (NVC56F_SEM_EXECUTE_OPERATION_RELEASE) |
                            (NVC56F_SEM_EXECUTE_RELEASE_WFI_EN << 20) |
                            (NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT << 24) |
                            (NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN << 25)
                    },
                    COMPUTE
            ));
            break;
        }
        case DMA:
        case DMA_D2H: {
            LIBRECUDA_ERR_PROPAGATE(enqueue(
                    makeNvMethod(4, NVC6B5_SET_SEMAPHORE_A, 3),
                    {
                            U64_HI_32_BITS(pSignal),
                            U64_LO_32_BITS(pSignal),

                            signalTarget
                    },
                    type
            ));
            // four word semaphore: payload + timestamp, flushed so the CPU observes preceding copies
            LIBRECUDA_ERR_PROPAGATE(enqueue(
                    makeNvMethod(4, NVC6B5_LAUNCH_DMA, 1),
                    {
                            (NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE << 2) |
                            (NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE << 3)
                    },
                    type
            ));
            break;
        }
    }

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::signalWaitCpu(NvSignal *pSignal, NvU64 signalTarget) {
    LIBRECUDA_VALIDATE(pSignal != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(ctxWaitForValue(ctx, &pSignal->value, signalTarget));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::signalWaitGpu(NvSignal *pSignal, NvU64 signalTarget, QueueType type) {
    if (pSignal->value >= signalTarget) {
        // No need to wait, if cpu can confirm.
        return LIBRECUDA_SUCCESS;
    }
    // host (PBDMA) methods are available on every channel
    LIBRECUDA_ERR_PROPAGATE(enqueue(
            makeNvMethod(0, NVC56F_SEM_ADDR_LO, 5),
            {
                    // little endian
                    U64_LO_32_BITS(pSignal),
                    U64_HI_32_BITS(pSignal),

                    // little endian
                    lo_32(signalTarget),
                    hi_32(signalTarget),

                    (NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ) |
                    (NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT << 24)
            },
            type
    ));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::switchQueueType(QueueType type) {
    if (type == currentQueueType) {
        return LIBRECUDA_SUCCESS;
    }
    if (timelineNotifyPending) {
        LIBRECUDA_ERR_PROPAGATE(signalNotify(timelineSignal, timelineCtr, currentQueueType));
        timelineNotifyPending = false;
    }
    LIBRECUDA_ERR_PROPAGATE(submitToFifo(currentQueueType));
    currentQueueType = type;

    // everything enqueued so far (on the other engine) must complete before work on this engine starts
    LIBRECUDA_ERR_PROPAGATE(signalWaitGpu(timelineSignal, timelineCtr, type));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::completeOperation(QueueType type, bool deferNotify) {
    timelineCtr++;
    if (deferNotify) {
        timelineNotifyPending = true;
    } else {
        LIBRECUDA_ERR_PROPAGATE(signalNotify(timelineSignal, timelineCtr, type));
        timelineNotifyPending = false;
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::startExecution() {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    if (timelineNotifyPending) {
        LIBRECUDA_ERR_PROPAGATE(signalNotify(timelineSignal, timelineCtr, currentQueueType));
        timelineNotifyPending = false;
    }
    for (int type = 0; type < NUM_QUEUE_TYPES; type++) {
        LIBRECUDA_ERR_PROPAGATE(submitToFifo(static_cast<QueueType>(type)));
    }
    LIBRECUDA_SUCCEED();
}

GPFifo &NvCommandQueue::fifoOf(QueueType type) {
    switch (type) {
        case COMPUTE:
            return ctx->compute_gpfifos[computeChannel];
        case DMA_D2H:
            return ctx->dma_d2h_gpfifo;
        case DMA:
        default:
            return ctx->dma_gpfifo;
    }
}

QueueType NvCommandQueue::copyQueueFor(const void *dst, const void *src) const {
    // device to host copies use the copy engine that is attached for writing system memory
    if (ctx->has_d2h_channel && !isDevicePtr(const_cast<void *>(dst)) && isDevicePtr(const_cast<void *>(src))) {
        return DMA_D2H;
    }
    return DMA;
}

libreCudaStatus_t NvCommandQueue::ensureEnoughLocalMem(LibreCUFunction function) {
    QUEUE_LOCK();
    NvU32 localMemReq = function->local_mem_req;
    launchSlmPerThread = 0;
    if (localMemReq == 0) {
        return LIBRECUDA_SUCCESS;
    }
    // stack size of functions with calls (recursion, function pointers) cannot be determined statically;
    // like CUDA, provide at least the configured stack limit
    if (function->has_calls) {
        localMemReq = maxOf(localMemReq, function->local_mem_req - function->local_size_bytes + ctx->stack_limit);
    }
    // the buffer is the context's; take a consistent snapshot of it
    std::unique_lock ctx_lock(ctx->mutex);
    if (localMemReq > ctx->slm_per_thread) {
        // grow the context's local memory buffer
        const auto &info = ctx->device->device_info;
        const NvU64 max_warps_per_sm = maxOf(1u, info[NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM].data);
        const NvU64 sm_per_tpc = maxOf(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC].data);
        // local memory is partitioned per TPC slot: GPCs present (SHADER_PIPE_COUNT) x max TPCs per GPC.
        // Fall back to the architecture maximum if the GPC count is not reported.
        const NvU32 num_gpcs = info[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT].data != 0
                                   ? info[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT].data
                                   : info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS].data;
        const NvU64 num_tpcs = maxOf(1u, num_gpcs) *
                               maxOf(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC].data);

        const NvU32 slm_per_thread = roundUp(localMemReq, 32);
        const NvU64 bytes_per_warp = roundUp(slm_per_thread * 32ull, 0x200);
        const NvU64 bytes_per_tpc = roundUp(bytes_per_warp * max_warps_per_sm * sm_per_tpc, 0x8000);
        const NvU64 n_slm_bytes = roundUp(bytes_per_tpc * num_tpcs, 0x200000);

        NvU64 slm_va{};
        LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, n_slm_bytes, false, true, false, 0, &slm_va));

        // kernels in flight (on any compute channel) may still use the old allocation
        if (ctx->slm_va != 0) {
            ctx->retired_allocs.push_back(ctx->slm_va);
        }
        ctx->slm_va = slm_va;
        ctx->slm_per_thread = slm_per_thread;
        ctx->slm_bytes_per_tpc = bytes_per_tpc;
    }

    // the QMD declares the size of the current buffer, which the channel must be pointed at (or at a larger one)
    // before the kernel runs (submitToFifo)
    launchSlmPerThread = ctx->slm_per_thread;
    if (ctx->slm_per_thread > pendingSlmPerThread) {
        pendingSlmVa = ctx->slm_va;
        pendingSlmBytesPerTpc = ctx->slm_bytes_per_tpc;
        pendingSlmPerThread = ctx->slm_per_thread;
    }

    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t NvCommandQueue::ensureTexturePools() {
    QUEUE_LOCK();
    if (ctx->texture_header_pool_va == 0 || programmedTexturePools == ctx->texture_header_pool_va) {
        LIBRECUDA_SUCCEED();
    }
    const NvU64 headers = ctx->texture_header_pool_va, samplers = ctx->texture_sampler_pool_va;
    LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(1, NVC6C0_SET_TEX_HEADER_POOL_A, 3),
                                    {NvU32(headers >> 32), NvU32(headers), LIBRECUDA_TEXTURE_POOL_ENTRIES - 1}, COMPUTE));
    LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(1, NVC6C0_SET_TEX_SAMPLER_POOL_A, 3),
                                    {NvU32(samplers >> 32), NvU32(samplers), LIBRECUDA_TEXTURE_POOL_ENTRIES - 1}, COMPUTE));
    programmedTexturePools = headers;
    LIBRECUDA_SUCCEED();
}

// QMD V03 (Ampere/Ada) is 0x40 dwords, QMD V05 (Blackwell) is 0x60 dwords
typedef uint32_t qmd_cmd_t[0x60];

static NvU32 smemConfig(uint32_t shmem_usage) {
    uint32_t min = UINT32_MAX;
    for (auto shmem_conf: {32, 64, 100}) {
        if ((shmem_conf * 1024) >= shmem_usage) {
            min = minOf<uint32_t>(shmem_conf * 1024, min);
        }
    }
    return (min / 4096) + 1;
}

libreCudaStatus_t
NvCommandQueue::buildQmdV05(LibreCUFunction function,
                            uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                            uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                            uint32_t shmem_usage, NvU64 kernargs_pointer, uint32_t *qmd_data,
                            const NvU32 *clusterDims) {
    // SASS version is derived from the raw GR SM version, e.g. 0xa04 -> 0xa4
    const NvU32 sm_version = ctx->device->device_info[NV2080_CTRL_GR_INFO_INDEX_SM_VERSION].data;
    const NvU32 sass_version = ((sm_version & 0xf00) >> 4) | (sm_version & 0xf);
    const NvU32 smem_config = smemConfig(shmem_usage);
    const NvU64 prog = function->func_va_addr;

    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_QMD_MAJOR_VERSION, , , 5, qmd_data);
    if (clusterDims[0] * clusterDims[1] * clusterDims[2] > 1) {
        // thread block clusters: the CTAs of a GPC CGA (cluster) are launched together on one GPC, and the grid
        // dimensions count clusters
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_QMD_TYPE, , , NVCEC0_QMDV05_00_QMD_TYPE_GRID_GPC_CGA, qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GPC_CGA_WIDTH, , , clusterDims[0], qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GPC_CGA_HEIGHT, , , clusterDims[1], qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GPC_CGA_DEPTH, , , clusterDims[2], qmd_data);
        gridDimX /= clusterDims[0];
        gridDimY /= clusterDims[1];
        gridDimZ /= clusterDims[2];
    } else {
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_QMD_TYPE, , , NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, qmd_data);
    }
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_QMD_GROUP_ID, , , 0x3F, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_INVALIDATE_TEXTURE_HEADER_CACHE, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_INVALIDATE_TEXTURE_SAMPLER_CACHE, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_INVALIDATE_TEXTURE_DATA_CACHE, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_INVALIDATE_SHADER_DATA_CACHE, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_API_VISIBLE_CALL_LIMIT, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_SAMPLER_INDEX, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_BARRIER_COUNT, , , 1, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CWD_MEMBAR_TYPE, , , NVCEC0_QMDV05_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_SASS_VERSION, , , sass_version, qmd_data);

    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_REGISTER_COUNT, , , function->num_registers, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7, , ,
                       maxOf<NvU64>(0x400u, roundUp(shmem_usage, 0x100u)) >> 7, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_MIN_SM_CONFIG_SHARED_MEM_SIZE, , , smem_config, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_MAX_SM_CONFIG_SHARED_MEM_SIZE, , , 0x1A, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE, , , smem_config, qmd_data);
    if (function->local_mem_req != 0) {
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4, , , launchSlmPerThread >> 4,
                           qmd_data);
    }

    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_PROGRAM_ADDRESS_LOWER_SHIFTED4, , , lo_32(prog >> 4), qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_PROGRAM_ADDRESS_UPPER_SHIFTED4, , , hi_32(prog >> 4), qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_PROGRAM_PREFETCH_SIZE, , , minOf<NvU64>(function->function_size >> 8, 0x1ff),
                       qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, , , lo_32(prog >> 8), qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, , , hi_32(prog >> 8), qmd_data);

    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GRID_WIDTH, , , gridDimX, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GRID_HEIGHT, , , gridDimY, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_GRID_DEPTH, , , gridDimZ, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CTA_THREAD_DIMENSION0, , , blockDimX, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CTA_THREAD_DIMENSION1, , , blockDimY, qmd_data);
    FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CTA_THREAD_DIMENSION2, , , blockDimZ, qmd_data);

    // constant buffers; addresses are stored shifted by 6, so they must be 64-byte aligned
    for (const auto &constant_info: function->constants) {
        const NvU64 addr = constant_info.const_nr == 0 ? kernargs_pointer : constant_info.address;
        LIBRECUDA_VALIDATE((addr & 0x3F) == 0, LIBRECUDA_ERROR_INVALID_VALUE);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(constant_info.const_nr), , ,
                           lo_32(addr >> 6), qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(constant_info.const_nr), , ,
                           hi_32(addr >> 6), qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CONSTANT_BUFFER_SIZE_SHIFTED4(constant_info.const_nr), , ,
                           roundUp(constant_info.size, 16) >> 4, qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CONSTANT_BUFFER_VALID(constant_info.const_nr), , ,
                           NVCEC0_QMDV05_00_CONSTANT_BUFFER_VALID_TRUE, qmd_data);
        FLD_SET_DRF_NUM_MW(CEC0_QMDV05_00_CONSTANT_BUFFER_INVALIDATE(constant_info.const_nr), , , 1, qmd_data);
    }
    LIBRECUDA_SUCCEED();
}

NvU64 NvCommandQueue::launchMemorySize(LibreCUFunction function, bool cooperative) const {
    // constant bank 0 (driver area + kernel parameters), the QMD after it and, for cooperative launches, their own
    // grid workspace (cooperative_groups' grid barrier) so that overlapping cooperative kernels never share a barrier.
    // The QMD address is passed shifted by 8, so every part keeps 256 byte alignment.
    const NvU64 cbuf0_size = function->constants.empty() ? 0 : function->constants[0].size;
    const NvU64 qmd_offset = roundUp(cbuf0_size, 1ull << 8);
    const NvU64 grid_ws_offset = roundUp(qmd_offset + sizeof(qmd_cmd_t), 1ull << 8);
    return grid_ws_offset + (cooperative ? (1ull << 8) : 0);
}

libreCudaStatus_t
NvCommandQueue::writeLaunch(LibreCUFunction function,
                            uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                            uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                            uint32_t sharedMemBytes, void **params, size_t numParams, bool cooperative,
                            const NvU32 *clusterDims, NvU64 kernargs_pointer, NvU64 *pQmdAddr) {
    // validate const0 exists
    LIBRECUDA_VALIDATE(!function->constants.empty() && function->constants[0].const_nr == 0,
                       LIBRECUDA_ERROR_INVALID_IMAGE);

    // check launch dimensions
    const NvU64 block_threads = NvU64(blockDimX) * blockDimY * blockDimZ;
    LIBRECUDA_VALIDATE(gridDimX >= 1 && gridDimY >= 1 && gridDimZ >= 1, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(blockDimX >= 1 && blockDimY >= 1 && blockDimZ >= 1, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(blockDimX <= 1024 && blockDimY <= 1024 && blockDimZ <= 64, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(gridDimX <= 0x7fffffff && gridDimY <= 65535 && gridDimZ <= 65535,
                       LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(block_threads <= minOf<NvU64>(1024, function->max_threads_per_block),
                       LIBRECUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    {
        // registers are allocated per warp with a granularity of 256 registers
        const NvU64 regs_per_warp = roundUp(maxOf<NvU64>(1, function->num_registers) * 32, 256);
        LIBRECUDA_VALIDATE(regs_per_warp * ceilDiv(block_threads, 32) <= 65536,
                           LIBRECUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    }

    // thread block clusters: compiled into the kernel (__cluster_dims__) or chosen at launch for kernels without
    NvU32 cluster[3] = {function->cluster_dims[0], function->cluster_dims[1], function->cluster_dims[2]};
    bool explicit_cluster = function->explicit_cluster;
    if (clusterDims != nullptr) {
        const bool compiled_dims = function->cluster_dims[0] * function->cluster_dims[1] * function->cluster_dims[2] > 1;
        for (int d = 0; d < 3; d++) {
            LIBRECUDA_VALIDATE(clusterDims[d] >= 1, LIBRECUDA_ERROR_INVALID_CLUSTER_SIZE);
            LIBRECUDA_VALIDATE(!compiled_dims || clusterDims[d] == function->cluster_dims[d],
                               LIBRECUDA_ERROR_INVALID_CLUSTER_SIZE);
            cluster[d] = clusterDims[d];
        }
        explicit_cluster = true;
    }
    if (explicit_cluster) {
        const bool clusters_supported = ctx->device->compute_class >= BLACKWELL_COMPUTE_A;
        LIBRECUDA_VALIDATE(clusters_supported, LIBRECUDA_ERROR_NOT_SUPPORTED);
        LIBRECUDA_VALIDATE(!cooperative, LIBRECUDA_ERROR_NOT_SUPPORTED);
        // larger (non-portable) clusters would need the QMD's LARGE_GPC_CGA fields
        LIBRECUDA_VALIDATE(NvU64(cluster[0]) * cluster[1] * cluster[2] <= LIBRECUDA_MAX_CLUSTER_SIZE,
                           LIBRECUDA_ERROR_INVALID_CLUSTER_SIZE);
        LIBRECUDA_VALIDATE(gridDimX % cluster[0] == 0 && gridDimY % cluster[1] == 0 && gridDimZ % cluster[2] == 0,
                           LIBRECUDA_ERROR_INVALID_CLUSTER_SIZE);
    }

    // like CUDA, more than 48 KiB of dynamic shared memory requires opting in via the function attribute
    LIBRECUDA_VALIDATE(sharedMemBytes <= function->max_dynamic_shared_mem, LIBRECUDA_ERROR_INVALID_VALUE);

    // dynamic shared memory is allocated after the static shared memory
    const uint32_t shmem_usage = function->shared_mem + sharedMemBytes;
    LIBRECUDA_VALIDATE(shmem_usage <= (100 << 10), LIBRECUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);

    // Blackwell moved the driver params in constbuf0 and uses QMD V05
    const bool is_blackwell = ctx->device->compute_class >= BLACKWELL_COMPUTE_A;
    const NvU64 cbuf0_size = function->constants[0].size;
    const NvU64 qmd_offset = roundUp(cbuf0_size, 1ull << 8);
    const NvU64 grid_ws_offset = roundUp(qmd_offset + sizeof(qmd_cmd_t), 1ull << 8);

    // populate constant bank 0
    {
        NvU32 driver_data[BLACKWELL_CBUF0_DRIVER_SIZE / sizeof(NvU32)] = {};
        // the driver area ends where the parameters start (0x160 before Blackwell, 0x380 on Blackwell). Kernels without
        // parameters carry no parameter base in their attributes, so the area is sized by the architecture.
        const size_t driver_area = is_blackwell ? BLACKWELL_CBUF0_DRIVER_SIZE : 0x160;
        const size_t driver_size = minOf<size_t>(minOf<size_t>(driver_area, cbuf0_size), sizeof(driver_data));
        LIBRECUDA_VALIDATE(function->param_info.empty() || function->param_base >= driver_size,
                           LIBRECUDA_ERROR_INVALID_IMAGE);

        // populate ntid (blockDim.x, ...) and nctaid (gridDim.x, ...) registers;
        // Blackwell moved them to c[0][0x360] and c[0][0x370]
        const size_t ntid_off = is_blackwell ? 216 : 0;
        const size_t nctaid_off = is_blackwell ? 220 : 3;
        driver_data[ntid_off + 0] = blockDimX;
        driver_data[ntid_off + 1] = blockDimY;
        driver_data[ntid_off + 2] = blockDimZ;

        driver_data[nctaid_off + 0] = gridDimX;
        driver_data[nctaid_off + 1] = gridDimY;
        driver_data[nctaid_off + 2] = gridDimZ;

        // little endian
        const size_t windows_off = is_blackwell ? 188 : 6;
        driver_data[windows_off + 0] = U64_LO_32_BITS(shared_mem_window);
        driver_data[windows_off + 1] = U64_HI_32_BITS(shared_mem_window);

        driver_data[windows_off + 2] = U64_LO_32_BITS(local_mem_window);
        driver_data[windows_off + 3] = U64_HI_32_BITS(local_mem_window);

        // cooperative_groups reads the grid workspace address from %envreg1 (high half) and %envreg2 (low half),
        // which ptxas compiles to c[0][0x254] / c[0][0x258] on sm_120 and c[0][0x8c] / c[0][0x90] on sm_75..89.
        // Non-cooperative launches leave them 0, so this_grid().is_valid() is false.
        if (cooperative) {
            // grid_workspace {wsSize, barrier}
            const NvU32 grid_ws[2] = {2 * sizeof(NvU32), 0};
            const NvU64 grid_ws_va = kernargs_pointer + grid_ws_offset;
            memcpy(reinterpret_cast<void *>(grid_ws_va), grid_ws, sizeof(grid_ws));
            const size_t envreg1_off = is_blackwell ? 0x254 / 4 : 0x8c / 4;
            LIBRECUDA_VALIDATE((envreg1_off + 2) * sizeof(NvU32) <= driver_size, LIBRECUDA_ERROR_INVALID_IMAGE);
            driver_data[envreg1_off] = hi_32(grid_ws_va);
            driver_data[envreg1_off + 1] = lo_32(grid_ws_va);
        }

        // %current_graph_exec (cudaGetCurrentGraphExec): c[0][0x2e8] on sm_120, c[0][0x120] on sm_75..89
        {
            const size_t graph_exec_off = is_blackwell ? 0x2e8 / 4 : 0x120 / 4;
            if ((graph_exec_off + 2) * sizeof(NvU32) <= driver_size) {
                driver_data[graph_exec_off] = lo_32(graphExec);
                driver_data[graph_exec_off + 1] = hi_32(graphExec);
            }
        }

        // device side printf finds its state through %envreg8 (high half) and %envreg9 (low half)
        if (ctx->printf_state_va != 0) {
            const size_t envreg8_off = is_blackwell ? 0x270 / 4 : 0xa8 / 4;
            if ((envreg8_off + 2) * sizeof(NvU32) <= driver_size) {
                driver_data[envreg8_off] = hi_32(ctx->printf_state_va);
                driver_data[envreg8_off + 1] = lo_32(ctx->printf_state_va);
            }
        }

        // thread block cluster: dimensions (c[0][0x2a0]), their reciprocals as floats (0x2b0), clusters in the grid
        // (0x2c0), blocks per cluster (0x2cc), explicit cluster flag (0x36c). Kernels launched without clusters
        // see clusters of one block.
        if (is_blackwell) {
            const NvU32 grid[3] = {gridDimX, gridDimY, gridDimZ};
            for (int d = 0; d < 3; d++) {
                driver_data[0x2a0 / 4 + d] = cluster[d];
                const float reciprocal = 1.0f / static_cast<float>(cluster[d]);
                memcpy(&driver_data[0x2b0 / 4 + d], &reciprocal, sizeof(reciprocal));
                driver_data[0x2c0 / 4 + d] = grid[d] / cluster[d];
            }
            driver_data[0x2cc / 4] = cluster[0] * cluster[1] * cluster[2];
            driver_data[0x36c / 4] = explicit_cluster ? 1 : 0;
        }

        // initial stack pointer
        if (is_blackwell) {
            driver_data[223] = 0xfffdc0;
        } else {
            driver_data[10] = 0xfffdc0;
            driver_data[11] = 0;
        }

        auto *cbuf0 = reinterpret_cast<NvU8 *>(kernargs_pointer);
        memcpy(cbuf0, driver_data, driver_size);

        // kernel parameters at the offsets recorded by the toolchain
        for (size_t i = 0; i < numParams; i++) {
            const auto &param_info = function->param_info[i];
            const size_t param_off = function->param_base + param_info.param_offset;
            LIBRECUDA_VALIDATE(param_off + param_info.param_size <= cbuf0_size, LIBRECUDA_ERROR_INVALID_IMAGE);
            LIBRECUDA_VALIDATE(params[i] != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
            memcpy(cbuf0 + param_off, params[i], param_info.param_size);
        }
    }

    qmd_cmd_t qmd_data{};
    size_t qmd_size = 0x40 * sizeof(NvU32);
    if (is_blackwell) {
        LIBRECUDA_ERR_PROPAGATE(buildQmdV05(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                            shmem_usage, kernargs_pointer, qmd_data, cluster));
        qmd_size = 0x60 * sizeof(NvU32);
    } else {
#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-branch-clone"

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_QMD_GROUP_ID, , , 0x3F, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SM_GLOBAL_CACHING_ENABLE, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_HEADER_CACHE, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_SAMPLER_CACHE, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_TEXTURE_DATA_CACHE, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_API_VISIBLE_CALL_LIMIT, , , 1, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SAMPLER_INDEX, , , 1, qmd_data);

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CWD_MEMBAR_TYPE, , , NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_QMD_MAJOR_VERSION, , , 3, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_INVALIDATE(0), , , 1, qmd_data);

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SHARED_MEMORY_SIZE, , , maxOf<NvU64>(0x400u, roundUp(shmem_usage, 0x100u)),
                           qmd_data);

        const uint32_t smem_config = smemConfig(shmem_usage);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE, , , smem_config, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE, , , 0x1A, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_REGISTER_COUNT_V, , , function->num_registers, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE, , , smem_config, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_BARRIER_COUNT, , , 1, qmd_data);
        if (function->local_mem_req != 0) {
            FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE, , , launchSlmPerThread, qmd_data);
        }
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_SIZE, , , function->function_size >> 8, qmd_data);

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, , , U64_LO_32_BITS(function->func_va_addr), qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, , , U64_HI_32_BITS(function->func_va_addr), qmd_data);

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_SASS_VERSION, , , 0x89,
                           qmd_data); // TODO: HARDCODING THIS SEEMS DESTINED FOR DISASTER
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED, , , function->func_va_addr >> 8,
                           qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, , , function->func_va_addr >> 40,
                           qmd_data);

        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_WIDTH, , , gridDimX, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_HEIGHT, , , gridDimY, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_RASTER_DEPTH, , , gridDimZ, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION0, , , blockDimX, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION1, , , blockDimY, qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CTA_THREAD_DIMENSION2, , , blockDimZ, qmd_data);

        // setup constant buffers
        const auto &constants = function->constants;
        for (auto constant_info: constants) {
            FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(constant_info.const_nr), , ,
                               U64_HI_32_BITS(constant_info.address),
                               qmd_data);
            FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(constant_info.const_nr), , ,
                               U64_LO_32_BITS(constant_info.address),
                               qmd_data);
            FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(constant_info.const_nr), , ,
                               roundUp(constant_info.size, 16) >> 4, qmd_data);
            FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_VALID(constant_info.const_nr), , ,
                               NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE,
                               qmd_data);
        }

        // setup param pointer
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0), , , U64_HI_32_BITS(kernargs_pointer),
                           qmd_data);
        FLD_SET_DRF_NUM_MW(C6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0), , , U64_LO_32_BITS(kernargs_pointer),
                           qmd_data);
#pragma clang diagnostic pop
    }

    // the QMD after the kernel arguments
    *pQmdAddr = kernargs_pointer + qmd_offset;
    memcpy(reinterpret_cast<void *>(*pQmdAddr), qmd_data, qmd_size);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t
NvCommandQueue::launchFunction(LibreCUFunction function,
                               uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                               uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                               uint32_t sharedMemBytes,
                               void **params, size_t numParams,
                               bool async, bool cooperative, const NvU32 *clusterDims) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(numParams == function->param_info.size(), LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(numParams == 0 || params != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(!function->constants.empty() && function->constants[0].const_nr == 0,
                       LIBRECUDA_ERROR_INVALID_IMAGE);

    LIBRECUDA_ERR_PROPAGATE(switchQueueType(COMPUTE));
    LIBRECUDA_ERR_PROPAGATE(ensureEnoughLocalMem(function));
    LIBRECUDA_ERR_PROPAGATE(ensureTexturePools());

    // A non-async launch must not overlap with preceding (async) kernels. The deferred release has wait-for-idle
    // semantics and blocks the channel until those kernels completed.
    if (!async && timelineNotifyPending) {
        LIBRECUDA_ERR_PROPAGATE(signalNotify(timelineSignal, timelineCtr, COMPUTE));
        timelineNotifyPending = false;
    }

    NvU64 kernargs_pointer{}, qmd_addr{};
    LIBRECUDA_ERR_PROPAGATE(allocKernArgs(&kernargs_pointer, launchMemorySize(function, cooperative)));
    LIBRECUDA_ERR_PROPAGATE(writeLaunch(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                        sharedMemBytes, params, numParams, cooperative, clusterDims, kernargs_pointer,
                                        &qmd_addr));

    // memory barrier
    {
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(1, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1),
                {
                        (NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE << 12) |
                        (NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE << 4) |
                        (NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE << 0)
                },
                COMPUTE
        ));
    }

    // commence launch
    {
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(1, NVC6C0_SEND_PCAS_A, 0x1),
                {NvU32(qmd_addr >> 8)},
                COMPUTE
        ));

        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(1, NVC6C0_SEND_SIGNALING_PCAS2_B, 0x1),
                {0x9}, // PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
                COMPUTE
        ));
    }

    ctx->launch_count.fetch_add(1, std::memory_order_relaxed);
    ctx->last_launch_time.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);

    // when async, the release is deferred so that consecutive async kernels may run concurrently.
    // timelineNotifyPending tells subsequent operations / startExecution to issue the release.
    LIBRECUDA_ERR_PROPAGATE(completeOperation(COMPUTE, async));
    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t NvCommandQueue::gpuMemcpy(void *dst, const void *src, size_t numBytes, bool async) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(dst != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(src != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    if (numBytes == 0) {
        LIBRECUDA_SUCCEED();
    }

    const QueueType queue = copyQueueFor(dst, src);
    LIBRECUDA_ERR_PROPAGATE(switchQueueType(queue));

    bool first = true;
    auto dst_addr = reinterpret_cast<NvU64>(dst);
    auto src_addr = reinterpret_cast<NvU64>(src);
    while (numBytes > 0) {
        const NvU64 n = minOf<NvU64>(numBytes, MAX_DMA_LINE_LENGTH);
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_OFFSET_IN_UPPER, 4),
                {
                        hi_32(src_addr),
                        lo_32(src_addr),

                        hi_32(dst_addr),
                        lo_32(dst_addr),
                },
                queue
        ));

        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_LINE_LENGTH_IN, 1),
                {
                        static_cast<NvU32>(n)
                },
                queue
        ));

        // the first transfer of an operation must not overlap with preceding ones (they may produce its input)
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_LAUNCH_DMA, 1),
                {
                        (static_cast<NvU32>(first ? NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
                                                  : NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED) << 0) |
                        (NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH << 7) |
                        (NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH << 8)
                },
                queue
        ));
        first = false;
        numBytes -= n;
        dst_addr += n;
        src_addr += n;
    }
    LIBRECUDA_ERR_PROPAGATE(completeOperation(queue, async));
    LIBRECUDA_SUCCEED();
}

// SET_*_BLOCK_SIZE fields of the copy engine; KIND_BPP (Blackwell's GOB formats for 8 / 16 bit texels) from NVIDIA's
// clcab5.h (open-gpu-doc), the others as in clc6b5.h
static NvU32 copyBlockSize(const CopySurface &surface) {
    return (0u << 0) |                                  // WIDTH: blocks are one GOB wide
           (surface.blockHeightLog2 << 4) |              // HEIGHT
           (surface.blockDepthLog2 << 8) |               // DEPTH
           (1u << 12) |                                  // GOB_HEIGHT_FERMI_8
           (surface.gobKind << 16);                      // KIND_BPP
}

libreCudaStatus_t NvCommandQueue::gpuCopyRegion(const CopySurface &dst, const CopySurface &src, NvU32 widthBytes,
                                                NvU32 height, NvU32 depth, bool async) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    if (widthBytes == 0 || height == 0 || depth == 0) {
        LIBRECUDA_SUCCEED();
    }
    const QueueType queue = copyQueueFor(reinterpret_cast<void *>(dst.address), reinterpret_cast<void *>(src.address));
    LIBRECUDA_ERR_PROPAGATE(switchQueueType(queue));
    for (NvU32 z = 0; z < depth; z++) {
        // pitch layout: the slice's address; block linear: the surface's, with the slice as the layer
        const NvU64 src_address = src.blockLinear ? src.address : src.address + z * src.slicePitch;
        const NvU64 dst_address = dst.blockLinear ? dst.address : dst.address + z * dst.slicePitch;
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_OFFSET_IN_UPPER, 8),
                {
                        hi_32(src_address), lo_32(src_address),
                        hi_32(dst_address), lo_32(dst_address),
                        static_cast<NvU32>(src.pitch), static_cast<NvU32>(dst.pitch), // PITCH_IN / PITCH_OUT
                        widthBytes, height                                           // LINE_LENGTH_IN / LINE_COUNT
                },
                queue
        ));
        if (src.blockLinear) {
            LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(4, NVC6B5_SET_SRC_BLOCK_SIZE, 5),
                                            {copyBlockSize(src), static_cast<NvU32>(src.pitch), src.height, src.depth,
                                             src.z + z}, queue));
            LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(4, NVC6B5_SRC_ORIGIN_X, 2), {src.x, src.y}, queue));
        }
        if (dst.blockLinear) {
            LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(4, NVC6B5_SET_DST_BLOCK_SIZE, 5),
                                            {copyBlockSize(dst), static_cast<NvU32>(dst.pitch), dst.height, dst.depth,
                                             dst.z + z}, queue));
            LIBRECUDA_ERR_PROPAGATE(enqueue(makeNvMethod(4, NVC6B5_DST_ORIGIN_X, 2), {dst.x, dst.y}, queue));
        }
        // the first transfer must not overlap with preceding work (it may produce its input)
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_LAUNCH_DMA, 1),
                {
                        (static_cast<NvU32>(z == 0 ? NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
                                                   : NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED) << 0) |
                        (NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE << 2) |
                        ((src.blockLinear ? NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_BLOCKLINEAR
                                          : NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH) << 7) |
                        ((dst.blockLinear ? NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR
                                          : NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH) << 8) |
                        (NVC6B5_LAUNCH_DMA_MULTI_LINE_ENABLE_TRUE << 9)
                },
                queue
        ));
    }
    LIBRECUDA_ERR_PROPAGATE(completeOperation(queue, async));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::gpuMemset(void *dst, NvU32 value, size_t elementSize, size_t numElements) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(dst != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(elementSize == 1 || elementSize == 2 || elementSize == 4, LIBRECUDA_ERROR_INVALID_VALUE);
    if (numElements == 0) {
        LIBRECUDA_SUCCEED();
    }
    LIBRECUDA_ERR_PROPAGATE(switchQueueType(DMA));

    const NvU32 component_size = elementSize == 1 ? NVC6B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_ONE
                                                  : elementSize == 2 ? NVC6B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_TWO
                                                                     : NVC6B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_FOUR;
    // like the UVM driver's copy engine memset: every destination element is the constant B
    LIBRECUDA_ERR_PROPAGATE(enqueue(
            makeNvMethod(4, NVC6B5_SET_REMAP_CONST_B, 2),
            {
                    value,
                    (NVC6B5_SET_REMAP_COMPONENTS_DST_X_CONST_B << 0) |
                    (component_size << 16) |
                    (NVC6B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_ONE << 24)
            },
            DMA
    ));

    bool first = true;
    auto dst_addr = reinterpret_cast<NvU64>(dst);
    while (numElements > 0) {
        const NvU64 n = minOf<NvU64>(numElements, MAX_DMA_LINE_LENGTH);
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_OFFSET_OUT_UPPER, 2),
                {hi_32(dst_addr), lo_32(dst_addr)},
                DMA
        ));
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_LINE_LENGTH_IN, 1),
                {static_cast<NvU32>(n)},
                DMA
        ));
        LIBRECUDA_ERR_PROPAGATE(enqueue(
                makeNvMethod(4, NVC6B5_LAUNCH_DMA, 1),
                {
                        (static_cast<NvU32>(first ? NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
                                                  : NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED) << 0) |
                        (NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH << 7) |
                        (NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH << 8) |
                        (NVC6B5_LAUNCH_DMA_REMAP_ENABLE_TRUE << 10)
                },
                DMA
        ));
        first = false;
        numElements -= n;
        dst_addr += n * elementSize;
    }
    LIBRECUDA_ERR_PROPAGATE(completeOperation(DMA, false));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::ensureStagingBuffers() {
    for (auto &staging_buffer: stagingBuffers) {
        if (staging_buffer == nullptr) {
            NvU64 va{};
            LIBRECUDA_ERR_PROPAGATE(gpuHostAlloc(ctx, STAGING_BUFFER_SIZE, &va));
            staging_buffer = reinterpret_cast<NvU8 *>(va);
        }
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::stagedMemcpy(void *dst, const void *src, size_t numBytes, bool dstIsHost) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(dst != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(src != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(ensureStagingBuffers());

    const size_t num_chunks = ceilDiv(numBytes, STAGING_BUFFER_SIZE);
    auto chunk_size = [&](size_t chunk) {
        return minOf<size_t>(STAGING_BUFFER_SIZE, numBytes - chunk * STAGING_BUFFER_SIZE);
    };
    // timeline value after which the staging buffer of a chunk can be reused / read
    NvU32 chunk_done[2] = {0, 0};

    if (!dstIsHost) {
        // host -> GPU: the CPU fills one staging buffer while the copy engine drains the other one
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            const size_t slot = chunk % 2;
            LIBRECUDA_ERR_PROPAGATE(signalWaitCpu(timelineSignal, chunk_done[slot]));
            memcpy(stagingBuffers[slot], static_cast<const NvU8 *>(src) + chunk * STAGING_BUFFER_SIZE,
                   chunk_size(chunk));
            LIBRECUDA_ERR_PROPAGATE(gpuMemcpy(static_cast<NvU8 *>(dst) + chunk * STAGING_BUFFER_SIZE,
                                              stagingBuffers[slot], chunk_size(chunk), false));
            LIBRECUDA_ERR_PROPAGATE(startExecution());
            chunk_done[slot] = timelineCtr;
        }
        LIBRECUDA_ERR_PROPAGATE(signalWaitCpu(timelineSignal, timelineCtr));
    } else {
        // GPU -> host: the copy engine fills one staging buffer while the CPU drains the other one
        auto enqueue_chunk = [&](size_t chunk) -> libreCudaStatus_t {
            const size_t slot = chunk % 2;
            LIBRECUDA_ERR_PROPAGATE(gpuMemcpy(stagingBuffers[slot],
                                              static_cast<const NvU8 *>(src) + chunk * STAGING_BUFFER_SIZE,
                                              chunk_size(chunk), false));
            LIBRECUDA_ERR_PROPAGATE(startExecution());
            chunk_done[slot] = timelineCtr;
            LIBRECUDA_SUCCEED();
        };
        for (size_t chunk = 0; chunk < minOf<size_t>(2, num_chunks); chunk++) {
            LIBRECUDA_ERR_PROPAGATE(enqueue_chunk(chunk));
        }
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            const size_t slot = chunk % 2;
            LIBRECUDA_ERR_PROPAGATE(signalWaitCpu(timelineSignal, chunk_done[slot]));
            memcpy(static_cast<NvU8 *>(dst) + chunk * STAGING_BUFFER_SIZE, stagingBuffers[slot], chunk_size(chunk));
            if (chunk + 2 < num_chunks) {
                LIBRECUDA_ERR_PROPAGATE(enqueue_chunk(chunk + 2));
            }
        }
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::recordEvent(LibreCUEvent event, LibreCUstream_ *pStream) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    if (event->signal == nullptr) {
        LIBRECUDA_ERR_PROPAGATE(ctxObtainEventSignal(ctx, &event->signal));
        event->ctx = ctx;
    }
    event->stream = pStream;
    event->target++;

    // the release executes after all preceding work of the stream: work on the current channel precedes it
    // in channel order, work on the other channel was awaited when switching to the current channel.
    LIBRECUDA_ERR_PROPAGATE(signalNotify(event->signal, static_cast<NvU32>(event->target), currentQueueType));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::waitEventGpu(LibreCUEvent event) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    if (event->signal == nullptr) {
        LIBRECUDA_SUCCEED(); // never recorded: nothing to wait for
    }
    if (event->signal->value >= event->target) {
        LIBRECUDA_SUCCEED();
    }
    // The wait executes on the current channel only. Making it an operation on the timeline orders work that is
    // later enqueued on the other channel after it as well (switching channels waits for the timeline).
    LIBRECUDA_ERR_PROPAGATE(signalWaitGpu(event->signal, event->target, currentQueueType));
    LIBRECUDA_ERR_PROPAGATE(completeOperation(currentQueueType, false));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::waitForEvent(LibreCUEvent event) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    if (event->signal == nullptr) {
        LIBRECUDA_SUCCEED();
    }
    LIBRECUDA_ERR_PROPAGATE(ctxWaitForValue(event->ctx, &event->signal->value, event->target));
    // see awaitExecution
    return ctxCheckChannelError(event->ctx);
}

libreCudaStatus_t NvCommandQueue::queryEvent(LibreCUEvent event) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    if (event->ctx != nullptr) {
        ctxFlushDoorbells(event->ctx);
    }
    if (event->signal == nullptr || event->signal->value >= event->target) {
        return ctxCheckChannelError(event->ctx);
    }
    return LIBRECUDA_ERROR_NOT_READY;
}

libreCudaStatus_t NvCommandQueue::getEventTimestamp(LibreCUEvent event, uint64_t *pTimestampOut) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(pTimestampOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(event->signal != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(event->signal->value >= event->target, LIBRECUDA_ERROR_NOT_READY); // event was not hit yet
    *pTimestampOut = event->signal->time_stamp;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::buildCommandList(LibreCUcommandList_ *list) {
    std::vector<NvU64> offsets;
    NvU64 size = 0;
    for (const auto &launch: list->launches) {
        LIBRECUDA_VALIDATE(!launch.function->constants.empty() && launch.function->constants[0].const_nr == 0,
                           LIBRECUDA_ERROR_INVALID_IMAGE);
        offsets.push_back(size);
        size += launchMemorySize(launch.function, false);
    }
    // per launch: cache invalidation, the QMD launch and its scheduling, and before the next one a semaphore release
    // with wait for idle into a scratch slot of the list (like the timeline release between dependent launches, it
    // waits for this channel's kernels only; a WAIT_FOR_IDLE method would stall the whole compute engine)
    constexpr NvU32 WORDS_PER_LAUNCH = 12;
    const NvU64 scratch_offset = roundUp(size, 1ull << 8);
    const NvU64 pushbuffer_offset = scratch_offset + (1ull << 8);
    const NvU64 total = pushbuffer_offset + list->launches.size() * WORDS_PER_LAUNCH * sizeof(NvU32);
    NvU64 va{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, total, false, false, true, 0, &va));

    std::vector<NvU32> pushbuffer;
    for (size_t i = 0; i < list->launches.size(); i++) {
        const auto &launch = list->launches[i];
        std::vector<void *> args;
        for (size_t offset: launch.argOffsets) {
            args.push_back(const_cast<NvU8 *>(launch.args.data()) + offset);
        }
        NvU64 qmd_addr{};
        const libreCudaStatus_t status = writeLaunch(launch.function, launch.grid[0], launch.grid[1], launch.grid[2],
                                                     launch.block[0], launch.block[1], launch.block[2],
                                                     launch.sharedMem, args.data(), args.size(), false,
                                                     launch.hasCluster ? launch.cluster : nullptr, va + offsets[i],
                                                     &qmd_addr);
        if (status != LIBRECUDA_SUCCESS) {
            gpuFree(ctx, va);
            return status;
        }
        if (i > 0) {
            // the launches depend on each other
            const NvU64 scratch = va + scratch_offset;
            const NvU32 release[] = {
                static_cast<NvU32>(makeNvMethod(0, NVC56F_SEM_ADDR_LO, 5)),
                U64_LO_32_BITS(scratch), U64_HI_32_BITS(scratch), NvU32(i), 0,
                (NVC56F_SEM_EXECUTE_OPERATION_RELEASE) | (NVC56F_SEM_EXECUTE_RELEASE_WFI_EN << 20) |
                (NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT << 24)
            };
            pushbuffer.insert(pushbuffer.end(), std::begin(release), std::end(release));
        }
        pushbuffer.push_back(makeNvMethod(1, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, 1));
        pushbuffer.push_back((NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE << 12) |
                             (NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE << 4) |
                             (NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE << 0));
        pushbuffer.push_back(makeNvMethod(1, NVC6C0_SEND_PCAS_A, 0x1));
        pushbuffer.push_back(NvU32(qmd_addr >> 8));
        pushbuffer.push_back(makeNvMethod(1, NVC6C0_SEND_SIGNALING_PCAS2_B, 0x1));
        pushbuffer.push_back(0x9); // PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
    }
    memcpy(reinterpret_cast<void *>(va + pushbuffer_offset), pushbuffer.data(), pushbuffer.size() * sizeof(NvU32));
    list->ctx = ctx;
    list->memory_va = va;
    list->pushbuffer_va = va + pushbuffer_offset;
    list->pushbuffer_words = static_cast<NvU32>(pushbuffer.size());
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t NvCommandQueue::launchCommandList(LibreCUcommandList_ *list) {
    QUEUE_LOCK();
    ENSURE_QUEUE_ATTACHED();
    LIBRECUDA_VALIDATE(list != nullptr && !list->launches.empty(), LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(list->ctx == nullptr || list->ctx == ctx, LIBRECUDA_ERROR_INVALID_CONTEXT);
    if (list->ctx == nullptr) {
        LIBRECUDA_ERR_PROPAGATE(buildCommandList(list));
    }
    LIBRECUDA_ERR_PROPAGATE(switchQueueType(COMPUTE));
    LIBRECUDA_ERR_PROPAGATE(ensureTexturePools());
    // the list's kernels start after the stream's preceding kernels completed
    if (timelineNotifyPending) {
        LIBRECUDA_ERR_PROPAGATE(signalNotify(timelineSignal, timelineCtr, COMPUTE));
        timelineNotifyPending = false;
    }
    // the stream's pending commands go first, then the list as an entry of its own
    LIBRECUDA_ERR_PROPAGATE(submitToFifo(COMPUTE));
    {
        auto &gpfifo = fifoOf(COMPUTE);
        std::lock_guard submit_lock(ctx->submit_mutex);
        const NvU32 ring_idx = gpfifo.put_value % gpfifo.entries_count;
        auto [done_signal, done_value] = gpfifo.entry_done[ring_idx];
        if (done_signal != nullptr) {
            LIBRECUDA_ERR_PROPAGATE(ctxWaitForValue(ctx, &done_signal->value, done_value));
        }
        const NvU64 entry = ((list->pushbuffer_va / sizeof(NvU32)) << 2) | (NvU64(list->pushbuffer_words) << 42) |
                            (1l << 41);
        // the slot is free again once the timeline release that follows the list (below) executed
        kickRing(gpfifo, entry, timelineCtr + 1);
    }
    ctx->launch_count.fetch_add(list->launches.size(), std::memory_order_relaxed);
    ctx->last_launch_time.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
    // released with wait for idle semantics: after the list's last kernel
    LIBRECUDA_ERR_PROPAGATE(completeOperation(COMPUTE, false));
    LIBRECUDA_SUCCEED();
}
