#ifndef LIBRECUDA_CMDQUEUE_H
#define LIBRECUDA_CMDQUEUE_H

#include <initializer_list>
#include <cstddef>
#include <vector>
#include <deque>
#include <cstdint>

#include "librecuda.h"
#include "librecuda_internal.h"

#include "nvidia/nvtypes.h"

typedef NvU64 NvMethod;

static inline NvMethod makeNvMethod(int subcommand, int method, int size, int typ = 2);

struct NvSignal {
public:
    volatile NvU64 value{};
    volatile NvU64 time_stamp{};
};

enum QueueType {
    COMPUTE,
    // copy engine channel for host to device / device to device copies and memsets
    DMA,
    // copy engine channel for device to host copies (the DMA channel if the GPU has no second async copy engine)
    DMA_D2H
};

constexpr int NUM_QUEUE_TYPES = 3;

struct CommandQueuePage {

    /**
     * CPU mapped GPU memory for the command queue.
     * Command queue is copied to this page for submission.
     */
    NvU32 *commandQueueSpace{};

    /**
     * Pointer into commandQueuePage. Commands are mem-copied in appending fashion into the command queue page and
     * commandWritePtr is used to keep track of the current offset from the base pointer.
     */
    uint64_t commandWriteIdx = 0;

    /**
     * Timeline value the GPU has to reach before all previously submitted contents of this page have been consumed.
     */
    NvU32 lastSubmittedTimelineValue = 0;
};

/**
 * Stream ordered command queue.
 *
 * Every operation increments the stream's timeline counter. Operations are executed either on the compute channel
 * (kernel launches) or the copy engine channel (DMA). Within a channel, work executes in order. When the stream
 * switches from one channel to the other, the pending commands of the old channel are submitted together with a
 * release of the timeline semaphore and the new channel's commands start with a GPU side acquire of that value,
 * so ordering between the engines is enforced without the CPU waiting.
 */
class NvCommandQueue {
private:
    /**
     * The parent cuda context
     */
    LibreCUcontext ctx;

    /**
     * State whether the queue has been initialized.
     * @code initializeQueue() @endcode must be called before queue methods can be called
     */
    bool initialized = false;

    /**
     * Commands built up for the compute and the copy channel respectively. Only one of them is non-empty at a time.
     *
     * Format:
     * - NvMethod (NvU32) nvMethod
     * - vararg NvU32 expected by method (tightly packed arguments) ; size is encoded in the NvMethod bit structure
     */
    std::vector<NvU32> commandBuffers[NUM_QUEUE_TYPES]{};

    /**
     * command queue page & stack ptr per channel
     */
    CommandQueuePage queuePages[NUM_QUEUE_TYPES]{};

    /**
     * CPU mapped GPU buffer of NvSignals, which can be claimed and freed again
     */
    NvSignal *signalPool{};

    /**
     * Vector of free signal handles (index into signalPool)
     */
    std::deque<NvU32> freeSignals{};

    /**
     * Primary signal used for synchronization
     */
    NvSignal *timelineSignal{};

    /**
     * Incrementing counter used for synchronization.
     * The timelineCtr advances first, as the queue is built, and the signal's value advances
     * to meet the timelineCtr. If they are equal, there is no async operation pending.
     */
    NvU32 timelineCtr = 0;

    /**
     * State whether operations were enqueued without being followed by a release of the timeline semaphore.
     * Async kernel launches defer the release so that they can run concurrently.
     */
    bool timelineNotifyPending = false;

    /**
     * The channel the most recent operation was enqueued on.
     */
    QueueType currentQueueType = COMPUTE;

    /**
     * Index of the context's compute channel this queue's kernels run on (assigned round-robin)
     */
    NvU32 computeChannel = 0;

    /**
     * Local memory buffer this queue last pointed its compute channel at. Tracked per queue (not per channel): another
     * queue's methods for the same channel may still be unsubmitted when this queue launches.
     */
    NvU64 programmedSlmVa = 0;

    /**
     * Texture header pool this queue last pointed its compute channel at (see programmedSlmVa)
     */
    NvU64 programmedTexturePools = 0;

    /**
     * GPU virtual address pointing to a page of memory used for storing the arguments to launched kernels
     */
    NvU8 *kernArgsPageVa{};

    /**
     * Byte offset into kernArgsPageVa to bump allocate memory inside the page.
     */
    size_t kernArgsWriteIdx{};

    /**
     * Pinned host buffers used to stage copies from/to pageable host memory
     */
    NvU8 *stagingBuffers[2]{};

private:
    NvU64 local_mem_window = 0xff000000;
    NvU64 shared_mem_window = 0xfe000000;

public:
    explicit NvCommandQueue(LibreCUcontext ctx);

    /**
     * Called when the parent context is destroyed: the queue's memory and channels are gone, so the queue becomes
     * unusable and its destructor must not touch the context anymore.
     */
    void detachFromContext();

    /**
     * Must be called before the queue is usable
     * @return status
     */
    libreCudaStatus_t initializeQueue();

    /**
     * Submits all enqueued commands to the GPU
     */
    libreCudaStatus_t startExecution();

    /**
     * Waits for all submitted work to complete
     */
    libreCudaStatus_t awaitExecution();

    /**
     * @return LIBRECUDA_SUCCESS if all submitted work has completed, LIBRECUDA_ERROR_NOT_READY otherwise
     */
    libreCudaStatus_t queryExecution();

    ~NvCommandQueue();

    libreCudaStatus_t ensureEnoughLocalMem(LibreCUFunction function);

    /**
     * Points the queue's compute channel at the context's texture pools, if they exist and it isn't yet
     */
    libreCudaStatus_t ensureTexturePools();

    libreCudaStatus_t
    launchFunction(LibreCUFunction function,
                   uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                   uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                   uint32_t sharedMemBytes,
                   void **params, size_t numParams, bool async, bool cooperative = false,
                   const NvU32 *clusterDims = nullptr);

    libreCudaStatus_t
    buildQmdV05(LibreCUFunction function,
                uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                uint32_t shmem_usage, NvU64 kernargs_pointer, uint32_t *qmd_data, const NvU32 *clusterDims);

    /**
     * Copies using the copy engine. Both pointers must be GPU accessible.
     */
    libreCudaStatus_t gpuMemcpy(void *dst, const void *src, size_t numBytes, bool async);

    /**
     * Copies between GPU accessible memory and pageable host memory through pinned staging buffers.
     * Returns once the copy has completed.
     */
    libreCudaStatus_t stagedMemcpy(void *dst, const void *src, size_t numBytes, bool dstIsHost);

    /**
     * Fills memory using the copy engine. elementSize must be 1, 2 or 4.
     */
    libreCudaStatus_t gpuMemset(void *dst, NvU32 value, size_t elementSize, size_t numElements);

    libreCudaStatus_t recordEvent(LibreCUEvent event, LibreCUstream_ *pStream);

    /**
     * Makes subsequent work of this queue wait on the GPU until the event has been reached
     */
    libreCudaStatus_t waitEventGpu(LibreCUEvent event);

    static libreCudaStatus_t waitForEvent(LibreCUEvent event);

    static libreCudaStatus_t queryEvent(LibreCUEvent event);

    static libreCudaStatus_t getEventTimestamp(LibreCUEvent event, uint64_t *pTimestampOut);

private:

    libreCudaStatus_t enqueue(NvMethod method, std::initializer_list<NvU32> arguments, QueueType type);

    libreCudaStatus_t obtainSignal(NvSignal **pSignalPtr);

    libreCudaStatus_t releaseSignal(NvSignal *signal);

    libreCudaStatus_t signalNotify(NvSignal *pSignal, NvU32 signalTarget, QueueType type);

    libreCudaStatus_t signalWaitCpu(NvSignal *pSignal, NvU64 signalTarget);

    /**
     * waits on the GPU until the signal reaches the specified target
     * @param pSignal the signal
     * @param signalTarget the target value
     * @param type the channel which should wait
     * @return status
     */
    libreCudaStatus_t signalWaitGpu(NvSignal *pSignal, NvU64 signalTarget, QueueType type);

    /**
     * Prepares the queue to enqueue an operation on the given channel. If the previous operation was enqueued on the
     * other channel, the commands for that channel are submitted and a GPU side wait is enqueued for the new one.
     */
    libreCudaStatus_t switchQueueType(QueueType type);

    /**
     * Ends the current operation: increments the timeline and, unless deferred, releases the timeline semaphore.
     */
    libreCudaStatus_t completeOperation(QueueType type, bool deferNotify);

    libreCudaStatus_t submitToFifo(QueueType type);

    libreCudaStatus_t allocKernArgs(NvU64 *pMemOut, size_t size);

    libreCudaStatus_t ensureStagingBuffers();

    GPFifo &fifoOf(QueueType type);

    /**
     * @return the copy engine channel to use for a copy between the given (GPU accessible) pointers
     */
    QueueType copyQueueFor(const void *dst, const void *src) const;
};

#endif //LIBRECUDA_CMDQUEUE_H
