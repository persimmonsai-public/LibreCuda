#ifndef LIBRECUDA_CMDQUEUE_H
#define LIBRECUDA_CMDQUEUE_H

#include <initializer_list>
#include <cstddef>
#include <vector>
#include <deque>
#include <cstdint>
#include <mutex>

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
     * Size of the page in words (allocated on the first submission, grown when the GPU is behind as it fills)
     */
    size_t capacity = 0;

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
 * One side of a copy engine region copy (NvCommandQueue::gpuCopyRegion)
 */
struct CopySurface {
    // start of the surface; for pitch layout the start of the region
    NvU64 address;
    bool blockLinear;
    // bytes between rows; for block linear the row size of the surface (a multiple of the 64 byte GOB width)
    NvU64 pitch;
    // pitch layout: bytes between slices
    NvU64 slicePitch;
    // block linear: rows and slices of the surface, log2 of the GOBs per block in y / z, the GOB kind (Blackwell:
    // NVCAB5_SET_*_BLOCK_SIZE_KIND_BPP, by texel size), and the region's origin (x in bytes)
    NvU32 height, depth;
    NvU32 blockHeightLog2, blockDepthLog2;
    NvU32 gobKind;
    NvU32 x, y, z;
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
/**
 * A sequence of kernel launches recorded once and submitted as one prebuilt pushbuffer (libreCuCommandList*). The
 * launches run one after another. It is built for a context on its first launch: one block of video memory holds every
 * launch's constant bank 0 and QMD, followed by the pushbuffer.
 */
struct LibreCUcommandList_ {
    struct Launch {
        LibreCUFunction function;
        NvU32 grid[3], block[3];
        NvU32 sharedMem;
        bool hasCluster;
        NvU32 cluster[3];
        // the arguments, packed one after another (pointers to them in argPtrs)
        std::vector<NvU8> args;
        std::vector<size_t> argOffsets;
    };
    std::vector<Launch> launches{};
    LibreCUcontext ctx{};
    NvU64 memory_va{};
    NvU64 pushbuffer_va{};
    NvU32 pushbuffer_words{};
    // chained launches (Blackwell): the counter the last grid increments, and how many launches of the list were
    // enqueued
    NvU64 completion_va{};
    NvU32 completions{};
};

// (cache line aligned: the queues of different threads would share lines their submissions write)
class alignas(64) NvCommandQueue {
private:
    /**
     * The parent cuda context
     */
    LibreCUcontext ctx;

    /**
     * Held by every operation on the queue. awaitExecution releases it while it waits, so other threads can enqueue
     * work on the stream meanwhile.
     */
    std::recursive_mutex mutex{};

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

    // unique among the process's queues (a destroyed queue's address may be reused)
    const NvU64 queueId = nextQueueId++;
    static inline std::atomic<NvU64> nextQueueId{1};

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
     * Index of the context's copy channels (dma_gpfifos, dma_d2h_gpfifos) this queue's copies run on
     */
    NvU32 copyChannel = 0;

    /**
     * Largest local memory buffer the kernels in the unsubmitted compute commands need. submitToFifo points the channel
     * at it if the channel's buffer is smaller: done at submission, because queues sharing a channel may submit in a
     * different order than they enqueued.
     */
    NvU64 pendingSlmVa = 0;

public:
    /**
     * The value of %current_graph_exec (cudaGetCurrentGraphExec) for kernels launched on the queue: the executable
     * graph being replayed, 0 outside graphs (libreCuStreamSetGraphExec)
     */
    NvU64 graphExec = 0;

    /**
     * Stream priority (set before initializeQueue): the queue gets the last compute channel, which normal queues
     * leave alone, and its kernels a QMD group of their own, so they start as soon as SMs free up instead of after
     * the grids launched before them
     */
    bool highPriority = false;

private:
    NvU64 pendingSlmBytesPerTpc = 0;
    NvU32 pendingSlmPerThread = 0;

    /**
     * Local memory per thread the QMD of the launch being built declares
     */
    NvU32 launchSlmPerThread = 0;

    /**
     * Texture header pool this queue last pointed its compute channel at (the pools never change once created)
     */
    NvU64 programmedTexturePools = 0;

    /**
     * The context's code_generation when this queue's launches last invalidated the shader caches (one more than any:
     * the first launch does)
     */
    NvU32 invalidatedCodeGeneration = ~0u;

    /**
     * GPU virtual address pointing to a page of memory used for storing the arguments to launched kernels
     */
    NvU8 *kernArgsPageVa{};

    /**
     * Byte offset into kernArgsPageVa to bump allocate memory inside the page.
     */
    size_t kernArgsWriteIdx{};

    /**
     * Size of the kernel argument page in bytes (allocated on the first launch, grown when the GPU is behind as it
     * fills)
     */
    size_t kernArgsCapacity{};

    /**
     * Command and kernel argument pages replaced by larger ones while the GPU may still read them: freed once the
     * timeline reached the value
     */
    std::vector<std::pair<NvU64, NvU32>> retiredPages{};

    /**
     * Frees the retired pages the GPU is done with (all: the queue is idle)
     */
    void freeRetiredPages(bool all);

    /**
     * Pinned host buffers used to stage copies from/to pageable host memory, and their sizes
     */
    NvU8 *stagingBuffers[2]{};
    size_t stagingSizes[2]{};

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

    /**
     * @return whether all submitted work has completed; false if another thread is using the queue (doesn't wait for
     * its lock and doesn't check for channel errors)
     */
    bool tryIsIdle();

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

    /**
     * Bytes of launch memory (constant bank 0, QMD, grid workspace) a launch of the function needs
     */
    NvU64 launchMemorySize(LibreCUFunction function, bool cooperative) const;

    /**
     * Validates a launch and writes its constant bank 0 and QMD to the launch memory at kernargs_pointer
     */
    libreCudaStatus_t
    writeLaunch(LibreCUFunction function,
                uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                uint32_t sharedMemBytes, void **params, size_t numParams, bool cooperative, const NvU32 *clusterDims,
                NvU64 kernargs_pointer, NvU64 *pQmdAddr);

    /**
     * Enqueues a command list (building it first if needed): its pushbuffer becomes a GPFIFO entry of its own after
     * the queue's pending commands, followed by the timeline release
     */
    libreCudaStatus_t launchCommandList(LibreCUcommandList_ *list);

    /**
     * Makes the queue's following work wait on the GPU until the 64 bit value at address (GPU accessible, 16 byte
     * aligned) is at least value
     */
    libreCudaStatus_t waitValueGpu(const volatile NvU64 *address, NvU64 value);

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
     * Copies a 3D region with the copy engine between surfaces in pitch or block linear layout (CUDA arrays). Both
     * surfaces must be GPU accessible. widthBytes x height x depth bytes / rows / slices.
     */
    libreCudaStatus_t gpuCopyRegion(const CopySurface &dst, const CopySurface &src, NvU32 widthBytes, NvU32 height,
                                    NvU32 depth, bool async);

    /**
     * Fills memory using the copy engine. elementSize must be 1, 2 or 4.
     */
    libreCudaStatus_t gpuMemset(void *dst, NvU32 value, size_t elementSize, size_t numElements);

    libreCudaStatus_t recordEvent(LibreCUEvent event, LibreCUstream_ *pStream);

    /**
     * Makes subsequent work of this queue wait on the GPU until the event has been reached
     */
    libreCudaStatus_t waitEventGpu(LibreCUEvent event);

    /**
     * Stream memory operations: waits until the 32 / 64 bit value at a GPU address compares to value (operation: an
     * NVC56F_SEM_EXECUTE_OPERATION_ACQ*), or writes value there once the stream's preceding work completed
     */
    libreCudaStatus_t memOpWait(NvU64 address, NvU64 value, bool is64, NvU32 operation);

    libreCudaStatus_t memOpWrite(NvU64 address, NvU64 value, bool is64);

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

    /**
     * Whether all work enqueued so far was submitted and has completed (queue lock held)
     */
    bool idle() const;

    /**
     * Writes a GPFIFO entry (whose slot is reusable once the timeline reaches doneValue) and rings the doorbell.
     * Called with the channel's submit lock held (release: given up before the doorbell) and the slot free.
     */
    void kickRing(GPFifo &gpfifo, NvU64 entry, NvU32 doneValue, ChannelLock *release = nullptr);

    libreCudaStatus_t buildCommandList(LibreCUcommandList_ *list);

    libreCudaStatus_t allocKernArgs(NvU64 *pMemOut, size_t size);

    libreCudaStatus_t ensureStagingBuffers(size_t numBytes);

    GPFifo &fifoOf(QueueType type);

    /**
     * @return the copy engine channel to use for a copy between the given (GPU accessible) pointers
     */
    QueueType copyQueueFor(const void *dst, const void *src) const;
};

#endif //LIBRECUDA_CMDQUEUE_H
