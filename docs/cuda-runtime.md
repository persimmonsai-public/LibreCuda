# Running CUDA programs on LibreCUDA

LibreCUDA can run ordinary CUDA programs, written against the CUDA runtime API and compiled with `nvcc`, on the NVIDIA
open kernel modules **without libcuda (the CUDA driver) and without libcudart (the CUDA runtime)**. The GPU is driven
directly through the kernel driver's ioctl interface (RM API + UVM).

No NVIDIA user space binary is reverse engineered. The implementation is based on:

- the open source [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): RM ioctl and control
  structures, class headers (channels, copy engine, QMD layouts), and the UVM driver's use of copy engines,
- the public CUDA toolkit headers (`cuda_runtime_api.h`, `driver_types.h`, `fatbinary_section.h`) for the runtime API
  types and signatures,
- inspection of the code `nvcc`/`ptxas` generate for our own test programs (ELF sections, `.nv.info` attributes,
  constant bank layout, relocations),
- prior open source work: the original LibreCUDA and tinygrad's NV backend.

Tested on an RTX 5060 (Blackwell, sm_120) with driver 610.57.04 (open kernel modules) and CUDA 13.2's `nvcc`.

## Components

```
 CUDA program (compiled by nvcc)
   │  __cudaRegisterFatBinary / __cudaRegisterFunction / __cudaLaunchKernel / cudaMalloc / ...
   ▼
 runtime/  libcudart.so.13 (LibreCUDA runtime)      fat binary parsing, streams, events, legacy default stream
   │  libreCu* driver API
   ▼
 driverapi/  LibreCUDA driver API                   ELF loader, command queues, QMDs, memory, copy engines
   │  ioctl
   ▼
 /dev/nvidiactl, /dev/nvidiaN, /dev/nvidia-uvm      NVIDIA open kernel modules
```

### Runtime (`runtime/`)

`runtime/src/cudart.cpp` implements the CUDA runtime API on top of the LibreCUDA driver API. It is built as
`libcudart.so.13` with the same soname and symbol version as NVIDIA's runtime, so it is a drop-in replacement:

- **Registration.** nvcc's host code registers the embedded fat binary, its kernels and its `__device__`/`__constant__`
  variables from a static constructor. The runtime records them and loads modules lazily per device.
- **Fat binaries.** The runtime picks the SASS (ELF) image for the device's architecture (or an older minor version of
  the same major). If there is none, it compiles the newest compatible PTX with the toolkit's `ptxas` and caches the
  result in `~/.cache/librecuda`. zstd compressed entries (CUDA 13 default) are decompressed with libzstd.
- **Launches.** The `<<<...>>>` syntax goes through `__cudaPushCallConfiguration`, `__cudaGetKernel` and
  `__cudaLaunchKernel`. `cudaLaunchKernel` and `cudaLaunchKernelExC` are supported as well.
- **Streams.** Each stream maps to a LibreCUDA command queue. The legacy default stream's implicit synchronization is
  implemented with GPU side waits on internal events. Work on the default stream waits for all blocking streams and
  vice versa; `cudaStreamNonBlocking` streams are exempt.
- **Memory.** Device memory, pinned host memory, managed memory, copies in all directions, memset, 2D copies, symbol
  copies, pointer attributes and memory info are supported.
- **Managed memory** (`cudaMallocManaged`, `cudaMemPrefetchAsync`, `cudaMemAdvise`) is UVM managed memory, like
  NVIDIA's: the allocation is a shared `mmap` of the UVM file at an address from the shared allocator (offset =
  address, which makes `uvm_mmap` create a managed range), so pages migrate between system memory and the GPUs on CPU
  and GPU (replayable) faults.
  - Prefetching is `UVM_MIGRATE` once the stream's work completed. Advice maps to UVM's read duplication, preferred
    location and accessed-by policies.
  - Copies and fills that involve managed memory use kernels or, with pageable memory on the other side, the CPU
    after the stream's work: copy engine faults aren't replayable, so the copy engine never touches managed memory.
  - `LIBRECUDA_MANAGED_BAR1=1` selects the previous scheme instead: device memory mapped into the CPU address space
    through BAR1, with no migration but uncached CPU reads (the driver doesn't allow cached mappings of video memory).
    It is much slower for CPU access (see Performance) and is kept as a fallback.
- **Several devices.** Each device gets its own context, and all share one address space (unified addressing), so
  pointers tell their device (`cudaPointerGetAttributes`) and memory of any device can be freed from any device.
  - Page-locked host memory is mapped on every device.
  - With `cudaDeviceEnablePeerAccess` (peer support from RM's P2P capabilities; PCIe BAR1 P2P on the test system),
    the peer's existing and future allocations are mapped on the device's GPU, so kernels and copies access them
    directly.
  - Without peer access, copies involving another device's memory (`cudaMemcpy*`, `cudaMemcpyPeer(Async)`) run on
    the owning devices' GPUs through host memory (64 MiB bounce buffer) and complete before returning; fills of another
    device's memory are refused, as in NVIDIA's runtime.
- **Introspection.** Device properties and attributes, function attributes, occupancy, error strings (generated from
  the toolkit's `driver_types.h`) and versions are supported.

### Driver API (`driverapi/`)

- **Device setup.** GPUs are enumerated and probed through RM. GPUs that fell off the bus are skipped.
  `CUDA_VISIBLE_DEVICES` selects GPUs by their index in the driver's list, and only the selected GPUs are opened. Each
  context gets a GPU virtual address space shared with the CPU through UVM, compute channels and copy engine
  channel(s).
- **Cooperative launches.** `cooperative_groups` finds its grid barrier ("grid workspace") through `%envreg1`/`%envreg2`,
  which ptxas reads from the driver area of constant bank 0 (`c[0][0x254]`/`0x258` on sm_120, `0x8c`/`0x90` on
  sm_75–89). Each cooperative launch gets its own zeroed workspace next to its QMD, so concurrent cooperative kernels
  don't share a barrier. The runtime rejects grids larger than the occupancy limit (blocks per SM × SMs) with
  `cudaErrorCooperativeLaunchTooLarge`. Only the sm_120 offsets were tested on hardware.
- **Device side printf.** nvcc compiles `printf` into a call of the external function `vprintf`. The loader resolves it
  to `librecuda_vprintf` (`driverapi/kernels/device_runtime/device_runtime.cu`, embedded per architecture by
  `compile_device_runtime.sh`) and
  raises the calling kernel to that function's register count. The function copies the format string, the arguments
  and `%s` strings into a record in pinned host memory; stream, event and device synchronization (and program exit)
  print the completed records. Details that matter on the GPU:
  - the caller uses `CALL.ABS` with an absolute return address, so the function is compiled with `-rdc=true`, the only
    mode in which ptxas returns with `RET.ABS`;
  - inside the caller, constant bank 4 belongs to the caller, so the function uses no globals or string literals and
    finds its state through `%envreg8`/`%envreg9` in the driver area of constant bank 0;
  - the record counter is in device memory (no PCIe atomics needed); each record's size is written last.
  The buffer is 1 MiB by default (`cudaLimitPrintfFifoSize`); calls that don't fit are dropped with a warning.
- **Device side assert.** `assert` calls the external function `__assertfail`, which the loader resolves to
  `librecuda_assertfail` in the same module. It writes a record with the assertion, file, line, function and thread
  position (printed on stderr in CUDA's format) and stops the kernel with a trap. The SM reports the trap as an
  illegal instruction (warp error 9); the assert record in the printf buffer turns the resulting GR exception into
  `cudaErrorAssert`.
- **Separate compilation (`-rdc=true`).** nvcc's device link step (host code it generates into the program) registers
  the already linked image through the regular `__cudaRegisterFatBinary` path. Calls across translation units stay in
  the image as `R_CUDA_ABS55_16_34` relocations on `CALL.ABS`: bits 2–9 of the target address go to instruction bits
  16–23 and bits 10–56 to bits 34–80 (found by disassembling patched instructions), which the loader applies.
- **Texture objects** (linear and pitched 2D memory). A texture handle is the index of the texture's header in the
  context's texture header pool. The QMD selects "sampler index via header index", so the sampler comes from the
  same index of the sampler pool: every texture object has its own header and sampler. In that mode the whole
  handle is the header index; a handle that also carries a sampler index in bits 20 and up points past the pool
  and reads as an invalid texture (zeros, no error). The pools (`SET_TEX_HEADER_POOL_*`, `SET_TEX_SAMPLER_POOL_*`) are
  created with the first texture object and programmed per queue. Blackwell reads V2 headers only (V1 headers read
  as invalid textures): `TEXHEAD_V2_1DRT` for `tex1Dfetch` on linear memory, `TEXHEAD_V2_PITCH` for `tex2D` on
  pitched memory, laid out as in `clcb97tex.h` (NVIDIA's Blackwell `clcd97tex.h` / `clce97tex.h` in open-gpu-doc are
  identical). Samplers (`TEXSAMP`) carry the address modes, point / linear filtering and the border color; normalized
  coordinates are a header field. Earlier GPUs get V1 `TEXHEAD_1D` headers for linear memory (untested) and no
  pitched textures. Every launch invalidates the texture caches (QMD), so new headers are seen.
- **CUDA arrays** (Blackwell). An array is device memory in the GPU's block linear layout: GOBs of 64 bytes x 8
  rows, blocks one GOB wide and 2^h x 2^d GOBs high / deep (at most 4 KiB, so page aligned memory is block aligned),
  laid out row of blocks by row of blocks, slice by slice; layers of layered arrays follow each other. Only the copy
  engine and the texture unit touch arrays, and both know the layout, including Blackwell's separate GOB formats for
  8 and 16 bit texels (the copy engine is told with `KIND_BPP` of `SET_*_BLOCK_SIZE`, from NVIDIA's `clcab5.h` in
  open-gpu-doc; the texture unit derives it from the format), so the swizzle inside a GOB never matters. Copies
  (`cudaMemcpy2DToArray` / `FromArray`, `cudaMemcpy3D`, `cudaMemcpyToArray`, array to array) are copy engine region
  copies between pitch and block linear surfaces; pageable host memory goes through a temporary device buffer.
  Texture objects on arrays use `TEXHEAD_V2_BL` headers with the array's block sizes. 1D arrays are 2D textures of one
  row (a `ONE_D` block linear header reads as invalid; `tex1D` samples 2D textures); 1D layered arrays use
  `ONE_D_ARRAY`, 2D layered arrays `TWO_D_ARRAY`. Cubemaps are layered arrays of 6 square faces (6 per cube for
  cubemap arrays) with `CUBEMAP` / `CUBEMAP_ARRAY` headers whose depth counts cubes; their surfaces address the faces
  as layers of a `TWO_D_ARRAY`.
- **Mipmapped arrays.** The levels of a layer follow each other; the hardware finds them from level 0's size and block
  size alone, shrinking the block height / depth of each level to fit it (like Mesa's `nil` layout code), and the
  layer stride is the size of all levels aligned to a level 0 block. `cudaGetMipmappedArrayLevel` returns views of the
  levels (for copies and surfaces). Headers carry the level count (`MAX_MIP_LEVEL`, `RES_VIEW_MAX_MIP_LEVEL`);
  samplers the mip filter, the level of detail bias (signed 5.8 fixed point) and its clamps (4.8).
- **Surface objects** (CUDA arrays). Surface loads and stores (`SULD` / `SUST`) take a bindless handle like texture
  fetches: a surface object is a header in the same pool, the texture header of its array. 1D surfaces need a `ONE_D`
  header (the type texture fetches reject). Like CUDA, any array can be used for surfaces (`cudaArraySurfaceLoadStore`
  isn't checked). Out of range accesses in the default trap mode stop the kernel (`cudaErrorIllegalAddress` or
  similar); zero mode ignores writes and reads 0.
- **Device side `malloc` / `free`.** Resolved to `librecuda_malloc` / `librecuda_free` in the same module. The heap
  (`cudaLimitMallocHeapSize`, 8 MiB by default) is allocated in device memory when the first module calling them is
  loaded. Blocks are powers of two (from 32 bytes, 16 byte header, 16 byte aligned data); freed blocks go onto
  lock-free free lists (version-tagged heads against ABA), 32 per size class chosen by the thread's SM, so threads
  rarely contend; otherwise blocks are carved with an atomic bump pointer. When the heap is exhausted, malloc takes
  blocks freed on other SMs, then returns NULL.
- **Dynamic parallelism** (kernels launching kernels, `-rdc=true`). A device side launch compiles into calls of
  `__cudaCDP2GetParameterBufferV2` and `__cudaCDP2LaunchDeviceV2` (public in `cuda_device_runtime_api.h`), which nvcc
  links from NVIDIA's `libcudadevrt`. That code ends in calls into NVIDIA's driver whose interface isn't public, but
  the linked image keeps the relocations of user code's calls to the `__cudaCDP2*` functions, so the loader redirects
  them to LibreCUDA's implementations in the device runtime module (the rest of NVIDIA's devrt code becomes
  unreachable; its driver calls resolve to a trap). The kernel addresses device code passes come from relocations 0x3e /
  0x3f (UMOV immediates) and 0x23 (in data); the loader writes the kernel's code address and the context maps it back.
  - A launch reserves a slot (64 byte header + 4 KiB of parameters) in a queue in pinned host memory
    (`cudaLimitDevRuntimePendingLaunchCount` slots, 2048 by default; launches beyond that fail with
    `cudaErrorLaunchPendingCountExceeded`). The kernel writes its arguments into the slot and marks it launched.
  - LibreCUDA can't launch kernels from the GPU, so the host does: a launch of a kernel from a module that launches
    kernels waits for the kernel (the host launch call is synchronous), then launches the requested kernels on the same
    stream, depth first: each child that may launch kernels is waited for and its own requests launched before the next
    child (tail launches after the other children; nesting up to 24 levels). Children therefore only start after their
    parent completed, and all device side streams behave as one ordered stream.
  - Device side `cudaMemcpyAsync` / `cudaMemsetAsync` and their 2D / 3D forms (pitched memory; no CUDA arrays) are
    queued as slots too and run on the stream in order with the launches. Device side events (create, record,
    `cudaStreamWaitEvent`, destroy) do nothing: the device side streams are one ordered stream anyway.
  - `cudaGetLastError` / `cudaPeekAtLastError` are per thread: an entry per hardware thread slot (SM, warp slot, lane),
    tagged with `%gridid` and the thread's position so that a later thread in the slot doesn't see it.
  - Also provided: `cudaGetDevice`, `cudaGetDeviceCount`, `cudaStreamCreateWithFlags` / `cudaStreamDestroy`,
    `cudaMalloc` / `cudaFree` (the `malloc` heap) and `cudaRuntimeGetVersion`. Other device runtime functions
    (`cudaDeviceGetAttribute`, `cudaGetErrorString`, occupancy, ...) stop the kernel with
    `cudaErrorIllegalInstruction`.
  - Device function pointers (`-rdc`) need nothing special: nvlink turns their relocations into the absolute address
    types the loader applies, and calls are `CALL.ABS` on the loaded address.
  - Kernels in such modules get at least the device runtime's register count (nvlink sized them for NVIDIA's code).
- **Thread block clusters** (Blackwell). The QMD's launch type becomes `GRID_GPC_CGA` with the cluster size in
  `GPC_CGA_WIDTH/HEIGHT/DEPTH`, and the grid dimensions then count clusters. The kernel computes `%cluster_nctaid`,
  `%clusterid`, `%nclusterid`, `%cluster_nctarank` and `%is_explicit_cluster` from the driver area of constant bank 0
  (`c[0][0x2a0]`: cluster dimensions, `0x2b0`: their reciprocals as floats, `0x2c0`: clusters in the grid, `0x2cc`:
  blocks per cluster, `0x36c`: explicit cluster flag), which every launch fills (a regular launch has clusters of one
  block). Compile-time sizes come from `EIATTR_CTA_PER_CLUSTER`; launch-time sizes from
  `cudaLaunchAttributeClusterDimension` / `libreCuLaunchKernelCluster`. Cluster barriers and distributed shared memory
  need nothing else. Launches whose cluster can't be resident on one GPC are rejected.
- **Stream concurrency.** A context has up to 8 compute channels (`LIBRECUDA_COMPUTE_CHANNELS`), each in its own
  subcontext of one channel group, so kernels on different channels run concurrently. A new stream gets the channel
  with the fewest live streams (round robin among equals, so channels of destroyed streams are reused); a channel is
  created when the first stream is assigned to it, since idle channels in the group slow
  every submission down (launch + synchronize 8.0 instead of 7.5 µs with 3 idle channels). Kernels within a stream
  stay ordered, and a stream's completion only waits for its own channel's work.
- **Module loader.** It parses the CUDA ELF image produced by nvcc:
  - lays out code, constant banks, initialized (`.nv.global.init`) and zero-initialized (`.nv.global`) globals in GPU
    memory and applies relocations (with addends),
  - exports `__device__`/`__constant__` symbols,
  - reads kernel parameter offsets and sizes, the parameter base in constant bank 0, register counts, static shared
    memory and stack sizes from the `.nv.info` attributes.
- **Kernel launch.** Each launch writes constant bank 0 (driver data + parameters) and a QMD (version 5 on Blackwell)
  into a kernel argument ring in GPU memory, then sends it through the compute channel. Kernels that call functions
  (recursion, function pointers, virtual functions) get a stack of `cudaLimitStackSize` per thread, like in CUDA.
- **Stream ordering.** Every operation of a stream advances the stream's timeline semaphore.
  - Within a channel, operations execute in order.
  - When a stream switches between the compute channel and a copy channel, the new channel performs a GPU side
    semaphore acquire. No CPU round trip is needed to order compute and copy work.
  - Events are semaphore releases with timestamps and can be waited upon from other streams.
- **Command buffers.** Command buffers and the GPFIFO ring live in video memory. The GPU never fetches commands over
  PCIe. The CPU writes them (and GPPut) through BAR1 and then reads GPPut back before ringing the doorbell, so the writes
  have reached video memory when the GPU looks for new work. Without the read-back the GPU could see a stale GPPut and
  go idle, leaving the submission unexecuted.
- **Memory operations.**
  - Device to device copies and device memsets use vectorized grid-stride kernels; they are about 2.7x and 5x faster
    than the copy engine.
  - Host transfers use the copy engine. Pageable memory is staged through pinned double buffers.
  - Host to device and device to host copies run on two separate async copy engines, so both directions overlap
    (`LIBRECUDA_ASYNC_COPY_ENGINES=0` falls back to the graphics copy engine for both).
- **Memory allocation.** Device allocations of up to 512 KiB come from a per-context pool: power-of-two size classes
  (256 B and up) carve 2 MiB chunks into slots, so small `cudaMalloc`/`cudaFree` calls make no driver calls. One empty
  chunk per size class is kept; others are released. Larger allocations are separate RM allocations.
- **CUDA graphs and stream capture** (record and replay). A graph is a set of nodes (kernels with copies of their
  arguments, copies, fills, host functions, child graphs, event records / waits, memory allocations and frees, empty
  nodes) and dependencies. Instantiation copies the graph, orders the nodes topologically and splits them into lanes
  (chains of nodes each depending on the previous one). `cudaGraphLaunch` replays the nodes through the same paths as
  the calls they stand for, the first lane on the launch stream and the others on internal streams of the executable
  graph (up to 8 lanes, `LIBRECUDA_GRAPH_LANES`), so independent branches run concurrently. Dependencies between lanes
  are events; the lanes start after the launch stream's earlier work and the launch stream waits for all of them. A
  replay costs as much as the individual calls (about 3.9 µs per kernel node on the test system; NVIDIA's runtime
  prebuilds command buffers, about 0.4 µs).
  - Memory allocation nodes (`cudaGraphAddMemAllocNode`, captured `cudaMallocAsync`) allocate their memory when they
    are created and keep it while a graph or executable graph refers to it; free nodes and `cudaFree` don't release
    it. Like CUDA, graphs with memory nodes can't be cloned or used as cloned child graphs, and have at most one
    executable graph at a time.
  - `cudaGraphAddNode` / `cudaGraphNodeGetParams` / `cudaGraphNodeSetParams` / `cudaGraphExecNodeSetParams`, the per
    node type setters of executable graphs, copies to and from symbols, disabled nodes (`cudaGraphNodeSetEnabled`),
    graph / node identifiers, kernel node attributes (cooperative and cluster dimension take effect, others are
    kept) and user objects are supported.
  - Stream capture turns work
  submitted to a capturing stream into nodes after the stream's tail nodes; an event recorded during the capture
  carries the tail, and a stream waiting for it joins the capture (fork / join across streams). Like CUDA,
  synchronizing a capturing stream, synchronous or pageable copies, legacy default stream work that would synchronize
  with a capturing blocking stream (`cudaErrorStreamCaptureImplicit`), unjoined forks and "unsafe" calls
  (`cudaMalloc`, `cudaFree`, `cudaDeviceSynchronize`, ... during a global or this thread's thread local capture,
  unless the thread is relaxed via `cudaThreadExchangeStreamCaptureMode`) fail and invalidate the capture; graph
  launches into a capturing stream are unsupported (as in NVIDIA's runtime). Executable graphs can be updated per node
  or with `cudaGraphExecUpdate` (same topology, no memory nodes). Not supported: conditional nodes (the condition is
  set by device code), external semaphores, edge types other than full dependencies, device side graph launch.
- **Threads.** The driver API is thread safe (a context must not be destroyed while other threads use it). Each
  stream's queue has its own lock. Writes to the shared GPFIFO rings, lazily set up context state (modules, the device
  runtime, local memory, compute channels, event signals, texture pools), allocation bookkeeping, printf output and
  channel error classification each have their own lock, taken in a fixed order (documented in
  `librecuda_internal.h`). The current context is per thread, like in CUDA. A wait for a stream releases the
  stream's lock while it waits, so other threads can keep enqueuing work on it. Waits never hold a context-wide lock.
  The runtime's API lock is a reader/writer lock. Work on streams other than the legacy default stream
  (kernel launches, async copies and fills, event records and waits, stream queries and synchronization) and
  `cudaMalloc` hold it shared plus the stream's own mutex, so threads using different streams submit in parallel.
  Work on the legacy default stream changes the state of every blocking stream, so it holds the lock exclusively, as
  do the remaining calls. Lazy device initialization and module loading have locks of their own. Waits for the GPU
  (stream, event and device synchronization, synchronous and staged copies, launches of kernels that launch kernels)
  and host functions / stream callbacks run without the API lock. While waiting, a thread holds a lifetime lock in
  shared mode, which `cudaStreamDestroy`, `cudaEventDestroy` and `cudaDeviceReset` take exclusively before freeing
  what a waiter may use. A thread waiting for work that another thread's later call enables therefore doesn't
  deadlock   (test `tests/cudart/multithreading`).
- **Context teardown.** Destroying a context waits for its streams, unregisters its channels and VA space from UVM and
  frees its memory and RM objects. Streams that outlive their context become unusable but can still be destroyed.
  `cudaDeviceReset` uses this: it destroys the device's streams, events, CUDA arrays, modules and context, and the next API call
  creates a new context.
- **Robustness.** Channel errors (MMU faults, illegal instructions, ...) are detected through the channel error
  notifier and reported as `cudaErrorIllegalAddress` / `cudaErrorLaunchFailure` instead of hanging. An optional wait
  timeout turns GPU hangs into errors.
  - SM exceptions (Xid 13: misaligned or out of range address, ...) don't stop the kernel: RM kills the faulting warp,
    the kernel completes, and RM marks the channel's error notifier only 10–40 ms later. Every synchronization
    therefore checks the notifier even when the work has completed (cheap: the notifiers are mapped cached), and such
    an error typically surfaces a few synchronizations (~20 ms) after the faulting kernel. The specific error is read
    once from the SM error state RM recorded, through a `GT200_DEBUGGER` object (`sm_exceptions.cpp`), which is then
    cleared (RM keeps it beyond the context). Warp error codes seen on the GPU: 0x5 misaligned PC
    (`cudaErrorInvalidPc`), 0x9 illegal instruction encoding (`cudaErrorIllegalInstruction`; also a trap), 0xe out of
    range address, also a stack overflow (`cudaErrorIllegalAddress`), 0xf misaligned address
    (`cudaErrorMisalignedAddress`), 0x10 invalid address space (`cudaErrorInvalidAddressSpace`); others give
    `cudaErrorLaunchFailure`.
  - At `cudaDeviceReset` and program exit, the runtime waits until 50 ms after the last launch and warns on stderr
    about a kernel failure the program was never told about.
- **Peer access** (`libreCuCtxEnablePeerAccess` / `libreCuCtxDisablePeerAccess`). Every allocation records the other
  contexts it is mapped on (`UVM_MAP_EXTERNAL_ALLOCATION` with their GPU's UUID at the same address). Page-locked host
  memory is mapped on all live contexts, including ones created later; device memory is mapped on the contexts given
  access to its context (existing allocations when access is enabled, new ones when they are made).
  - UVM enables BAR1 and NVLink peers itself when it registers the GPUs; `UVM_ENABLE_PEER_ACCESS` then fails with
    `NV_ERR_INVALID_DEVICE`, which means the pair is already enabled. Pairs LibreCUDA enabled itself (PCIe peers) are
    disabled when one of their contexts is destroyed.
  - `libreCuMemGetContext` and `libreCuMemIsAccessibleFrom` tell the runtime which device owns memory and whether a
    GPU can reach it; `libreCuDeviceGetP2PCapabilities` reads RM's `NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS`.

## Usage

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds the driver API (`build/driverapi/libdriverapi.a`) and, if the CUDA headers and libzstd are found, the
runtime (`build/lib/libcudart.so.13`).

### Run an unmodified CUDA binary

Programs built with `nvcc -cudart=shared` load `libcudart.so.13` at run time. Point the dynamic loader at LibreCUDA's:

```bash
nvcc -O3 -arch=sm_120 -cudart=shared my_program.cu -o my_program
LD_LIBRARY_PATH=build/lib ./my_program
```

`ldd my_program` then shows LibreCUDA's `libcudart.so.13`. Neither libcuda nor NVIDIA's libcudart is loaded.
Alternatively, link directly: `nvcc -cudart=none my_program.cu -Lbuild/lib -lcudart`.

### Environment variables

| Variable | Effect |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Comma separated GPU indices (index in the driver's GPU list). Only these GPUs are opened. |
| `LIBRECUDA_DEBUG=1` | Debug output from the driver API and the runtime. |
| `LIBRECUDA_WAIT_TIMEOUT_MS` | Fail waits for the GPU after this many milliseconds (`cudaErrorLaunchTimeout`) instead of waiting forever. |
| `LIBRECUDA_COMPUTE_CHANNELS` | Number of compute channels streams are spread over (1–16, default 8). `1` runs all kernels one after another. |
| `LIBRECUDA_GRAPH_LANES` | Maximum number of streams a CUDA graph launch runs its independent branches on (1–64, default 8). `1` replays graphs serially on the launch stream. |
| `LIBRECUDA_SMALL_ALLOC_POOL=0` | Disable the pool for small device allocations (every `cudaMalloc` becomes its own RM allocation). |
| `LIBRECUDA_ASYNC_COPY_ENGINES=0` | Use the graphics copy engine for all copies instead of two async copy engines (host→device and device→host). |
| `LIBRECUDA_MANAGED_BAR1=1` | Managed memory as device memory mapped through BAR1 (no page migration, uncached CPU reads) instead of UVM managed memory. |
| `LIBRECUDA_PTXAS` | ptxas binary used to compile PTX-only fat binaries (default: `$CUDA_HOME/bin/ptxas`, `/usr/local/cuda/bin/ptxas`, `PATH`). |

### Tests

```bash
# driver API tests (compile their kernels for the GPU's architecture)
cmake -B build -DLIBRECUDA_TEST_ARCH=sm_120 && cmake --build build -j
# CUDA runtime tests: compiled with nvcc (only when changed) and run on LibreCUDA's runtime
tests/cudart/run_tests.sh build
# the runtime tests 20 times in a row, checking the GPU's PCIe errors (AER) and Xids after each run
tests/cudart/stress.sh build 20 02:00.0
```

`tests/cudart` contains 33 self-checking CUDA programs (`multi_gpu` needs two devices and passes as skipped with one):
- tiled and register-blocked SGEMM; reductions with warp shuffles and atomics,
- `__constant__`/`__device__` globals, function pointers and virtual functions,
- local memory and recursion; dynamic shared memory beyond 48 KiB,
- scan, radix and bitonic sort; n-body,
- streams and events; concurrent kernels on different streams; cooperative launches with `grid.sync()`; thread block clusters with distributed shared memory; device side `printf`, `assert`, `malloc` and `free`; separate compilation (`-rdc`); dynamic parallelism; several host threads; texture objects on linear and pitched 2D memory and CUDA arrays; surface objects; CUDA graphs and stream capture (concurrent branches, memory nodes, generic node parameters, user objects); several devices with and without peer access; managed memory; error reporting for SM exceptions; memory operations; many small allocations; `cudaDeviceReset`; Thrust and CUB; PTX JIT.

## Performance

Measured with `benchmarks/bench.cu` on an RTX 5060 whose PCIe link runs degraded at x4 Gen 4, so host transfers are
link limited to about 7.2 GB/s.

| Benchmark | LibreCUDA |
|---|---|
| Async kernel launch (empty kernel / 5 arguments) | 1.07 / 1.14 µs |
| Launch + `cudaDeviceSynchronize` round trip | 4.42 µs |
| GPU time per empty kernel | 1.13 µs |
| 4 byte `cudaMemcpy` latency | 3.6–3.9 µs |
| H2D / D2H pinned, 1 MiB | 6.99 / 6.88 GB/s |
| H2D / D2H pinned, 256 MiB | 7.23 / 7.15 GB/s |
| H2D / D2H pageable, 256 MiB | 7.17 / 7.02 GB/s |
| H2D ∥ D2H on two streams (graphics copy engine / async copy engines) | 7.19 / 10.94 GB/s |
| Device memory, triad kernel | 393 GB/s |
| `cudaMemcpy` device to device | 387 GB/s (copy engine before: 145 GB/s) |
| `cudaMemset` | 425 GB/s (copy engine before: 81 GB/s) |
| `cudaMalloc` / `cudaFree`, ≤ 512 KiB (pool) | 0.02 / 0.11 µs |
| `cudaMalloc` / `cudaFree`, larger | about 44 / 35 µs |
| SGEMM 4096³ (simple register tiling) | 5.6–7.2 TFLOP/s |

On the same GPU and boot, NVIDIA's CUDA (static cudart and libcuda) failed both benchmark attempts with Xid 13 errors
caused by corrupted command fetches over the faulty PCIe link (method data `0xdeaddead`). NVIDIA keeps command buffers
in system memory, while LibreCUDA keeps them in video memory, so no NVIDIA numbers could be obtained on this system.

Synchronization across devices and host threads (`benchmarks/multi_gpu_sync.cu`, 2000 iterations, two RTX 5060
at 2.5 GT/s x4, typical of 3 runs each; both runtimes have runs up to 2x slower):

| Benchmark | LibreCUDA | NVIDIA |
|---|---|---|
| Launch + `cudaStreamSynchronize`, one device | 8.3 µs | 8.0 µs |
| Launch on both devices + `cudaDeviceSynchronize` both | 12.4 µs | 9.4 µs |
| GPU to GPU ping-pong through `cudaStreamWaitEvent`, per hop | 8.4 µs | 8.1–13.4 µs |
| Launch on device 0 + `cudaEventSynchronize` with device 1 current | 8.6 µs | 8.2 µs |
| `cudaMemcpyPeer` 4 B, peer access / none | 8.9 / 45 µs | 7.1 / 21–55 µs |
| `cudaMemcpyPeerAsync` 4 B + sync, peer access / none | 8.9 / 29 µs | 6.3 / 13.5 µs |
| `cudaMemcpyPeer` 64 MiB, peer access / none | 0.5–0.8 / 0.37 GB/s | 0.91 / 0.89 GB/s |
| Launch + sync round trips, 1 / 2 / 4 threads on one device | 120 / 125 / 205 per ms | 125 / 248 / 448 per ms |
| Launch + sync round trips, 2 / 4 threads over two devices | 226 / 390 per ms | 249 / 480 per ms |

Threads on one device are limited by command submission over the slow link (see Limitations, threading), not by
locks or channels: 1 and 4 compute channels give the same numbers. Copies without peer access go through a host
buffer in two sequential steps, so they reach about half the link's bandwidth.

Managed memory, 64 MiB on one RTX 5060 (`cudaMallocManaged` buffer, simple CPU loops and a kernel adding 1):

| | LibreCUDA (UVM) | NVIDIA | LibreCUDA (`LIBRECUDA_MANAGED_BAR1=1`) |
|---|---|---|---|
| CPU write, first touch / again | 6.2 / 21.9 GB/s | 6.3 / 22.3 GB/s | 0.72 / 0.72 GB/s |
| CPU read | 27 GB/s | 31 GB/s | 0.001 GB/s |
| First kernel after CPU writes (migration) | 102 ms | 156 ms | 7.4 ms |
| Kernel again | 0.38 ms | 0.40 ms | 7.2 ms |
| CPU read after the GPU wrote | 0.79 GB/s | 0.46 GB/s | 0.009 GB/s |

CUDA graphs (`cudaGraphLaunch`, one RTX 5060; empty kernels and kernels of about 20k dependent FMAs in one block):

| Graph | LibreCUDA, lanes | LibreCUDA, `LIBRECUDA_GRAPH_LANES=1` | NVIDIA |
|---|---|---|---|
| Chain of 100 empty kernels | 392 µs | 392 µs | 41 µs |
| 4 branches × 25 empty kernels | 412 µs | 390 µs | 39 µs |
| 4 branches × 25 heavier kernels | 917 µs | 3352 µs | 872 µs |

## Limitations

- **Dynamic parallelism:** children run after their parent completed, launched by the host (see above), so a parent
  can't wait for or communicate with its children, a host launch of a kernel that launches kernels blocks until all of
  them completed, and device launches cost a host round trip. Device side copies of CUDA arrays and a few device
  runtime queries (attributes, error strings, occupancy) aren't supported.
- **Device side `malloc`:** freed blocks are only reused for allocations of the same power-of-two size class and are
  never merged, so a heap fragmented by many small blocks can't serve large allocations.
  Multi-device cooperative launches are not supported either.
- **Textures:** texture objects on linear memory (`tex1Dfetch`), pitched 2D memory and CUDA arrays (1D, 2D, 3D,
  layered, cubemaps, mipmapped; Blackwell only) and surface objects on arrays. Texture / surface references (removed in
  CUDA 12) aren't supported. Linear textures need 256 byte alignment (CUDA requires 512).
- **Thread block clusters** are limited to the portable size of 8 blocks (no non-portable sizes), Blackwell only,
  and can't be combined with cooperative launches.
- **Several devices:** copies between devices without peer access are staged through host memory and synchronous;
  copies of another device's CUDA arrays, and copies between arrays and another device's memory, aren't supported.
  Peer atomics follow RM's capabilities (none over PCIe on the test system).
- **Managed memory:** prefetches are synchronous (they wait for the stream, then migrate);
  `cudaMemRangeGetAttribute(s)`, `cudaMemPrefetchBatchAsync` and stream attachment (`cudaStreamAttachMemAsync`) aren't
  supported; the attach flags of `cudaMallocManaged` make no difference.
- **Host functions:** `cudaLaunchHostFunc` and stream callbacks run synchronously when they are enqueued.
- **Host registration:** `cudaHostRegister` is accepted, but registered memory is still copied through staging buffers.
- **Threading:** threads only submit in parallel on streams other than the legacy default stream. On the test system,
  submissions are limited by PCIe anyway: each launch writes about 1.3 KiB through BAR1 and reads GPPut back over the
  2.5 GT/s x4 link, so 1, 2 and 4 threads all reach about 230 launches/ms.
- **Stream concurrency:** with more streams than compute channels (4 by default), streams share channels, and
  kernels of streams on the same channel run one after another.
- **Clock rates:** the memory clock (`cudaDevAttrMemoryClockRate`) is read from the GPU's highest performance level.
  The core clock (`cudaDevAttrClockRate`) is a fixed 2 GHz estimate: RM doesn't report the GPC clock's maximum through a
  documented control.
- **Faulty PCIe links corrupt data silently.** On the test system's degraded Gen 4 links, `tests/cudart/streams_events`
  failed about 1 in 30 runs with blocks of NaN in data that crossed PCIe, some coinciding with PCIe completion
  timeouts. With the link forced to 2.5 GT/s (no PCIe errors at all), about 190 runs passed, so this is attributed to
  the link, not LibreCUDA. Check `lspci -vv` (UESta/CESta) when host transfers return wrong data.
