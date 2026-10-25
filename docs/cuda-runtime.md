# Running CUDA programs on LibreCUDA

LibreCUDA can run ordinary CUDA programs, written against the CUDA runtime API or the CUDA driver API and compiled with
`nvcc`, on the NVIDIA open kernel modules **without NVIDIA's libcuda (the CUDA driver) and without NVIDIA's libcudart
(the CUDA runtime)**. LibreCUDA also provides cuBLAS and cuBLASLt, so PyTorch's eager operations, matrix products and
training run on it too, except for operations that need NVIDIA's other math libraries (see [PyTorch](#pytorch)). The GPU is driven
directly through the kernel driver's ioctl interface (RM API + UVM).

No NVIDIA user space binary is reverse engineered. The implementation is based on:

- the open source [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): RM ioctl and control
  structures, class headers (channels, copy engine, QMD layouts), and the UVM driver's use of copy engines,
- the public CUDA toolkit headers (`cuda_runtime_api.h`, `driver_types.h`, `fatbinary_section.h`, `cuda.h`) for the
  runtime and driver API types and signatures,
- inspection of the code `nvcc`/`ptxas` generate for our own test programs (ELF sections, `.nv.info` attributes,
  constant bank layout, relocations),
- prior open source work: the original LibreCUDA and tinygrad's NV backend,
- NVIDIA's open source (BSD) CUTLASS templates for the GEMM kernels of LibreCUDA's cuBLAS.

Tested on an RTX 5060 (Blackwell, sm_120) with driver 610.57.04 (open kernel modules) and CUDA 13.2's `nvcc`.

## Components

```
 CUDA program (compiled by nvcc), PyTorch, ...
   │  __cudaRegisterFatBinary / __cudaLaunchKernel / cudaMalloc / ...      cuModuleLoadData / cuLaunchKernel / ...
   ▼                                                                      ▼
 blas/  libcublas.so.13, libcublasLt.so.13          GEMM engine: CUTLASS and generic kernels (nvcc, on the runtime)
   │
 runtime/  libcudart.so.13 (LibreCUDA runtime)  ◄── libcuda.so.1 (filter: CUDA driver API of the same library)
   │  libreCu* driver API                       fat binary parsing, streams, events, legacy default stream
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
    location and accessed-by policies. UVM has no query for them, so the runtime records advice and prefetches per
    page range and answers `cudaMemRangeGetAttribute(s)` from that, like CUDA: a value only if the whole range has
    it. Batch prefetches and discards (`cudaMemPrefetchBatchAsync`, `cudaMemDiscard*BatchAsync`) are supported.
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
    directly. Copies of 256 KiB or more between two GPUs' memories run on the faster of the two GPUs' copy engines:
    the first such copy between two GPUs times an 8 MiB copy on each (engines differ a lot with the PCIe topology),
    and when it isn't the stream's GPU, the copy runs on a private stream of the other GPU, ordered with the stream's
    work through events on the GPUs.
  - Without peer access, copies between two devices' memories (`cudaMemcpy*`, `cudaMemcpyPeer(Async)`) go through
    page-locked host memory in 2 MiB chunks (64 KiB buffers for small copies), three slots: the source GPU copies chunk
    i + 1 to the host while the destination GPU copies chunk i from it, chained with events on the GPUs and ordered
    with the stream's work (asynchronous copies stay asynchronous). Copies between host memory and another device's
    memory run on the owning device and complete before returning; fills of another device's memory are refused, as
    in NVIDIA's runtime.
- **Host functions** (`cudaLaunchHostFunc`, `cudaStreamAddCallback`, graph host nodes) run asynchronously on a
  thread of the runtime: the stream records an event and then waits on the GPU for a ticket in page-locked memory
  (`libreCuStreamWaitValue64`, a semaphore acquire); the thread waits for the event, calls the function and writes the
  ticket, so the function runs after the stream's preceding work and the following work after the function. Functions
  run one at a time, in the order they were enqueued.
- **Introspection.** Device properties and attributes, function attributes, occupancy, error strings (generated from
  the toolkit's `driver_types.h`) and versions are supported.
- **Memory pools.** Each device has its default pool only (`cudaDeviceGetDefaultMemPool`, `cudaDeviceGetMemPool`);
  `cudaMallocAsync` allocations are ordinary allocations, which the pool's attributes count (used / reserved memory and
  their high watermarks). The release threshold and reuse policies can be set and read; trimming has nothing to
  release. `cudaMemPoolSetAccess` enables peer access for other devices.
- **Profiler** start / stop have no effect (no profiler can attach).
- **Interprocess communication** (`cudaIpc*`, `cuIpc*`): the handle of device memory names the RM memory object
  containing the address (for small allocations, their chunk) and the address's offset in it; exporting gives the
  object a share policy that lets clients of the same OS user duplicate it (`NV04_SHARE`,
  `RS_SHARE_TYPE_OS_SECURITY_TOKEN`), and opening duplicates it into the process's RM client (`NV04_DUP_OBJECT`) and
  maps it (opening the same memory again returns the same mapping, which stays until as many closes). Events created
  with `cudaEventInterprocess` keep their semaphore and the target of their latest record in a shared page of system
  memory, which the other process maps and reads when it waits. Handles open while the exporting process lives, on
  the GPU they were made on or on another GPU with peer access to it: the opening process gets a context on the
  owning GPU, duplicates the object there and maps it for its current GPU as a peer's memory
  (`tests/cudart/ipc.cu`, `tests/cudart/ipc_peer.cu`).

### cuBLAS and cuBLASLt (`blas/`)

`libcublas.so.13` and `libcublasLt.so.13` implement the part of cuBLAS and cuBLASLt that PyTorch and similar libraries
use, with NVIDIA's sonames and symbol versions. They are CUDA programs themselves (built with nvcc against the LibreCUDA
runtime); only the public `cublas_api.h` / `cublasLt.h` define the types.

- **GEMM engine** (`blas/src/gemm.cu`, in `libcublasLt.so.13`): `D = alpha op(A) op(B) + beta C` for column-major
  operands, strided or pointer array batches, host or device scalars.
  - A matrix times 1 to 4 vectors (decoding: a linear layer's `x W^T` with a token or a few) with half or bfloat16
    operands, A transposed and both 16 byte aligned: a GEMV kernel, each warp summing two outputs per vector from 16
    byte loads (401 GB/s against the tensor core kernels' 331 GB/s for a Qwen3-4B decode step's GEMVs on an RTX 5060).
  - Half and bfloat16 operands (half / bfloat16 or float output) use CUTLASS tensor core kernels (the sm_80 `mma.sync`
    kernels, which run on newer GPUs), with 16 byte accesses when sizes and leading dimensions allow and 4 byte accesses
    otherwise. Tiles of 64 x 64 (6 stages) when k is at least 256, unless 128 x 128 tiles would fill the GPU 4 times or
    more and the k x n operand outgrows L2 (the CTAs running at once walk down the rows of a few tile columns, so each
    column of small tiles reads that operand again, from memory once it no longer fits); otherwise 128 x 128 (4 stages).
    The small tiles keep more SMs busy: llama.cpp's 512 token prompt GEMMs (Qwen3-4B bf16) run within 1-2% of cuBLAS,
    up to 15% faster than with the large tiles, which they used to get. Operands of odd sizes or leading dimensions are
    copied into aligned, zero padded scratch memory, multiplied there and the result copied back, so they also run on
    tensor cores.
  - Float operands use TF32 tensor cores when the math mode or compute type allows it, otherwise exact CUTLASS SIMT
    kernels; double operands CUTLASS SIMT kernels.
  - 8 bit float operands (e4m3 / e5m2, cuBLASLt only) use CUTLASS's sm_89 FP8 tensor core kernels (no TMA, so they
    run on sm_120 too; float accumulation) with bf16, half, float, e4m3 or e5m2 output. Built for sm_120a, they use
    sm_120's block scaled `mma.sync` (`.kind::mxf8f6f4`) with scale factors 1 instead of sm_89's e4m3 / e5m2 one, which
    like the plain `.kind::f8f6f4` runs at half its rate on an RTX 5060 (85 against 170 TFLOPS in a loop of
    instructions). Their epilogue applies cuBLASLt's per-tensor scales
    (`D = scaleD (alpha scaleA scaleB AB + beta scaleC C)`), adds a bias of D's type (as C with leading dimension 0) and
    computes D's absolute maximum before scaling (one atomic per warp). `mma.sync` takes 8 bit operands only as A transposed and B not (cuBLASLt's "TN" layout): operands in
    another layout or misaligned are transposed / copied into scratch memory first.
  - Everything else (complex numbers, conjugation, strided vectors of GEMV, small unaligned problems, device scalars of
    another type) runs on a generic shared memory kernel.
- **cuBLAS**: handles, streams, workspaces, pointer and math modes; `{S,D,C,Z,H}gemm`, `SgemmEx`, `GemmEx`, their
  strided batched and batched (pointer array) forms; vector and matrix copies between host and device;
  (`blas/src/blas_routines.cu`) the level 1 routines (`axpy`, `scal`, `copy`, `swap`, `dot*`, `nrm2`, `asum`,
  `i*amax` / `i*amin`, `rot`, `rotg`, `rotm`), level 2 (`gemv`, `ger*`, `symv` / `hemv`, `syr*` / `her*`, `trmv`,
  `trsv`) and level 3 (`trsm`, `trmm`, `symm` / `hemm`, `syrk` / `herk`, `syr2k` / `her2k`), and the batched
  factorizations `getrfBatched`, `getrsBatched`, `geqrfBatched`, `gelsBatched`, `trsmBatched`. Triangular solves are
  recursive (GEMMs between halves, one thread block for small triangles); the products of triangular and symmetric
  matrices are GEMMs on expanded copies in scratch memory. Other functions of `cublas_api.h` exist and return
  `CUBLAS_STATUS_NOT_SUPPORTED`.
- **cuBLASLt**: matmul descriptors, matrix layouts (column and row major, strided batches) and preferences, stored in
  the header's opaque storage so the `*Init` functions work; `cublasLtMatmulAlgoGetHeuristic` returns one algorithm,
  which needs no workspace; `cublasLtMatmul` with the epilogues bias, ReLU and GELU (tanh approximation, like cuBLASLt)
  and their combinations: in the CUTLASS kernels' epilogue for 16 bit operands and output with 16 byte accesses (the
  bias read as C with leading dimension 0 when beta is 0), else in a pass over the result. FP8 matmuls as
  `torch._scaled_mm` makes them: e4m3 / e5m2 A and B, per-tensor scales of A, B, C and D
  (`CUBLASLT_MATMUL_DESC_*_SCALE_POINTER`, scale mode `SCALAR_32F`), D's absolute maximum
  (`CUBLASLT_MATMUL_DESC_AMAX_D_POINTER`, 8 bit D as in NVIDIA's cuBLASLt), a C that isn't read (beta 0) of another type
  than D, and the bias epilogue. cuBLAS handles can be passed as cuBLASLt handles (PyTorch does).

### cuSOLVER and cuSPARSE (`blas/`)

`libcusolver.so.12` implements the dense functions of `cusolverDn.h` that PyTorch uses (`blas/src/cusolver.cu`), for
float, double and complex types, with the typed and the 64 bit generic (`X...`) interfaces:

- **LU** (`getrf`, `getrs`), **Cholesky** (`potrf`, `potrs`, batched) and **QR** (`geqrf`, `orgqr` / `ungqr`,
  `ormqr` / `unmqr`): one thread block per matrix for small problems (`blas/src/lapack_device.cuh`, LAPACK's
  conventions for pivots, reflectors and `info`); large ones are blocked: panels factored by one block, block reflectors
  (`larft`) applied with GEMMs, trailing updates with GEMMs and triangular solves.
- **Hermitian eigen decomposition** (`syevd` / `heevd`, `syevj` / `heevj` and their batched forms, `Xsyevd`,
  `XsyevBatched`): two-sided Jacobi rotations in rounds of disjoint pairs, one thread block per rotation; the host
  checks for convergence after each sweep. Eigenvalues are ascending, as in LAPACK.
- **Singular value decomposition** (`gesvd`, `gesvdj`, `gesvdjBatched`, `gesvdaStridedBatched`): one-sided Jacobi on
  the matrix or its conjugate transpose (whichever is taller); singular vectors of zero singular values and the extra
  columns of full `U` / `V` are completed to orthonormal bases.
- **Non-symmetric eigen decomposition** (`Xgeev`, right eigenvectors): on the host like cuSOLVER's hybrid algorithm,
  in double precision: Householder reduction to Hessenberg form, shifted complex QR iterations and back substitution.
  Real matrices get LAPACK's output format (real parts and imaginary parts, eigenvectors of conjugate pairs packed into
  two real columns).

Not supported: the LDL^T factorization (`sytrf`, `Xsytrs`, PyTorch's `linalg.ldl_factor`); the remaining functions of
`cusolverDn.h` return `CUSOLVER_STATUS_NOT_SUPPORTED`.

`libcusparse.so.12` (`blas/src/cusparse.cu`) implements cuSPARSE's generic API and the legacy functions PyTorch uses,
for half, bfloat16 (computed in float), float, double and complex values with 32 or 64 bit indices and base 0 or 1:

- **Descriptors:** sparse matrices in CSR, CSC and COO format with strided batches and fill mode / diagonal type
  attributes, dense matrices (row or column major, strided batches), dense and sparse vectors.
- **SpMV, SpMM** (all operations on A and B), **SDDMM**: on a CSR view of op(A). CSR matrices are used as they are,
  CSC matrices as the CSR of their transpose; other cases are converted in the caller's buffer by a stable radix sort
  of the entries by their row in op(A). A warp computes a row (SpMV) or a row's 32 output columns (SpMM).
- **SpGEMM** (workEstimation / compute / copy): all products are expanded, sorted by (row, column) and equal positions
  added up in order, so the results are deterministic.
- **SpSV, SpSM** (lower / upper, unit / non-unit, transposed): sync-free substitution, a warp per row in dependency
  order (rows are handed out by a counter; a row waits for flags of the rows it needs).
- **Conversions:** sparse to dense, dense to sparse, `Csr2cscEx2`; `Axpby`, `Gather`, `Scatter`, `SpVV`.
- **Legacy:** `bsrmv`, `bsrmm`, `bsrsv2`, `bsrsm2` (BSR solves expand the blocks to CSR), `csrgeam2`, `Xcoosort`,
  `Xcsrsort`, `Xcoo2csr`, identity permutations.

The other functions of `cusparse.h` (incomplete factorizations, tridiagonal solvers, blocked / sliced ELL, SpMVOp and
SpMMOp) return `CUSPARSE_STATUS_NOT_SUPPORTED`.

### cuDNN (`blas/`)

`libcudnn.so.9` implements the parts of cuDNN 9 that PyTorch uses, with both of its interfaces (only NVIDIA's public
`cudnn*.h` headers, from the `nvidia-cudnn` pip package or `CUDNN_INCLUDE_DIR`, are used for types):

- **Convolutions** (`blas/src/cudnn_conv.cu`, `blas/src/cudnn_implicit.cu`): forward, backward data and backward
  filter, 1 to 3 spatial dimensions, strides, dilations, asymmetric padding, groups, any tensor strides (channels last
  too), half / bfloat16 / float / double data. Three implementations, the first that applies is used:
  - 1x1 convolutions with stride 1 and no padding of NCHW- or NHWC-like tensors are GEMMs per image on the tensors
    themselves (batched GEMMs; the filter gradient sums per-image products).
  - Half and bfloat16 1D / 2D convolutions without groups, and float ones when TF32 is allowed, are CUTLASS implicit
    GEMMs (tensor cores, float accumulation): the operands are copied into channels-last layouts with the channels
    padded to multiples of 8 (16 bit data; 4 for the forward pass of inputs with at most 4 channels, which has its own
    kernel) or 4 (float data), the kernels write the result (scaled by alpha; float when beta is used) and it is copied
    into the caller's layout. 128 x 64 tiles for at most 64 output channels; the filter gradient splits the batch
    serially over several thread blocks.
  - Everything else is im2col and GEMMs on the GEMM engine: the input windows of a chunk of images are gathered and
    multiplied by the packed filters; the data gradient multiplies the output gradient by the transposed filters and
    gathers the windows back (col2im, no atomics); the filter gradient folds the batch into the GEMM's inner
    dimension. `LIBRECUDA_CUDNN_IM2COL=1` uses it for all convolutions.

  Float data is multiplied as TF32 when the caller allows it: the legacy API unless the convolution's math type is
  `CUDNN_FMA_MATH`, the backend API with engine 1, which heuristics offer first for float graphs and which carries the
  `CUDNN_NUMERICAL_NOTE_TENSOR_CORE` note (PyTorch filters such engines out unless `torch.backends.cudnn.allow_tf32`;
  engine 0 is exact). `NVIDIA_TF32_OVERRIDE=0` disables TF32.
- **The backend (graph) API** that cudnn-frontend, and PyTorch's convolutions, use (`blas/src/cudnn_backend.cpp`):
  descriptors are attribute stores (descriptor attributes are deep copied when set, as callers destroy theirs); engine
  heuristics offer the engines (no knobs) of graphs of one convolution; execution plans report the convolution's
  workspace. `LIBRECUDA_CUDNN_DEBUG=1` traces the calls.
- **The legacy API** (`blas/src/cudnn_legacy.cpp`): tensor, filter, convolution, activation, pooling, dropout and
  spatial transformer descriptors; convolutions (one algorithm per direction, reported by the `Get*` / `Find*`
  functions), convolution with bias and activation; batch normalization (training with running statistics, inference,
  backward; spatial, persistent and per-activation modes; fused addition and activation); pooling forward; the spatial
  transformer (affine grid and bilinear sampler, forward and backward) (`blas/src/cudnn_ops.cu`). Spatial batch
  normalization of tensors whose positions have one stride (NCHW, NHWC) reduces per channel and group of images in
  parallel (Welford statistics in float, merged in double; double data in double), then normalizes elementwise.
- **Recurrent networks** (`blas/src/cudnn_rnn.cu`): RNN (ReLU, tanh), LSTM and GRU with several layers, both
  directions, sequences of different lengths (padded and packed layouts), initial / final states, dropout between
  layers, inference and training (backward data and weights). The input projections of a layer are one GEMM, each
  time step a GEMM and a cell kernel; training keeps layer outputs, gates and gate gradients in the reserve space. The
  weight layout is the one PyTorch's weight flattening expects. Projections (`proj_size`) are not supported.
- **CTC loss**: log-space forward-backward per sequence, gradients with respect to the activations (softmax
  normalization) or the probabilities.

### cuFFT (`blas/`)

`libcufft.so.12` (`blas/src/cufft.cu`) implements batched 1D, 2D and 3D complex-to-complex, real-to-complex and
complex-to-real transforms of half, float and double data with cuFFT's advanced layouts (`cufftXtMakePlanMany` /
`cufftXtExec`, which PyTorch uses, and the classic `cufftPlan*` / `cufftMakePlan*` / `cufftExec*` functions; work
areas allocated by the plan or by the caller). A transform gathers its input into a complex work buffer (inputs of
complex-to-real transforms are completed by Hermitian symmetry), runs self-sorting Stockham passes per dimension (mixed
radix: 2 to 8 and primes up to 61), and scatters the result in the output layout and type. Lengths with larger prime
factors use Bluestein's algorithm (a power of two convolution with a chirp whose spectrum is computed in double
precision when the plan is made). Half data is computed in float.

Lines of up to 4096 elements (2048 in double precision) are transformed in shared memory, several lines per thread
block (so strided axes are loaded and stored coalesced), with specialized radix 2, 3, 4, 5, 7 and 8 butterflies held
in registers; the first and last axes read the caller's input and write its output directly. Real transforms of an
even last axis 2m are complex transforms of length m of the pairs of values, split (or merged) in the same kernel,
and the other axes then transform the half spectrum. Longer lines run as Stockham passes of large radices (up to 256,
each a transform in shared memory); other lines (long strided axes, factors too large for a block) run a kernel per
radix pass.

### cuRAND (`blas/`)

`libcurand.so.10` (`blas/src/curand.cu`) implements cuRAND's host API: all generators (XORWOW, MRG32k3a, MTGP32,
MT19937, Philox4_32_10, Sobol' 32 / 64 bit and their scrambled forms), seeds, offsets, orderings, streams, host
generators (they run on the GPU; the results are copied to host memory), uniform, normal, log-normal, Poisson and
binomial results, Poisson distributions for kernels (`curandCreatePoissonDistribution`), and Sobol' direction vectors
and scramble constants. The generators are the public device API's (`curand_kernel.h`), arranged as cuRAND's
documentation specifies for each ordering: XORWOW's 4096 subsequences, MRG32k3a's 81920 (legacy: 4096), Philox's
65536 in groups of four, MTGP32's 192 (legacy: 64) parameter sets in blocks of 256 results. XORWOW, MRG32k3a and
Philox keep a state per subsequence between calls, so results split over several calls equal one large call. Sobol'
uses Joe and Kuo's direction numbers (`new-joe-kuo-6.21201`, 20,000 dimensions, packed by `blas/gen_sobol.py`).
MT19937 runs 8192 generators, generator g starting at output g * 2^1000 of the CPU version's sequence (the jumps are
polynomials computed by `blas/gen_mt_jump.cpp`), in groups of eight results. Poisson and binomial results come from
Marsaglia, Tsang and Wang's square histogram, or a normal approximation for variances above 2000 (as cuRAND's
documentation says for Poisson).

Raw bits and uniform results of XORWOW, MRG32k3a, MTGP32, Philox and Sobol' are bit for bit NVIDIA's, and so are all
distributions of MTGP32 and Sobol' and Philox's normals; the other results follow unpublished choices of cuRAND
(pairing of Box-Muller values, Poisson methods, MT19937's permutation, the scrambled Sobol' constants) and differ in
their values, not in their distributions.

### NCCL (`blas/`)

`libnccl.so.2` (`blas/src/nccl.cpp`) implements NCCL's communicators (`ncclCommInitRank*`, `ncclCommInitAll`, split,
shrink, destroy / abort), the collectives (all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all) with
all reduction operations including pre-multiplied sums, send / receive and group calls, for ranks in one process or in
several. The ranks of a communicator share a host memory segment (POSIX shared memory, named by the unique id), and
there are two transports:

- **Device transport** (GPU to GPU), when every rank's GPU reaches every other rank's memory (the same GPU, or peer
  access) and the driver has stream memory operations (`cuStreamWaitValue64` / `cuStreamWriteValue64`, LibreCUDA's or
  NVIDIA's). At initialization each rank allocates a region of device memory (flags, and per peer an inbox and an outbox
  of four chunk slots: 2 MB between two ranks, else 1 MB; as many again for graphs) and the ranks exchange them
  (interprocess handles, or pointers within a process). Each directed edge then carries chunks by push (the sender's
  kernel writes into the receiver's inbox) or by pull (the receiver's kernel reads the sender's outbox), whichever was
  measurably faster at initialization: on PCIe the GPU that issues the transfers matters. All-reduce, reduce-scatter and
  all-gather are rings; broadcast and reduce are chains; all-to-all and send / receive are direct. Each chunk step is a
  kernel that combines the received chunk with the rank's own data (half and bfloat16 through float) and writes the
  result to its output and / or the next rank's slot, then a flag write in the peer's memory; the next step waits for
  its flag on the GPU. A communicator's operations run in order on its own high priority stream, which joins the
  callers' streams; the calls return as soon as the work is enqueued, as NCCL's do.

  Between two ranks on different GPUs (tensor parallelism), all-reduces are exchanges, which keep both directions of
  the link busy: one rank (the one whose copy engine sends and whose kernels read the other's memory fastest) uploads
  its data with its copy engine into the other rank's inbox, while it reads the other rank's data (staged in its
  outbox) with a kernel of 32 blocks that combines as it reads; the other rank combines the uploaded data with its
  own. Both compute a + b (the same results), in chunks of about an eighth of the message (256 KB to 2 MB), and the
  transport's streams have high priority, so that the chunks keep flowing while the ranks compute. Up to 8 MB the
  kernels wait for their chunk and announce their progress themselves (a thread polling the flag, the last block
  writing the next one), which notices a flag sooner than the GPU's semaphore acquires. Above 16 KB and up to 1 MB
  (`LIBRECUDA_NCCL_DRIVEN_BYTES`, 0: none) the exchange is one kernel per rank, driven by the faster rank: the
  other rank copies its data into a staging buffer of its region and sets a flag in the driver's; the driver writes
  its data into the other rank's inbox (two slots, alternating) while it reads the staged data, combines, and sets a
  flag in the other rank's region, after which that rank combines too. Every transfer across the link is the
  driver's, in the directions where it is fast, and each thread issues its read before its write (on PCIe a read
  request doesn't pass the writes issued before it, so reads behind 64 KB of writes waited for them). Up to 64 KB
  (and up to 16 KB when the driven exchange runs) the exchange is one kernel per rank that pushes into the peer's
  inbox, waits for the peer's data and combines; these smallest exchanges run directly on the caller's stream when
  the communicator's previous operation used that stream too (no events around it). Up to 5 KB
  (`LIBRECUDA_NCCL_LL_BYTES`, at most 8 KB) that kernel uses LL stores (NCCL's low-latency protocol): each 16 byte
  store into the peer's slots carries 8 bytes of data and the operation's epoch twice, once per 8 bytes, which the
  link writes whole, so the receiver knows its data arrived by polling the data itself: no flag write after a fence,
  no slot release. Epochs alternate between two slot sets, and a rank only writes a set again after it received the
  peer's next epoch, which the peer sends once it read this one. The flags double the bytes sent, which costs more
  than it saves above about 6 KB here (the GPU on the Gen5 slot writes into the other at 2.3 GB/s). Operations
  captured into graphs keep the flag protocol.

  Collectives captured into CUDA graphs (`torch.cuda.graphs`, captured DDP or tensor-parallel layers) run in graph
  mode: a graph replays the values it was captured with, so sequence numbers baked into its nodes would let its waits
  pass at once on every replay, and stream memory operations can't be captured on LibreCUDA. Each region therefore has
  a second set of slots with flags per edge and slot that toggle: the sender sets the slot's ready flag to 1, the
  receiver resets it and sets the sender's freed flag (whoever waits for a flag resets it before it signals what lets
  the other side set it again). The kernels wait and signal themselves, on the captured stream (and a second stream
  joining the capture for the uploads of large pair all-reduces, between a kernel waiting for the slot and one
  announcing the chunk); the slots follow per edge cursors that advance at capture, so every replay uses the same
  slots in the same order on all ranks. Graphs and uncaptured operations use separate flags and slots and mix in any
  order (as long as all ranks issue and replay them in the same order, as with NCCL).
- **Host transport** otherwise, and for fp8 types: each rank has a data slot and each pair of ranks a mailbox in the
  shared segment. Collectives proceed in chunks of 16 MB: each rank copies its chunk into its slot, waits for the
  others, computes its result from all slots (reductions on the CPU) and copies it to its GPU. All operations of a
  group advance in one progress loop, so several ranks driven by one thread and matching sends and receives don't
  deadlock. The calls wait for the stream's earlier work and return when the results are on the GPU.

Environment variables: `LIBRECUDA_NCCL_P2P=0` (or NCCL's `NCCL_P2P_DISABLE=1`) selects the host transport;
`LIBRECUDA_NCCL_EDGES=push` or `pull` forces every edge of the device transport one way (for tests);
`LIBRECUDA_NCCL_ALGO=ring` makes two rank all-reduces rings too;
`LIBRECUDA_NCCL_DEBUG` prints the measured bandwidth and the choice of every edge.

**Symmetric memory and the device API** (NCCL 2.28's `nccl_device.h`; `blas/src/nccl_window.cu`). Kernels compiled
against NCCL's own device headers communicate themselves: they load from and store to every rank's memory and
synchronize with LSA barriers, so communication can be fused into computation. `ncclMemAlloc` allocates with the
driver's virtual memory API (`cuMemCreate`, exportable to a file descriptor). `ncclCommWindowRegister` (collective)
maps every rank's allocation behind the buffer into one range per window, rank r's at r times a stride of whole
4 GiB, which is the layout `ncclGetLsaPointer` / `ncclGetPeerPointer` compute with; the allocations travel between
processes as file descriptors over Unix sockets. `ncclDevCommCreate` allocates a zeroed resource window for the LSA
barriers, the LL all-to-all buffers and the caller's resource buffers and fills `ncclDevComm`; teams
(`ncclTeamWorld`, `ncclTeamLsa`, `ncclTeamRail`) and the requirement helpers are there too. Only the driver API's
public functions are used (LibreCUDA's or NVIDIA's `libcuda.so.1`), so it runs on both runtimes, and kernels written
for it run unchanged with NVIDIA's NCCL (`tests/nccl/run.sh` runs `tests/nccl/device_api.cu` all three ways). The LSA
team is the whole communicator; windows are ordinary memory to the collectives.

### CUDA driver API (`libcuda.so.1`)

`runtime/src/cuda_driver.cpp` implements the CUDA driver API (`cuda.h`) in the runtime's library, on the runtime, so
both APIs work on the same objects, as with NVIDIA's libraries:

- **Contexts.** A device's primary context is the runtime's device. Each thread has a context stack; making a context
  current (`cuCtxSetCurrent`, `cuCtxPushCurrent`) makes its device the runtime's current device, and `cudaSetDevice`
  or the first runtime call that needs a device makes the device's primary context current on a thread without one
  (device queries don't). Contexts created with `cuCtxCreate` are further handles of the device's primary context.
- **Objects.** `CUstream` and `CUevent` are the runtime's streams and events (including the legacy and per-thread
  default stream handles), `CUgraph` / `CUgraphExec` its graphs, `CUmemoryPool` its memory pools.
- **Modules.** `cuModuleLoadData(Ex)`, `cuModuleLoad` and `cuModuleLoadFatBinary` take cubins, fat binaries (picked
  like the runtime's) and PTX (compiled with `ptxas`). Modules are loaded on the current device right away (load
  errors are reported there) and on other devices when used there. Their functions are kernel handles of the runtime
  (like `cudaKernel_t`), so `cuLaunchKernel`, `cuLaunchKernelEx` (launch attributes are the runtime's) and
  `cuLaunchCooperativeKernel` go through the runtime's launch path: stream semantics, stream capture, graphs and
  function attributes / occupancy all apply. Parameters can be passed as `kernelParams` or as a buffer
  (`CU_LAUNCH_PARAM_BUFFER_POINTER`), laid out with the kernel's parameter offsets. `cuLink*` links a single input
  (cubin, fat binary or PTX).
- **Memory.** Allocation (device, host, managed, pitched, async), copies (1D, 2D and 3D of linear memory, peer),
  fills of 8, 16 and 32 bit elements (1D and 2D), address ranges, pointer attributes, prefetches, advice and range
  attributes, memory pools.
- **Virtual memory management** (`tests/cudart/vmm.cu`). `cuMemCreate` allocates device memory (an RM memory object
  of 2 MB pages; the granularity is 2 MB) and `cuMemAddressReserve` reserves addresses, which are never reused.
  `cuMemMap` maps a whole allocation (offset 0, its size) into a reserved range, the same UVM external range kind as
  other device memory, and `cuMemSetAccess` maps it on further devices (peer access). `cuMemRetainAllocationHandle`,
  `cuMemGetAllocationPropertiesFromHandle` and `cuMemRelease` (the memory lives until its last mapping is gone) work
  as CUDA's. Allocations export to POSIX file descriptors (`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`): the
  descriptor is a memory file holding the allocation's interprocess handle, which `cuMemImportFromShareableHandle`
  opens in any process. Read-only access is read-write; no other handle types, partial mappings or compression.
- **Stream memory operations.** `cuStreamWaitValue32/64` (GEQ, compared cyclically like CUDA, EQ, AND and NOR),
  `cuStreamWriteValue32/64` and `cuStreamBatchMemOp` on any address the GPU reaches (device memory, a peer's memory,
  mapped host memory): host semaphore acquires on the stream's channel (the channel group yields to others while the
  value isn't there yet) and releases once the channel's preceding work completed, writing only the 4 or 8 bytes
  (`tests/cudart/stream_memops.cu`). They can't be captured into graphs.
- **Entry points.** `cuGetProcAddress` (and the runtime's `cudaGetDriverEntryPoint(ByVersion)`) return the newest
  version of each function. `cuGetExportTable` finds nothing: NVIDIA's private tables aren't public.
- **Errors.** Driver API calls report `CUresult` values (the runtime's codes are the driver's, apart from runtime-only
  codes, reported as `CUDA_ERROR_INVALID_VALUE`) and leave the runtime's `cudaGetLastError` unchanged. Error names are
  generated from `cuda.h`.

`libcuda.so.1` is an ELF filter library: its symbol table lists the driver API functions, and the dynamic loader
resolves them in `libcudart.so.13` (found through its `$ORIGIN` run path). Programs and libraries that link against
libcuda or `dlopen` it therefore use the runtime's state.

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
  print the completed records (a device synchronization once, after waiting for all streams; a wait finds out from
  the record counter whether there is anything to print before it checks which streams are idle). Details that matter
  on the GPU:
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
  - LibreCUDA can't launch kernels from the GPU, so the host does: a launch of a kernel that launches kernels waits
    for the kernel (the host launch call is synchronous), then launches the requested kernels on the same
    stream, depth first: each child that may launch kernels is waited for and its own requests launched before the next
    child (tail launches after the other children; nesting up to 24 levels). Children therefore only start after their
    parent completed, and all device side streams behave as one ordered stream.
  - Device side `cudaMemcpyAsync` / `cudaMemsetAsync` and their 2D / 3D forms are queued as slots too and run on the
    stream in order with the launches. A `cudaMemcpy3DAsync` involving CUDA arrays (which NVIDIA's device runtime
    refuses) carries its parameters to the host, where the runtime does the array copy. Device side events (create, record,
    `cudaStreamWaitEvent`, destroy) do nothing: the device side streams are one ordered stream anyway.
  - `cudaGetLastError` / `cudaPeekAtLastError` are per thread: an entry per hardware thread slot (SM, warp slot, lane),
    tagged with `%gridid` and the thread's position so that a later thread in the slot doesn't see it.
  - Also provided: `cudaGetDevice`, `cudaGetDeviceCount`, `cudaStreamCreateWithFlags` / `cudaStreamDestroy`,
    `cudaMalloc` / `cudaFree` (the `malloc` heap), `cudaRuntimeGetVersion`, `cudaDeviceGetAttribute` and
    `cudaDeviceGetLimit` (tables the host writes into the device runtime's state), `cudaDeviceGetCacheConfig` /
    `cudaDeviceGetSharedMemConfig`, `cudaGetErrorName` / `cudaGetErrorString` (the runtime's error strings compiled
    into the device runtime), and `cudaFuncGetAttributes` / `cudaOccupancyMaxActiveBlocksPerMultiprocessor(WithFlags)`
    (from records of the context's kernels that the driver writes at module load, computed like the host runtime).
  - Device function pointers (`-rdc`) need nothing special: nvlink turns their relocations into the absolute address
    types the loader applies, and calls are `CALL.ABS` on the loaded address.
  - Kernels in such modules get at least the device runtime's register count (nvlink sized them for NVIDIA's code).
  - Which kernels launch kernels (or copies, fills, graphs) from the device follows from the call graph in the image's
    relocations: a kernel whose code, or a function it calls (`CALL.ABS` relocations), refers to a launch function.
    Other kernels of the module launch asynchronously and can be replayed from prebuilt graph segments. Functions
    called through pointers aren't visible in the call graph: if a function whose address is stored in data launches
    kernels, all kernels of the module count as launching.
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
  with the fewest live streams (round robin among equals, so channels of destroyed streams are reused). All channels
  are created with the context: GSP adds no subcontext or channel to the group while a kernel of the context runs on
  and on (an NCCL graph's kernel waiting for its peer, say: the allocation fails after 4 s, then Xid 109), and idle
  channels cost nothing measurable (launch + synchronize 5.4 µs with one or all eight in use). A new stream submits its
  setup without waiting for it, so it can be created next to such a kernel even on the kernel's channel. Kernels
  within a stream stay ordered, and a stream's completion only waits for its own channel's work.
- **Stream priorities.** Two levels, 0 and -1 (`cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`).
  High priority streams get the last compute channel (normal streams use the others) and their grids a QMD group of
  their own: they get SMs as soon as some free up, while equal priority grids wait until the grids launched before
  them got all theirs (`tests/cudart/stream_priority`: a kernel launched during a 40 ms grid runs after 5 ms instead
  of 40). Communication kernels overlapping computations need this (see NCCL).
- **Module loader.** It parses the CUDA ELF image produced by nvcc:
  - lays out code, constant banks, initialized (`.nv.global.init`) and zero-initialized (`.nv.global`) globals in GPU
    memory and applies relocations (with addends),
  - exports `__device__`/`__constant__` symbols,
  - reads kernel parameter offsets and sizes, the parameter base in constant bank 0, register counts, static shared
    memory and stack sizes from the `.nv.info` attributes.
- **Kernel launch.** Each launch writes constant bank 0 (driver data + parameters) and a QMD (version 5 on Blackwell)
  into a kernel argument ring in GPU memory, then sends it through the compute channel. Kernels that call functions
  (recursion, function pointers, virtual functions) get a stack of `cudaLimitStackSize` per thread, like in CUDA.
  The QMD asks for the shared memory carveout (SM configuration: 32, 64 or 100 KB) that holds as many CTAs per SM as
  threads, registers and the CTA limit allow, as NVIDIA's driver does; fitting a single CTA, as it used to, ran
  llama.cpp's prompt flash attention (35 KB per CTA, two per SM) 30% slower. A kernel that follows another on a
  stream costs about 1.3–1.7 µs of GPU time (a semaphore release with wait for idle orders them), against 1.25 µs on
  NVIDIA's runtime. In a graph segment the QMDs are chained instead (below): 0.75 µs per kernel, against 0.63 µs in a
  CUDA graph on NVIDIA's runtime (tiny dependent kernels). llama.cpp's decode (Qwen3-4B bf16, two GPUs, about 500
  kernels per token) went from 78.2 to 79.3 tokens/s with the chaining (`-sm tensor`; NVIDIA's runtime: 82.5) and runs
  at 49.2 tokens/s with `-sm layer` (NVIDIA: 49.9); the remaining 4% of `-sm tensor` (0.5 ms per token) is not the
  ordering of a segment's kernels.
- **Stream ordering.** Every operation of a stream advances the stream's timeline semaphore.
  - Within a channel, operations execute in order.
  - When a stream switches between the compute channel and a copy channel, the new channel performs a GPU side
    semaphore acquire. No CPU round trip is needed to order compute and copy work.
  - Events are semaphore releases with timestamps and can be waited upon from other streams. `cudaEventDestroy`
    doesn't wait for queued work recording the event (frameworks destroy events right after enqueuing them, e.g. one
    per PyTorch collective): an event still pending is released once it completed. The pending ones are looked at
    again once there are twice as many as at the last look, so creating and destroying events costs about 0.1 µs
    however many are pending (it used to grow with their number: 94 µs per destruction with 2000 pending). A context's
    pool of event semaphores grows by 4096 when all are taken. An event recorded again before its previous record
    was reached gets a semaphore of its own for the new record: the two releases may come from different streams and
    land in either order, and the earlier one landing last used to set the semaphore back, so that waits for the new
    record never completed (two-rank PyTorch `DistributedDataParallel` hung in about every other run: LibreCUDA's NCCL
    records one event on the caller's stream and again on the next stream it joins). A semaphore goes back to the
    pool once its last record was reached, and its value is never reset: its next event counts on from it, so waits
    for its earlier records that haven't executed yet still find their targets reached.
- **Command buffers.** Command buffers and the GPFIFO ring live in video memory. The GPU never fetches commands over
  PCIe. The CPU writes them (and GPPut) through BAR1 and then reads GPPut back before ringing the doorbell, so the writes
  have reached video memory when the GPU looks for new work. Without the read-back the GPU could see a stale GPPut and
  go idle, leaving the submission unexecuted.
  - The read-back is a PCIe round trip (about 2 µs on the test system's x4 2.5 GT/s link, half of an async launch), so
    the doorbell of back-to-back submissions is held back: a channel whose doorbell rang less than
    `LIBRECUDA_DOORBELL_DELAY_US` (default 20 µs) ago only collects ring entries (up to 16). A thread per context rings
    them after the delay, and every wait for the GPU (synchronization, queries, ring space) rings pending doorbells
    first, so synchronization never waits for the timer. Async launches: 3.99 → about 2.3 µs; launch + synchronize
    is unchanged. A wait rings the other contexts' pending doorbells too (unless one is submitting at that moment): the
    awaited work may wait for another GPU's (a kernel polling memory the other GPU's kernel writes, as llama.cpp's
    all-reduce does), whose submission would otherwise sit out the delay. Two GPUs' kernels meeting through host
    memory with a host wait per round took 35-42 µs per round instead of 5.4 (`tests/cudart/multi_gpu`).
- **Memory operations.**
  - Device to device copies and device memsets use vectorized grid-stride kernels; they are about 2.7x and 5x faster
    than the copy engine. Copies between GPUs (peer access) use the copy engine of the stream's GPU: as fast over the
    link, and it leaves the SMs and the other direction of the link to other work (with a kernel reading the peer's
    memory at the same time, 4.9 + 6.4 GB/s on the test system instead of 3.9 + 3.5 with two kernels).
  - Host transfers use the copy engine. Pageable memory is staged through pinned double buffers.
  - Host to device and device to host copies run on two separate async copy engines, so both directions overlap
    (`LIBRECUDA_ASYNC_COPY_ENGINES=0` falls back to the graphics copy engine for both).
- **Memory allocation.** Device allocations of up to 1 MiB come from a per-context pool: power-of-two size classes
  (256 B to 1 MiB) carve 2 MiB chunks into slots, so small `cudaMalloc`/`cudaFree` calls make no driver calls (a chunk
  is five: an RM allocation, a UVM range and its mapping, and their release; NVIDIA's runtime packs up to 1 MiB the
  same way). One empty
  chunk per size class is kept; others are released. Larger allocations are separate RM allocations.
- **CUDA graphs and stream capture** (record and replay). A graph is a set of nodes (kernels with copies of their
  arguments, copies, fills, host functions, child graphs, event records / waits, memory allocations and frees, empty
  nodes) and dependencies. Instantiation copies the graph, orders the nodes topologically and splits them into lanes
  (chains of nodes each depending on the previous one). `cudaGraphLaunch` replays the nodes through the same paths as
  the calls they stand for, the first lane on the launch stream and the others on internal streams (up to 8 lanes,
  `LIBRECUDA_GRAPH_LANES`), so independent branches run concurrently. Dependencies between lanes are events; the lanes
  start after the launch stream's earlier work and the launch stream waits for all of them. The internal streams
  belong to the launch stream and serve every graph launched on it (those run one after another anyway): per graph,
  vLLM's piecewise graphs would need thousands of streams, each with 768 KB of command pages.
  - Consecutive kernel nodes of a lane (no dependencies to or from other lanes in between) form a segment that is
    replayed from prebuilt commands (a driver command list, `libreCuCommandList*`): on its first launch on a device,
    every kernel's constant bank 0 and QMD are written once to video memory together with a pushbuffer that launches
    the first. On Blackwell each QMD schedules the next one when its grid completed (`DEPENDENT_QMD0`, action
    `SCHEDULE`, as tinygrad chains its launches): the grid's end flushes its writes from L1 (`CWD_MEMBAR_TYPE`) and
    the next grid invalidates L1 and its constant banks when it starts, and the last QMD adds 1 to a completion
    counter of the list (a QMD semaphore release with system membar), which the stream acquires before it goes on. A
    replay of the list waits for the counter of the list's previous replay too, which may have been on another
    stream. Instead of a host semaphore release with wait for idle between the kernels (which waits for the channel's
    own kernels only; the compute class's `WAIT_FOR_IDLE` stalls the whole engine; still used before Blackwell),
    which costs 1.3 µs of GPU time per kernel, a chained kernel costs 0.75 µs (NVIDIA: 0.63 µs). Dropping the
    end-of-grid membar (0.56 µs) or the L1 invalidation (0.66 µs) would be faster but not safe; a host semaphore
    acquire of a QMD release after each kernel is slower (2 µs). A replay submits the pushbuffer as one GPFIFO entry:
    about 1.1 µs per kernel node instead of 3.9 µs for a launch (NVIDIA: about 0.4 µs). Changed
    node parameters rebuild the segment at the next launch; kernels that need host work per launch (local memory,
    kernels that launch kernels) and cooperative kernels are launched individually.
  - Conditional nodes (IF / IF-ELSE, WHILE, SWITCH): a conditional handle is the device address of the condition
    value, which `cudaGraphSetConditional` in device code stores (redirected by the loader to the device runtime).
    The node waits for its lane's preceding work, reads the value and runs the selected body graphs on the lane, so
    graph launches with conditional nodes wait on the host at each condition. `cudaGraphCondAssignDefault` handles get
    their default value at the start of each launch.
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
  or with `cudaGraphExecUpdate` (same topology, no memory nodes). Not supported: external semaphores.
  - Device-updatable kernel nodes (`cudaLaunchAttributeDeviceUpdatableKernelNode`, on a node or in a captured
    launch): the node's `cudaGraphDeviceNode_t` is the address of a record in pinned host memory with the kernel's
    parameters at their offsets in the parameter buffer, the grid and the enabled state. The device side
    `cudaGraphKernelNodeSetParam` / `SetGridDim` / `SetEnabled` / `UpdatesApply` (redirected to the device runtime)
    write it and set flags; each launch of the executable graph first takes over flagged updates (rebuilding the
    node's prebuilt segment). Host updates of the node replace the device's. Like CUDA, such graphs have one
    executable graph and can't be launched from the device.
  - Edge data is kept per dependency and validated like CUDA (programmatic edges between kernel nodes from their
    programmatic or launch completion port). Programmatic edges and programmatic dependent launches
    (`cudaLaunchAttributeProgrammaticStreamSerialization`, captured as programmatic edges) run like full
    dependencies, which is a valid schedule for them.
  - Graphs launched from device code (`cudaGraphLaunch` in a kernel, graphs instantiated with
    `cudaGraphInstantiateFlagDeviceLaunch`) go through the device launch queue like dynamic parallelism: the host
    runs a fire and forget launch right after the launching kernel and a tail launch after the whole launching graph
    (tail launches of tail launches in order, so a graph can relaunch itself). `cudaGetCurrentGraphExec` reads the
    executable graph the driver writes to constant bank 0 (`%current_graph_exec`: `c[0][0x2e8]` on sm_120,
    `c[0][0x120]` before) for kernels launched while a graph replays.
- **Threads.** The driver API is thread safe (a context must not be destroyed while other threads use it). Each
  stream's queue has its own lock. Writes to the shared GPFIFO rings, lazily set up context state (modules, the device
  runtime, local memory, compute channels, event signals, texture pools), allocation bookkeeping, printf output and
  channel error classification each have their own lock, taken in a fixed order (documented in
  `librecuda_internal.h`). The current context is per thread, like in CUDA. A wait for a stream releases the
  stream's lock while it waits, so other threads can keep enqueuing work on it. Waits never hold a context-wide lock.
  The runtime's API lock is a reader/writer lock. Work on streams other than the legacy default stream
  (kernel launches, async copies and fills, event records and waits, stream queries and synchronization) and
  `cudaMalloc` hold it shared plus the stream's own mutex, so threads using different streams submit in parallel, and
  so do event creation and destruction (with a mutex of their own; destroying an event recorded during an ongoing
  capture takes the lock exclusively).
  Work on the legacy default stream changes the state of every blocking stream, so it holds the lock exclusively, as
  do the remaining calls. Lazy device initialization and module loading have locks of their own. Waits for the GPU
  (stream, event and device synchronization, synchronous and staged copies, launches of kernels that launch kernels)
  and host functions / stream callbacks run without the API lock. While waiting, a thread holds a lifetime lock in
  shared mode, which `cudaStreamDestroy` and `cudaDeviceReset` take exclusively before freeing what a waiter may use;
  `cudaEventSynchronize` holds another one shared that `cudaEventDestroy` takes exclusively, so destroying an event
  doesn't wait for threads waiting for streams or devices (PyTorch's NCCL watchdog destroys events while the main
  thread synchronizes). A thread waiting for work that another thread's later call enables therefore doesn't
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
runtime (`build/lib/libcudart.so.13`) and the CUDA driver API (`build/lib/libcuda.so.1`). If nvcc and CUTLASS
(`git submodule update --init third_party/cutlass`) are found, it builds cuBLAS too (`build/lib/libcublas.so.13`,
`build/lib/libcublasLt.so.13`) for `CMAKE_CUDA_ARCHITECTURES` (default 120).

### Run an unmodified CUDA binary

Programs built with `nvcc -cudart=shared` load `libcudart.so.13` at run time. Point the dynamic loader at LibreCUDA's:

```bash
nvcc -O3 -arch=sm_120 -cudart=shared my_program.cu -o my_program
LD_LIBRARY_PATH=build/lib ./my_program
```

`ldd my_program` then shows LibreCUDA's `libcudart.so.13`. Neither libcuda nor NVIDIA's libcudart is loaded.
Alternatively, link directly: `nvcc -cudart=none my_program.cu -Lbuild/lib -lcudart`. Programs using the driver API
(`nvcc ... -lcuda`, linked against the toolkit's stub or LibreCUDA's `libcuda.so`) load LibreCUDA's `libcuda.so.1` the
same way.

### PyTorch

PyTorch's CUDA builds bring their own copies of NVIDIA's libraries and find them through `RPATH` (which
`LD_LIBRARY_PATH` doesn't override), and load libcuda with `dlopen`. Preload LibreCUDA's libraries instead; libraries
asking for the same sonames then get them:

```bash
L=build/lib
LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcusolver.so.12:\
$L/libcusparse.so.12:$L/libcudnn.so.9:$L/libcufft.so.12:$L/libnccl.so.2 python3 my_script.py
```

Tested with torch 2.11.0+cu130 (`tests/pytorch/eager_ops.py`): tensor creation and copies (including pinned memory and
half / bfloat16 conversions), elementwise math, broadcasting, reductions and scans, indexing, gather / scatter,
sort / topk / unique, shape operations, random numbers, activations, softmax, layer norm, cross entropy, embeddings,
pooling and interpolation, complex numbers, operations PyTorch compiles at run time with NVRTC (jiterator), matrix
products of all types (`mm`, `bmm`, `baddbmm`, `mv`, `dot`, complex), `nn.Linear` (cuBLASLt with bias), convolutions
without cuDNN, autograd, Adam training loops of an elementwise model and an MLP, streams and events and the caching
allocator all match the CPU's results. A 4 layer transformer encoder trained with bfloat16 autocast reaches the same
loss as on NVIDIA's CUDA. `import torch` still maps NVIDIA's `libcudart.so.13` and `libcublas*.so.13` (torch preloads
them with `ctypes`, `RTLD_LOCAL`), but every lookup resolves to LibreCUDA's preloaded libraries.

`torch.linalg` works (`tests/pytorch/linalg_ops.py`): `solve`, `inv`, `det` / `slogdet`, `lu` / `lu_factor` /
`lu_solve`, `cholesky` (and `cholesky_ex`, `cholesky_solve`), `qr`, `geqrf` / `householder_product`, `lstsq`,
`solve_triangular`, `matrix_exp`, `eigh` / `eigvalsh`, `eig` / `eigvals`, `svd` / `svdvals` (reduced and full),
`pinv`, `matrix_rank`, matrix norms, and gradients through decompositions, for single and batched matrices of float,
double and complex types.

cuDNN operations work (`tests/pytorch/cudnn_ops.py`, with and without `TORCH_CUDNN_V8_API_DISABLED=1`): convolutions
(1d, 2d, 3d, strided, dilated, grouped, depthwise, channels last, transposed) with their gradients in all floating
point types, batch normalization, pooling, `affine_grid` / `grid_sample`, LSTM / GRU / RNN (packed sequences, states,
dropout, half precision), the CTC loss, and the training of a small convolutional network.

`torch.fft` works (`tests/pytorch/fft_ops.py`): `fft` / `ifft` / `rfft` / `irfft` / `hfft` of any length (mixed radix
and Bluestein), 2D and N-D transforms with all normalizations, non-contiguous inputs, half and double precision,
`stft` / `istft` and gradients.

CUDA tensors can be shared between processes with `torch.multiprocessing` (`tests/pytorch/multiprocessing_ops.py`:
tensors written just before they are sent, views into one storage, 256 MB tensors; the consumer's in-place changes
are seen by the producer).

`torch.distributed` with the NCCL backend works (`tests/pytorch/nccl_ops.py`, two processes on two GPUs, or
`NCCL_TEST_WORLD` processes spread over them): all
collectives for all data types and reduction operations, send / receive and batched point-to-point operations, object
collectives, process groups, `torch.cuda.nccl` in one process, `DistributedDataParallel` training with
`SyncBatchNorm`, which matches training on one GPU, and collectives captured into a CUDA graph and replayed with new
inputs, with uncaptured ones in between.

Sparse tensors work (`tests/pytorch/sparse_ops.py`, preloading `libcusparse.so.12`): COO, CSR and BSR matrices times
dense matrices and vectors (`sparse.mm`, `addmm`, `addmv`, half / bfloat16 too), CSR @ CSR and COO @ COO, CSR + CSR,
`sampled_addmm`, triangular solves with sparse CSR and BSR matrices, conversions between layouts and autograd through
`sparse.mm`. NVIDIA's own math libraries can't stand in for missing LibreCUDA ones: they carry a static copy of NVIDIA's
runtime, which needs libcuda's private interfaces (`cuGetExportTable`).

`torch.compile` works (`tests/pytorch/compile_ops.py`): TorchInductor's Triton kernels are compiled with `ptxas` and
loaded and launched through LibreCUDA's driver API (Triton's launcher links libcuda, which resolves to LibreCUDA's).
Tested: pointwise fusion, reductions and normalization, dynamic shapes, indexing and scatter, compiled backward passes,
Triton GEMM templates with `max-autotune` (Inductor offers them only on GPUs with 68 SMs or more; the test lifts that
limit), CUDA graphs (`mode="reduce-overhead"`, on the runtime's graphs) and a transformer layer under bfloat16 autocast.
Triton's TMA kernels (host-side tensor descriptors, `cuTensorMapEncodeTiled`) aren't supported; Inductor doesn't use
them by default.

### vLLM

vLLM's prebuilt wheels run on LibreCUDA unchanged: their extensions link `libcudart.so.13` and `libcuda.so.1`
dynamically, so preloading LibreCUDA's libraries is enough. Tested: vLLM 0.27.1 (PyTorch 2.13, CUDA 13) serving
Qwen3-4B in bf16 with tensor parallelism 2 on the two RTX 5060, with torch.compile, piecewise and full CUDA graphs,
FlashAttention and vLLM's custom all-reduce (CUDA IPC) next to LibreCUDA's NCCL:

```bash
L=build/lib
export LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcudnn.so.9:$L/libnccl.so.2
export VLLM_NCCL_SO_PATH=$L/libnccl.so.2   # vLLM loads its NCCL itself
export LIBRECUDA_PTXAS=<venv>/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas   # PTX newer than the toolkit's
CUDA_VISIBLE_DEVICES=0,1 vllm bench latency --model Qwen/Qwen3-4B --tensor-parallel-size 2 --max-model-len 2048 \
    --gpu-memory-utilization 0.80
```

Kernels that come as PTX only (for sm_120) are compiled with `ptxas` when first loaded, which needs a `ptxas` at least
as new as the CUDA that generated the PTX (here the pip package's 13.4; the toolkit's 13.2 failed). With greedy
sampling the generations match NVIDIA's CUDA token for token for 26 to 64 tokens, then may take another, equally
coherent turn: LibreCUDA's GEMMs and all-reduces round bf16 sums in a different order.

Other models and features, each run on NVIDIA's CUDA and on LibreCUDA with greedy outputs compared (vLLM 0.27.1, the two
RTX 5060): Qwen3.5-0.8B (linear attention layers: Triton / FLA kernels, on one GPU and with tensor parallelism 2),
Qwen2.5-0.5B, Qwen3-4B-AWQ (4 bit AWQ through the Marlin kernels; identical outputs), an FP8 KV cache, n-gram
speculative decoding, long prompts with prefix caching, JSON-schema structured output, top-k / top-p / min-p sampling
with logprobs and repetition penalty, beam search, embeddings (Qwen3-0.6B as a pooling model) and eager mode all run;
their outputs match NVIDIA's or take a different, equally coherent turn after 20 to 150 characters.

- NVFP4 models (Qwen3.5-0.8B-NVFP4, Qwen3-0.6B-NVFP4) run with vLLM's Marlin FP4 kernels, which take
  `VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4LinearKernel,FlashInferCutlassNvFp4LinearKernel,CutlassNvFp4LinearKernel`:
  its native FP4 GEMMs for sm_120 are CUTLASS kernels using TMA, whose descriptors LibreCUDA can't make (tensor maps,
  see Limitations). Without the variable they fail to start ("Failed to initialize the TMA descriptor 801").
- FP8 weights (`--quantization fp8`, FP8 checkpoints) run through PyTorch's `torch._scaled_mm`, that is LibreCUDA's
  cuBLASLt FP8 matmuls, with
  `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel`: vLLM's own FP8 GEMMs for
  sm_120 are CUTLASS and FlashInfer kernels using TMA (launched with a missing descriptor they fault, Xid 31). Qwen3-4B
  with online FP8 quantization (per-token activation scales: vLLM's `ChannelWiseTorchFP8ScaledMMLinearKernel`, a
  `_scaled_mm` with float output, then the scales) gives coherent outputs; with the same kernels on NVIDIA's CUDA one
  of 4 generations is identical and the others take another turn after 10 to 34 characters, about as far as NVIDIA's
  own CUTLASS and `_scaled_mm` paths agree with each other (2 of 4 identical, the others after 30 and 34).

### SGLang

SGLang's wheels run on LibreCUDA unchanged too. Tested: SGLang 0.5.22 (PyTorch 2.14, CUDA 13, FlashInfer 0.7.1)
serving Qwen3-4B in bf16 with tensor parallelism 2, with piecewise prefill and full decode CUDA graphs and FlashInfer
attention:

```bash
export LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcudnn.so.9:$L/libnccl.so.2
export SGLANG_NCCL_SO_PATH=$L/libnccl.so.2   # SGLang loads its NCCL itself
export LIBRECUDA_PTXAS=<venv>/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas
export FLASHINFER_USE_CUDA_NORM=1            # FlashInfer's CUDA RMSNorm instead of its CuTe DSL one
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.benchmark.one_batch --model-path Qwen/Qwen3-4B --tp-size 2 \
    --batch-size 1 --input-len 512 --output-len 128 --mem-fraction-static 0.8 --context-length 2048
```

Kernels written with NVIDIA's CuTe DSL (FlashInfer's default RMSNorm on sm_120, FlashAttention 4) don't run: their
runtime asks the driver for private export tables (`cuGetExportTable`), which LibreCUDA doesn't have. SGLang's custom
all-reduce doesn't start (it falls back to NCCL): its CUDA runtime wrapper picks the copy of NVIDIA's `libcudart` that
torchvision bundles under another name (`torchvision.libs/libcudart.<hash>.so.13`, which preloading can't replace), and
NVIDIA's runtime on LibreCUDA's driver fails for the same reason. Greedy generations match NVIDIA's CUDA token for token
for at least 30 tokens.

### Environment variables

| Variable | Effect |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Comma separated GPU indices (index in the driver's GPU list). Only these GPUs are opened. |
| `LIBRECUDA_DEBUG=1` | Debug output from the driver API and the runtime. |
| `LIBRECUDA_WAIT_TIMEOUT_MS` | Fail waits for the GPU after this many milliseconds (`cudaErrorLaunchTimeout`) instead of waiting forever. |
| `LIBRECUDA_COMPUTE_CHANNELS` | Number of compute channels streams are spread over (1–16, default 8). `1` runs all kernels one after another. |
| `LIBRECUDA_DOORBELL_DELAY_US` | How long after a doorbell back-to-back submissions only collect ring entries before the doorbell rings (default 20). `0` rings the doorbell for every submission. |
| `LIBRECUDA_GRAPH_LANES` | Maximum number of streams a CUDA graph launch runs its independent branches on (1–64, default 8). `1` replays graphs serially on the launch stream. |
| `LIBRECUDA_SMALL_ALLOC_POOL=0` | Disable the pool for small device allocations (every `cudaMalloc` becomes its own RM allocation). |
| `LIBRECUDA_ASYNC_COPY_ENGINES=0` | Use the graphics copy engine for all copies instead of two async copy engines (host→device and device→host). |
| `LIBRECUDA_MANAGED_BAR1=1` | Managed memory as device memory mapped through BAR1 (no page migration, uncached CPU reads) instead of UVM managed memory. |
| `NVIDIA_TF32_OVERRIDE=0` | cuDNN: no TF32 for float convolutions. |
| `LIBRECUDA_CUDNN_IM2COL=1` | cuDNN: run all convolutions as im2col and GEMMs (for comparisons). |
| `LIBRECUDA_PERF_BOOST=0` | Don't boost the GPU's performance state while a context exists (the GPU then raises its clocks by itself under load). |
| `LIBRECUDA_PTXAS` | ptxas binary used to compile PTX-only fat binaries (default: `$CUDA_HOME/bin/ptxas`, `/usr/local/cuda/bin/ptxas`, `PATH`). |
| `LIBRECUDA_PROFILE_API=1` | At exit, print the calls and host time of every runtime function (those taking the API lock; the outermost call of a thread), most time first. |

### Tests

```bash
# driver API tests (compile their kernels for the GPU's architecture)
cmake -B build -DLIBRECUDA_TEST_ARCH=sm_120 && cmake --build build -j
# CUDA runtime tests: compiled with nvcc (only when changed) and run on LibreCUDA's runtime
tests/cudart/run_tests.sh build
# the runtime tests 20 times in a row, checking the GPU's PCIe errors (AER) and Xids after each run
tests/cudart/stress.sh build 20 02:00.0
```

`tests/cudart` contains 50 self-checking CUDA programs (`multi_gpu` needs two devices and passes as skipped with one):
- tiled and register-blocked SGEMM; reductions with warp shuffles and atomics,
- `__constant__`/`__device__` globals, function pointers and virtual functions,
- local memory and recursion; dynamic shared memory beyond 48 KiB,
- scan, radix and bitonic sort; n-body,
- streams and events; asynchronous host functions; concurrent kernels on different streams; cooperative launches with `grid.sync()`; thread block clusters with distributed shared memory; device side `printf`, `assert`, `malloc` and `free`; separate compilation (`-rdc`); dynamic parallelism and device runtime queries; several host threads; texture objects on linear and pitched 2D memory and CUDA arrays; surface objects; CUDA graphs and stream capture (concurrent branches, memory nodes, conditional nodes, device side launches and node updates, edge data, generic node parameters, user objects); several devices with and without peer access; managed memory and its policy queries; error reporting for SM exceptions; memory operations; many small allocations; `cudaDeviceReset`; Thrust and CUB; PTX JIT; the CUDA driver API; cuBLAS and cuBLASLt (its FP8 matmuls checked against NVIDIA's cuBLASLt too); cuSOLVER and cuBLAS's level 1-3 and batched routines; cuRAND's host API and cuSPARSE (both checked against NVIDIA's libraries too).

## Performance

In summary, LibreCUDA against NVIDIA's CUDA (runtime 13.2, driver 610.57.04, NVIDIA's libraries; NVIDIA's NCCL at its
fastest setting for each size) on the test system's two RTX 5060, from the tables below:

| Area | Benchmark | LibreCUDA | NVIDIA | LibreCUDA vs NVIDIA |
|---|---|---|---|---|
| Runtime | Async kernel launch | 1.14 µs | 1.28 µs | 1.12x |
| | Launch + synchronize round trip | 5.59 µs | 4.67 µs | 0.84x |
| | Pinned host transfers, 256 MiB (H2D / D2H) | 14.5 / 14.3 GB/s | 14.5 / 14.3 GB/s | 1.00x |
| | Device memory (triad kernel) | 392 GB/s | 387 GB/s | 1.01x |
| | `cudaMalloc` / `cudaFree`, 1 MiB | 16 / 14 µs | 18 / 15 µs | 1.1x |
| | Managed memory (migration, CPU access) | | | about 1.0x |
| CUDA graphs | Chain of 100 empty kernels | 57 µs | 41 µs | 0.72x |
| | 4 branches × 25 kernels with work | 826 µs | 818 µs | 0.99x |
| Several GPUs | `cudaMemcpyPeer` 64 MiB, GPU 0 → 1 / 1 → 0 | 7.04 / 7.14 GB/s | 2.15 / 7.14 GB/s | 3.3x / 1.0x |
| | Launch + sync round trips from 4 threads, one GPU | 404 per ms | 924 per ms | 0.44x |
| All-reduce (NCCL) | 4 KB | 5.6 µs | 12.3 µs | 2.19x |
| | 1 MB | 217 µs | 300 µs | 1.38x |
| | 256 MB | 53.8 ms | 66.4 ms | 1.23x |
| GEMM (cuBLAS) | bfloat16 4096³ | 40.3 TFLOPS | 41.5 TFLOPS | 0.97x |
| | A decode step's GEMVs (Qwen3-4B, bf16, one rank of two) | 10.0 ms | 10.6 ms | 1.06x |
| Tensor parallelism | Row-parallel layer, M = 512 / 2048 / 8192 | 1.05 / 3.94 / 15.04 ms | 1.48 / 5.48 / 21.27 ms | 1.39–1.41x |
| vLLM, Qwen3-4B on 2 GPUs | Latency, batch 1 (512 + 128 tokens) | 1.70 s | 1.69 s | 0.99x |
| | Throughput, 200 requests | 2.89 requests/s | 2.27 requests/s | 1.27x |

LibreCUDA is ahead in collectives and what builds on them (tensor-parallel layers, batched serving), in peer copies
from the GPU whose own engines write slowly and in launch overhead; even in streaming transfers, memory bandwidth,
managed memory, dense GEMMs and single-request decoding; behind in synchronous round trips (about 1 µs), graph replay
overhead (0.16 µs per kernel) and round trips from several threads at once. The tensor-parallel row is the faster of
LibreCUDA's `ncclAllReduce` and its device API kernel for each size.

Measured with `benchmarks/bench.cu` on an RTX 5060 at PCIe Gen 5 x4 (the test system's GPU 0), LibreCUDA against
NVIDIA's CUDA 13.2 runtime (13.2.86) and driver 610.57.04 on the same GPU and boot:

| Benchmark | LibreCUDA | NVIDIA |
|---|---|---|
| Async kernel launch (empty kernel / 5 arguments) | 1.14 / 1.14 µs | 1.28 / 1.35 µs |
| Launch + `cudaDeviceSynchronize` round trip | 5.59 µs | 4.67 µs |
| GPU time per empty kernel | 1.14 µs | 1.27 µs |
| 4 byte `cudaMemcpy` latency (H2D pinned / D2H pageable) | 4.95 / 4.73 µs | 3.33 / 4.23 µs |
| H2D / D2H pinned, 1 MiB | 13.3 / 13.4 GB/s | 12.8 / 12.9 GB/s |
| H2D / D2H pinned, 256 MiB | 14.5 / 14.3 GB/s | 14.5 / 14.3 GB/s |
| H2D / D2H pageable, 256 MiB | 14.2 / 11.1 GB/s | 14.3 / 13.1 GB/s |
| H2D ∥ D2H on two streams | 21.6 GB/s | 21.6 GB/s |
| Device memory, triad kernel | 392 GB/s | 387 GB/s |
| Device memory, copy kernel (read + write) | 388 GB/s | 380 GB/s |
| `cudaMemcpy` device to device | 387 GB/s | 389 GB/s |
| `cudaMemset` | 420 GB/s | 426 GB/s |
| `cudaMalloc` / `cudaFree` of 100 × 4 KiB (first use) | 6.5 / 0.2 µs | 7.2 / 1.4 µs |
| `cudaMalloc` / `cudaFree` of 100 × 1 MiB | 16 / 14 µs | 18 / 15 µs |
| `cudaMalloc` / `cudaFree` of 100 × 64 MiB | 33 / 207 µs | 37 / 203 µs |
| SGEMM 4096³ (simple register tiling, the same kernel) | 6.96 TFLOP/s | 6.91 TFLOP/s |

On GPU 1 (Gen 4 x4, behind the chipset) host transfers reach 7.05 / 7.15 GB/s pinned and 9.3 GB/s in both directions
together on both runtimes; there the launch + synchronize round trip takes 6.63 µs (NVIDIA: 5.79 µs) and the 4 byte
copies 6.45 / 5.81 µs (4.86 / 5.22 µs). LibreCUDA's synchronous round trips are about 1 µs slower than NVIDIA's; everything that
streams is within a few percent, and launches and the GPU time per kernel are about 10% shorter. Allocations of up to 1 MiB come two or more from one 2 MiB chunk on
both runtimes (above, each is an allocation of its own from the kernel driver: about 37 / 32 µs, NVIDIA 35 / 30 µs).

Synchronization and copies across devices and host threads (`benchmarks/multi_gpu_sync.cu`, 2000 iterations, the two
RTX 5060: GPU 0 at Gen 5 x4, GPU 1 at Gen 4 x4 behind the chipset; median of 3 runs):

| Benchmark | LibreCUDA | NVIDIA |
|---|---|---|
| Launch + `cudaStreamSynchronize`, one device | 5.4 µs | 5.3 µs |
| Launch on both devices + `cudaDeviceSynchronize` both | 8.4 µs | 7.2 µs |
| GPU to GPU ping-pong through `cudaStreamWaitEvent`, per hop | 5.6 µs | 4.9 µs |
| Launch on device 0 + `cudaEventSynchronize` with device 1 current | 6.1 µs | 4.7 µs |
| `cudaMemcpyPeer` 4 B, peer access / none | 4.7 / 14.9 µs | 3.2 / 13 µs |
| `cudaMemcpyPeerAsync` 4 B + sync, peer access / none | 4.5 / 13.8 µs | 4.1 / 10.4 µs |
| `cudaMemcpyPeer` 64 MiB with peer access, GPU 0 → 1 / 1 → 0 | 7.04 / 7.14 GB/s | 2.15 / 7.14 GB/s |
| `cudaMemcpyPeer` 64 MiB without peer access, GPU 0 → 1 / 1 → 0 | 6.88 / 7.03 GB/s | 6.75 / 7.07 GB/s |
| Launch + sync round trips, 1 / 2 / 4 threads on one device | 181 / 292 / 404 per ms | 181 / 403 / 924 per ms |
| Launch + sync round trips, 2 / 4 threads over two devices | 283 / 433 per ms | 346 / 662 per ms |

GPU 0's copy engine writes into GPU 1 at about 2.2 GB/s and reads from it at 1.65 GB/s, GPU 1's engine reaches
7.0 / 7.1 GB/s in both directions. NVIDIA's runtime copies with the source GPU's engine, judging by the bandwidth (2.15 GB/s from GPU 0 to 1 with either
device current),
LibreCUDA's with the faster one (GPU 1's both ways; LibreCUDA's NCCL uses it too). Without peer access both runtimes
overlap the two GPUs' halves of a copy through host memory. The 4 byte copies without peer access include the first
copy's setup (copy streams and staging buffers, 0.75 ms, about 1.5 µs per copy here); afterwards they take 12.5 µs
synchronous, 14.2 µs with `cudaMemcpyPeerAsync` + sync (NVIDIA: 10.8 / 13.0 µs). Synchronous round trips from several
threads scale worse: 404 per ms from 4 threads against NVIDIA's 924.

Managed memory, 64 MiB on one RTX 5060 (`benchmarks/managed.cu`: a `cudaMallocManaged` buffer, simple CPU loops and a
kernel adding 1):

| | LibreCUDA (UVM) | NVIDIA | LibreCUDA (`LIBRECUDA_MANAGED_BAR1=1`) |
|---|---|---|---|
| CPU write, first touch / again | 6.1 / 23.0 GB/s | 6.2 / 22.3 GB/s | 11.3 / 11.4 GB/s |
| CPU read | 12.9 GB/s | 12.9 GB/s | 0.01 GB/s |
| First kernel after CPU writes (migration) | 10.8 ms | 10.3 ms | 0.66 ms |
| Kernel again | 0.38 ms | 0.40 ms | 0.38 ms |
| CPU read after the GPU wrote | 4.3 GB/s | 4.3 GB/s | 0.06 GB/s |

CUDA graphs (`benchmarks/graphs.cu`: `cudaGraphLaunch` back to back on one RTX 5060; empty kernels, and kernels of
about 20k dependent FMAs in one block):

| Graph | LibreCUDA | NVIDIA |
|---|---|---|
| Chain of 100 empty kernels | 57 µs | 41 µs |
| 4 branches × 25 empty kernels | 28 µs | 19 µs |
| 4 branches × 25 heavier kernels | 826 µs | 818 µs |

A replay costs LibreCUDA about 0.57 µs per kernel against NVIDIA's 0.41 µs (the chain took 112 µs before the kernels
of a prebuilt segment were chained on the GPU); with work in the kernels the difference disappears.

### PyTorch eager operations

`benchmarks/pytorch_eager.py` with torch 2.11.0+cu130 on 02 (best of 3; NVIDIA: torch's own CUDA libraries):

| Operation | LibreCUDA | NVIDIA |
|---|---|---|
| `x.add_(1)`, 1024 floats | 2.4 µs | 4.0 µs |
| `x.sum()`, 1024 floats | 4.4 µs | 3.9 µs |
| `x.add_(1)`, 64 MiB | 344 µs | 350 µs |
| `x.sum()`, 64 MiB | 163 µs | 169 µs |
| `x[0].item()` (copy to the host and synchronize) | 16.8 µs | 9.4 µs |

### GEMM (cuBLAS)

`benchmarks/gemm.cu` (`cublasGemmStridedBatchedEx`, A not transposed, B transposed; best of 3 runs of 10):

| Operation | LibreCUDA | NVIDIA cuBLAS |
|---|---|---|
| half 4096³ | 40.3 TFLOPS | 41.6 TFLOPS |
| bfloat16 4096³ | 40.3 TFLOPS | 41.5 TFLOPS |
| half → float 4096³ | 40.2 TFLOPS | 41.5 TFLOPS |
| float 4096³ (exact) | 13.5 TFLOPS | 14.6 TFLOPS |
| float 4096³ (TF32) | 20.2 TFLOPS | 20.1 TFLOPS |
| double 1024³ | 0.23 TFLOPS | 0.27 TFLOPS |
| half 512³ | 15.5 TFLOPS | 18.2 TFLOPS |
| half, 64 × 128×128×64 batched | 9.2 TFLOPS | 13.9 TFLOPS |
| half 1001³ (odd sizes: repacked) | 14.0 TFLOPS | 17.3 TFLOPS |
| half 50257×512×768 (language model head) | 20.9 TFLOPS | 20.2 TFLOPS |

FP8 matmuls (`torch._scaled_mm`, e4m3, per-tensor scales, bf16 output; best of 3 runs of 20):

| Operation | LibreCUDA | NVIDIA cuBLASLt |
|---|---|---|
| 4096³ | 139 TFLOPS | 144 TFLOPS |
| 8192³ | 136 TFLOPS | 140 TFLOPS |
| 512 x 9728 x 2560 (a prefill's MLP GEMM) | 205 µs | 270 µs |
| 512 x 2560 x 2560 | 56 µs | 76 µs |
| 64 x 4096 x 4096 | 28 µs | 28 µs |
| 16 x 4096 x 2560 | 16 µs | 17 µs |

That is 3.4 times LibreCUDA's bf16 rate. Before the block scaled instruction (sm_89's `mma.sync`) they ran at 79
TFLOPS at 4096³, exactly twice the bf16 rate. 128 x 128 tiles in groups of 8 (L2 reuse: 8192³ took 100 TFLOPS
without) except when they would fill the GPU less than twice over; unlike the 16 bit kernels, the FP8 ones keep the
large tiles where B fits in L2 (4096³ on 64 x 64 tiles: 110 TFLOPS).

Training a 4 layer transformer encoder (d = 256, feed-forward 1024, 32 × 64 tokens, bfloat16 autocast, AdamW) takes
4.2 ms per step on LibreCUDA and 3.5 ms on NVIDIA's CUDA. LibreCUDA's cuBLAS on NVIDIA's runtime takes 3.3 ms: its
kernels take as long as NVIDIA's cuBLAS kernels (both are CUTLASS kernels with 64 x 64 tiles for these sizes), and the
rest of the difference is the cost of the about 280 kernel launches per step through the test system's 2.5 GT/s link.

Contexts boost the GPU to its highest performance state while they exist (`NV2080_CTRL_CMD_PERF_BOOST`, as CUDA
contexts do), so work runs at full clocks from the start instead of after the GPU's own ~100 ms ramp (a 4096³ half
GEMM at the start of a process: 40.6 instead of 24.8 TFLOPS).

### Convolutions (cuDNN)

PyTorch with `torch.backends.cudnn.benchmark = True`, batch 32 (`fwd+bwd`: forward, input and weight gradients):

| Convolution | LibreCUDA | NVIDIA cuDNN |
|---|---|---|
| half 64->64 3x3, 56x56, forward | 0.49 ms | 0.33 ms |
| half 64->64 3x3, 56x56, fwd+bwd | 1.68 ms | 1.15 ms |
| half 256->64 1x1, 56x56, fwd+bwd | 1.02 ms | 0.95 ms |
| half 128->128 3x3, 28x28, fwd+bwd | 1.21 ms | 0.84 ms |
| float 64->64 3x3, 56x56, fwd+bwd (TF32) | 2.43 ms | 2.27 ms |
| float 128->128 3x3, 28x28, fwd+bwd (TF32) | 2.20 ms | 1.48 ms |
| small ResNet training step (half autocast, 32 x 224x224) | 16.4 ms | 12.1 ms |

Convolutions of NCHW tensors spend about a third of their time copying the tensors into and out of the kernels'
channels-last layout. Exact float convolutions (`allow_tf32 = False`) are im2col and float GEMMs, about 5 times
slower than with TF32.

### FFT (cuFFT)

| Transform (PyTorch) | LibreCUDA | NVIDIA cuFFT |
|---|---|---|
| complex float, 16 x 2^20 | 6.05 ms | 1.43 ms |
| real float, 256 x 4096 | 0.088 ms | 0.011 ms |
| complex float, 64 x 1000 | 0.020 ms | 0.006 ms |
| complex float, 64 x 997 (prime: Bluestein) | 0.095 ms | 0.010 ms |
| real 2D float, 32 x 3 x 256 x 256 | 0.79 ms | 0.26 ms |

(Small transforms of NVIDIA's cuFFT run from the L2 cache in this benchmark, which repeats them on the same data.)

### Sparse matrices (cuSPARSE)

PyTorch on a 20000 x 20000 float CSR matrix:

| Operation | LibreCUDA | NVIDIA cuSPARSE |
|---|---|---|
| SpMV (1M nonzeros) | 0.027 ms | 0.051 ms |
| SpMM (1M nonzeros, 64 columns) | 0.39 ms | 0.25 ms |
| SpGEMM (100k nonzeros each) | 0.37 ms | 0.26 ms |

### Random numbers (cuRAND)

256 MB of uniform floats per call, RTX 5060:

| Generator | LibreCUDA | NVIDIA cuRAND |
|---|---|---|
| XORWOW | 422 GB/s | 381 GB/s |
| MRG32k3a | 188 GB/s | 191 GB/s |
| MTGP32 | 267 GB/s | 266 GB/s |
| MT19937 | 87 GB/s | 147 GB/s |

### All-reduce (NCCL)

Float sums between two processes on the test system's two RTX 5060 (PCIe Gen5 x4 and Gen4 x4, peer access), from C
(`benchmarks/nccl_allreduce.cu`: all-reduces back to back on one stream), time per all-reduce and algorithm bandwidth.
LibreCUDA's NCCL on LibreCUDA and on NVIDIA's runtime (only `libnccl.so.2` taken from LibreCUDA), NVIDIA's NCCL 2.28
with its default (P2P) and with its fastest setting for large messages on this system (shared memory, 4 channels:
`NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=4`; its copy engine modes `NCCL_P2P_USE_CUDA_MEMCPY` and
`NCCL_SHM_USE_CUDA_MEMCPY` hang here); the last column compares LibreCUDA with the faster of NVIDIA's two:

| Size | LibreCUDA | LibreCUDA's NCCL, NVIDIA's runtime | NVIDIA NCCL (P2P) | NVIDIA NCCL (shared memory) | LibreCUDA vs NVIDIA's best |
|---|---|---|---|---|---|
| 256 B | 3.26 µs | 2.67 µs | 6.15 µs | 8.22 µs | 1.89x |
| 4 KB | 5.62 µs | 5.22 µs | 12.3 µs | 19.4 µs | 2.19x |
| 64 KB | 20.1 µs | 19.6 µs | 31.9 µs | 35.5 µs | 1.59x |
| 256 KB | 64.4 µs | 60.8 µs | 117 µs | 87.3 µs | 1.36x |
| 1 MB | 217 µs (4.83 GB/s) | 217 µs (4.83 GB/s) | 468 µs (2.24 GB/s) | 300 µs (3.49 GB/s) | 1.38x |
| 16 MB | 3.37 ms (4.97 GB/s) | 3.72 ms (4.51 GB/s) | 7.51 ms (2.23 GB/s) | 4.53 ms (3.70 GB/s) | 1.34x |
| 256 MB | 53.8 ms (4.99 GB/s) | 59.4 ms (4.52 GB/s) | 120 ms (2.24 GB/s) | 66.4 ms (4.04 GB/s) | 1.23x |

Each range has its own algorithm: up to 5 KB LL stores (6.6 µs at 4 KB and 5.7 µs at 256 bytes before them), above
16 KB and up to 1 MB the driven exchange (64 KB: 30.0 µs before it, 1 MB: 328 µs), above 1 MB the exchange with the
copy engine. The link of the GPU on the Gen4 slot carries about 11 GB/s in both directions together, depending on the
engines: the GPU's copy engine sends 7.1 GB/s and its kernels read the other GPU's memory at 6.4 GB/s, but writes
into it from the other GPU reach 2.3 GB/s (kernels) or 2.2 GB/s (copy engine). The exchange all-reduce sends 4.9 GB/s
with the copy engine while the kernel reads 6.3 GB/s; two kernels (the rings, NCCL's P2P) share about 7.4 GB/s. The
driven exchange moves 64 KB each way in 13 µs (9.8 GB/s across both directions); the rest is the chain of the staging
copy, two flags crossing the link and the other rank's combine, which below about 20 KB costs more than the slow
direction's pushes. Through PyTorch (`dist.all_reduce`) an all-reduce takes about 4 µs longer (8 KB: 11.2 µs on
LibreCUDA against 7.6 µs from C; 11.5 µs with LibreCUDA's NCCL on NVIDIA's runtime, 13.9 µs with NVIDIA's NCCL).

Captured into a CUDA graph (20 all-reduces per graph, replayed), the kernels wait and signal themselves and the
uploads take three nodes per chunk: 0.007 ms at 4 KB, 0.096 ms at 256 KB, 0.36 ms at 1 MB, 3.71 ms at 16 MB and
55.7 ms at 256 MB on LibreCUDA (0.006, 0.093, 0.33, 3.77 and 56.8 ms on NVIDIA's runtime).

### Tensor-parallel GEMM

A row-parallel layer on the two GPUs: Y = X @ W in bf16, M x 8192 by 8192 x 4096, each rank multiplying its half of K,
then an all-reduce of the M x 4096 partial results, through PyTorch.

Decode-sized layers (`benchmarks/tp_small.py`), GEMM then all-reduce, time per layer (in parentheses the all-reduce
alone; NVIDIA's NCCL: the faster of its two settings for each size):

| M (all-reduce) | GEMM alone (LibreCUDA / cuBLAS) | LibreCUDA | LibreCUDA's NCCL, NVIDIA's runtime | NVIDIA NCCL | LibreCUDA vs NVIDIA |
|---|---|---|---|---|---|
| 1 (8 KB) | 99.9 / 98.2 µs | 109.7 µs (11.2) | 101.9 µs (11.5) | 112.2 µs (13.9) | 1.02x |
| 8 (64 KB) | 100.2 / 93.2 µs | 121.5 µs (20.8) | 115.0 µs (19.8) | 129.1 µs (31.2) | 1.06x |
| 32 (256 KB) | 100.9 / 89.1 µs | 167.9 µs (64.9) | 158.6 µs (61.0) | 186.7 µs (89.3) | 1.11x |
| 128 (1 MB) | 136.3 / 120.8 µs | 357.7 µs (217.0) | 341.2 µs (217.6) | 436.1 µs (307.3) | 1.22x |

At these sizes the GEMM dominates: each rank reads 32 MB of weights, at least 75 µs at the GPU's 448 GB/s, and
LibreCUDA's GEMM (about 75% of that) is 2 to 13% slower than cuBLAS. Captured in CUDA graphs the layers take 109, 127,
200 and 488 µs on LibreCUDA (110, 118, 184 and 408 µs with LibreCUDA's NCCL on NVIDIA's runtime): captured all-reduces
use the flag protocol, not the LL stores or driven exchanges.

Larger layers (`benchmarks/tp_gemm.py`): the best of GEMM then all-reduce and of chunked overlap (the all-reduce of a
row block runs while the next block is multiplied, in 4 or 8 blocks). Time per layer:

| M (output) | GEMM alone | LibreCUDA | LibreCUDA's NCCL, NVIDIA's runtime | NVIDIA NCCL (P2P) | NVIDIA NCCL (shared memory, 4 channels) |
|---|---|---|---|---|---|
| 512 (4 MB) | 0.44 ms | 1.05 ms | 1.06 ms | 2.15 ms | 1.48 ms |
| 2048 (16 MB) | 1.68 ms | 4.90 ms | 4.44 ms | 8.39 ms | 5.48 ms |
| 8192 (64 MB) | 6.66 ms | 16.75 ms | 16.23 ms | 32.67 ms | 21.27 ms |

With LibreCUDA's NCCL the layer is 1.23x to 1.40x faster than with NVIDIA's NCCL at its fastest setting, and 1.9x
to 2.0x faster than at its default (on NVIDIA's runtime; on LibreCUDA, whose GEMMs are slower, 1.12x to 1.41x). At
M = 512 the row blocks' all-reduces (1 MB) are driven exchanges. The overlap needs the transport's high priority
streams: without them the communication kernels wait behind the GEMM's pending blocks (17.4 instead of 20.2 ms at
M = 8192 on NVIDIA's runtime).

The same layer with the all-reduce written with the device API (`benchmarks/tp_gemm_device.cu`, plain CUDA): cuBLAS
writes the partial results straight into a symmetric window and an 8 block kernel per row block reduces them through
LSA pointers, in place, between two LSA barriers (reduce-scatter and all-gather in one kernel). Because this system's
GPU 0 moves data over its link slowly when its SMs issue the transfers (about 2 GB/s against 6 to 7 GB/s for GPU 1),
the kernel lets GPU 1 sum all elements (`TP_SHARE0=0`: it reads GPU 0's partial results and writes the sums back);
with an even split the layer takes 1.5x to 1.8x as long. Time per layer, chunked overlap in 4 blocks:

| M (output) | GEMM alone | device API, LibreCUDA | device API, NVIDIA's runtime | LibreCUDA's `ncclAllReduce` (same program, LibreCUDA) | NVIDIA NCCL `ncclAllReduce` (default) |
|---|---|---|---|---|---|
| 512 (4 MB) | 0.44 ms | 1.13 ms | 1.11 ms | 1.05 ms | 1.82 ms |
| 2048 (16 MB) | 1.67 ms | 3.94 ms | 3.87 ms | 4.94 ms | 6.44 ms |
| 8192 (64 MB) | 6.63 ms | 15.04 ms | 14.81 ms | 17.09 ms | 24.34 ms |

From M = 2048 the device API layer is the fastest way to run it on this system: 1.39x and 1.41x faster than with
NVIDIA's NCCL at its fastest setting (5.48 and 21.27 ms above). The device API kernels give the same times with
NVIDIA's NCCL (its window implementation) as with LibreCUDA's. At M = 512 LibreCUDA's `ncclAllReduce` is faster: its
1 MB row blocks are driven exchanges, which write their own data while they read, while the device API kernel's
writes (the sums) wait for its reads. A plain 4 KB all-reduce kernel (two barriers around the reads) takes 11.9 µs,
slower than `ncclAllReduce`'s 5.6 µs: the barriers' flag round trips cross the link twice.

### vLLM

vLLM 0.27.1 serving Qwen3-4B (bf16) with tensor parallelism 2 on the two RTX 5060 (`vllm bench`, 0.80 of the memory,
512 input and 128 output tokens per request). The second column runs every all-reduce through LibreCUDA's NCCL
(`--disable-custom-all-reduce`), the third is NVIDIA's CUDA with only LibreCUDA's NCCL:

| Benchmark | LibreCUDA | LibreCUDA, NCCL all-reduces | NVIDIA's CUDA, LibreCUDA's NCCL | NVIDIA's CUDA and NCCL | LibreCUDA vs NVIDIA |
|---|---|---|---|---|---|
| Latency, batch 1 | 1.70 s per request | 1.65 s | 1.69 s | 1.69 s | 1.00x / 1.02x |
| Throughput, 200 requests | 2.95 requests/s (3400 tokens/s) | 3.08 requests/s (3549 tokens/s) | 2.65 requests/s | 2.27 requests/s (2616 tokens/s) | 1.30x / 1.36x |

Batched requests run 1.30x faster (1.36x with NCCL all-reduces): LibreCUDA's NCCL alone makes them 1.17x faster (their
large prefill all-reduces; the small ones are vLLM's own custom all-reduce kernels on every stack), the rest of the
stack another 1.11x. A single request decodes as fast as on NVIDIA's CUDA (12.2 ms per token against 12.0 ms): each step
reads all weights, one matrix times one vector per layer, which cuBLAS / cuBLASLt run with a dedicated kernel for up to
4 columns (a decode step's GEMVs read 401 GB/s, cuBLAS 378 GB/s). With the tensor core GEMM kernels instead decoding was
14% slower (331 GB/s: their tiles waste most of their width on one column). The remaining 0.15 ms per token is vLLM's
custom all-reduce, 10.7 µs per call on LibreCUDA against 8.4 µs (72 per step).

llama.cpp (build 3f03b17, CUDA backend, the same weights as a bf16 GGUF, both GPUs, flash attention) for comparison,
`llama-bench` (512-token prompt, 128 generated) and `llama-batched-bench` (32 sequences of 512 + 128 tokens):

| llama.cpp | NVIDIA's CUDA | LibreCUDA |
|---|---|---|
| Tensor split: prompt / generation | 4158 / 82.5 tokens/s (1.67 s per request) | 4091 / 79.4 tokens/s (1.74 s) |
| Layer split: prompt / generation | 4288 / 49.9 tokens/s | 4162 / 49.4 tokens/s |
| Throughput, 32 sequences | 2888 tokens/s | 2782 tokens/s (tensor), 2781 (layer) |

The same llama.cpp binary runs 1–4% slower on LibreCUDA. LibreCUDA's cuBLAS alone (on NVIDIA's runtime) costs its prompt
processing 0.9%. Its tensor-split decoding loses 4% in the attention part of each layer (about 13 kernels between two
all-reduces, 7.6 µs more per layer and GPU), consistent with each dependent kernel costing about 0.4 µs more GPU time
than on NVIDIA's runtime (see Kernel launch); the feed-forward part (4 kernels) loses 1 µs. vLLM on LibreCUDA is the
fastest of the four combinations, for single requests and batched ones.

### SGLang

SGLang 0.5.22 on the same model and GPUs (`sglang.benchmark.one_batch`: 512 input and 128 output tokens, batch 1;
`sglang.benchmark.offline_throughput`: 200 random requests of 512 + 128 tokens; 0.8 of the memory). On LibreCUDA every
all-reduce goes through LibreCUDA's NCCL (see SGLang above); on NVIDIA's CUDA with SGLang's custom all-reduce, or with
NVIDIA's NCCL (`--disable-custom-all-reduce`):

| Benchmark | LibreCUDA | NVIDIA's CUDA, custom all-reduce | NVIDIA's CUDA and NCCL |
|---|---|---|---|
| Prefill, 512 tokens | 0.111 s | 0.166 s | 0.140 s |
| Decode, per token | 11.9 ms | 12.4 ms | 12.9 ms |
| Latency, batch 1 | 1.63 s | 1.74 s | 1.78 s |
| Throughput, 200 requests | 5.96 requests/s (3816 tokens/s) | 4.39 requests/s (2810 tokens/s) | 4.59 requests/s (2938 tokens/s) |

## Limitations

- **cuBLAS / cuBLASLt:** no int8 matmuls, auxiliary epilogue outputs or bias gradients; FP8 matmuls take only
  per-tensor scales (no row-wise `OUTER_VEC_32F` or block scales: PyTorch's row-wise `_scaled_mm` on sm_120 doesn't
  reach cuBLASLt but runs its own CUTLASS kernels, which use TMA), no bias or activation with 8 bit output, and
  run at about half the speed of NVIDIA's (see Performance); epilogues of float output are a
  separate pass over the result. GEMM kernels have two tile sizes, so small matrices use the
  GPU less well than cuBLAS's kernels. The batched factorizations run one thread block per matrix, which suits the
  small matrices they are meant for.
- **cuDNN:** exact float, double, grouped and 3D convolutions are im2col and GEMMs, 5 to 10 times slower than
  cuDNN's kernels (see Performance); one engine / algorithm per convolution, no fused operation graphs (convolution with pointwise
  operations, attention, normalization graphs) in the backend API; no RNN projections; pooling forward only.
- **cuFFT:** transforms are 3 to 10 times slower than cuFFT (see Performance): each transform in shared memory loads
  all its lines before it computes and stores them, and lines longer than a block holds take a global memory round
  trip per large radix pass. Real transforms of odd lengths run as complex ones. No callbacks, multi-GPU (`cufftXt*`
  descriptors) or `cufftw`.
- **cuRAND:** host generators run on the GPU. Results whose arrangement cuRAND doesn't document (see cuRAND above)
  differ from NVIDIA's values. MT19937 is slower than cuRAND's: seeding takes 0.16 s (8191 jumps of 2^1000 results;
  cuRAND: 0.06 s) and it generates 87 GB/s of results (cuRAND: 147 GB/s).
- **NCCL:** the ranks must run on one machine (no network transports). The exchange all-reduce is for two ranks;
  more ranks use rings (pre-multiplied sums always do). The device transport needs peer access
  between all GPUs of a communicator (else the host transport runs, at the bandwidth of host copies, and its calls
  block the calling thread until the collective completed). Between more than two ranks, small messages are slower than NCCL's:
  every ring step is a kernel and a flag round trip across the link (the LL protocol is for two ranks). Non-blocking
  communicators complete their initialization immediately. The host transport's collectives can't be captured into
  graphs, and on LibreCUDA's runtime neither can the device transport's when ranks of several processes share a GPU
  (a running kernel isn't preempted for another process's, see "Several processes on one GPU": a graph's kernel
  waiting for a rank on its own GPU would keep that rank's kernels from running until the GPU kills a context with
  Xid 109, on every rank; measured with `tests/pytorch/nccl_ops.py` at 3 and 4 ranks on two GPUs. Such captures fail
  on all ranks). An uncaptured operation fails while the stream of
  the communicator's last small exchange (run directly on its caller's stream) is capturing. The device API has the
  LSA team only (no multimem or network: GIN barriers, signals and counters), window buffers must be at the same
  offset of their allocations on every rank, and the ranks of a window must be in separate threads or processes. No
  gather / scatter, `ncclCommRevoke` or growing communicators (`ncclCommGrow`); at most 64 ranks.
- **cuSPARSE:** no incomplete factorizations (`csrilu02`, `csric02`), tridiagonal / pentadiagonal solvers, ELL
  formats or JIT operations (SpMVOp, SpMMOp); SpGEMM only without transposes and with beta = 0, and it expands all
  products (memory grows with their number). SpMM is about 1.5 times slower than cuSPARSE (see Performance).
- **cuSOLVER:** Jacobi eigen and singular value decompositions are slower than cuSOLVER's divide and conquer
  algorithms for large matrices (they converge in about 10 sweeps of n^3 work each); SVDs are limited to
  min(m, n) <= 1024. `Xgeev` runs on the host. No LDL^T factorization.
- **Driver API:** only the primary context exists per device (`cuCtxCreate` returns another handle of it, without its
  own memory or streams). Virtual memory management maps whole allocations only, with file descriptor handles.
  Not supported: tensor maps (`cuTensorMapEncode*`: the descriptor format isn't public), green contexts, the library API
  (`cuLibrary*`, `cuKernel*`), CUDA arrays and texture objects through the driver API (the runtime's work), linking
  several inputs, and `cuGetExportTable`, so NVIDIA libraries with a static runtime (cuBLAS, cuDNN, ...) can't run.

- **Dynamic parallelism:** children run after their parent completed, launched by the host (see above), so a parent
  can't wait for or communicate with its children, a host launch of a kernel that launches kernels blocks until all of
  them completed, and device launches cost a host round trip. Device side copies of CUDA arrays and a few device
  runtime queries (attributes, error strings, occupancy) aren't supported.
- **Device side `malloc`:** freed blocks are only reused for allocations of the same power-of-two size class and are
  never merged, so a heap fragmented by many small blocks can't serve large allocations.
  Multi-device cooperative launches are not supported either.
- **Textures:** texture objects on linear memory (`tex1Dfetch`), pitched 2D memory and CUDA arrays (1D, 2D, 3D,
  layered, cubemaps, mipmapped; Blackwell only) and surface objects on arrays. Texture / surface references (removed in
  CUDA 12) aren't supported. Linear textures must be aligned to 512 bytes like in CUDA (the hardware needs 256).
- **Thread block clusters** are Blackwell only and can't be combined with cooperative launches. Clusters have up to 8
  blocks, or with `cudaFuncAttributeNonPortableClusterSizeAllowed` up to the SMs of a GPC (10 on the RTX 5060, like
  NVIDIA's runtime). Cluster occupancy (`cudaOccupancyMaxActiveClusters`) counts at most 2 resident blocks per SM, as
  NVIDIA's runtime does on sm_120.
- **Several devices:** copies between devices without peer access are staged through host memory; those with another
  device's CUDA arrays, and those between host memory and another device's memory, are synchronous. Peer atomics follow RM's capabilities (none over PCIe on the test system).
- **Managed memory:** prefetches are synchronous (they wait for the stream, then migrate); the attach flags of
  `cudaMallocManaged` and `cudaStreamAttachMemAsync` make no difference (all managed memory is accessible from every
  stream); discards keep the contents.
- **Host registration:** `cudaHostRegister` is accepted, but registered memory is still copied through staging buffers.
- **Threading:** threads only submit in parallel on streams other than the legacy default stream. On the test system,
  submissions are limited by PCIe anyway: each launch writes about 1.3 KiB through BAR1 and reads GPPut back over the
  2.5 GT/s x4 link, so 1, 2 and 4 threads all reach about 230 launches/ms.
- **Several processes on one GPU:** the GPU switches between processes only between kernels; a running kernel isn't
  preempted at the instruction level as on NVIDIA's runtime. Another process's work waits until the kernel ends (and
  that process's context creation does too), and a kernel still running about 4.3 s after another process submitted
  work is killed by the GPU's context switch timeout (Xid 109 `CTX SWITCH TIMEOUT`; the kernel fails with
  `cudaErrorLaunchFailure`). So kernels of two processes on one GPU can't wait for each other. Measured on an RTX 5060
  with kernels spinning on all SMs (24 blocks of 32 threads per SM, or one of 1024) and with a kernel waiting for a
  flag that another process's kernel sets in IPC memory; NVIDIA's runtime switches to the other process within 0.1 ms.
  Requesting CILP (instruction level) or CTA preemption for the channels
  (`NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE`, accepted by RM) doesn't change it: the context switch still times
  out. `cudaDevAttrComputePreemptionSupported` is 0.
- **Stream concurrency:** with more streams than compute channels (8 by default), streams share channels, and
  kernels of streams on the same channel run one after another.
- **Clock rates:** the memory clock (`cudaDevAttrMemoryClockRate`) is read from the GPU's highest performance level.
  The core clock (`cudaDevAttrClockRate`) is a fixed 2 GHz estimate: RM doesn't report the GPC clock's maximum through a
  documented control.
- **Faulty PCIe links corrupt data silently.** On the test system's degraded Gen 4 links, `tests/cudart/streams_events`
  failed about 1 in 30 runs with blocks of NaN in data that crossed PCIe, some coinciding with PCIe completion
  timeouts. With the link forced to 2.5 GT/s (no PCIe errors at all), about 190 runs passed, so this is attributed to
  the link, not LibreCUDA. Check `lspci -vv` (UESta/CESta) when host transfers return wrong data.
