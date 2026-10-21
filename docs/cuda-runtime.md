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
- **Memory.** Device memory, pinned host memory, managed memory (CPU-mapped device memory), copies in all directions,
  memset, 2D copies, symbol copies, pointer attributes and memory info are supported.
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
  to `librecuda_vprintf` (`driverapi/kernels/printf/printf.cu`, embedded per architecture by `compile_printf.sh`) and
  raises the calling kernel to that function's register count. The function copies the format string, the arguments
  and `%s` strings into a record in pinned host memory; stream, event and device synchronization (and program exit)
  print the completed records. Details that matter on the GPU:
  - the caller uses `CALL.ABS` with an absolute return address, so the function is compiled with `-rdc=true`, the only
    mode in which ptxas returns with `RET.ABS`;
  - inside the caller, constant bank 4 belongs to the caller, so the function uses no globals or string literals and
    finds its state through `%envreg8`/`%envreg9` in the driver area of constant bank 0;
  - the record counter is in device memory (no PCIe atomics needed); each record's size is written last.
  The buffer is 1 MiB by default (`cudaLimitPrintfFifoSize`); calls that don't fit are dropped with a warning.
- **Stream concurrency.** A context has up to 4 compute channels (`LIBRECUDA_COMPUTE_CHANNELS`), each in its own
  subcontext of one channel group, so kernels on different channels run concurrently. Streams are assigned to channels
  round-robin; a channel is created when the first stream is assigned to it, since idle channels in the group slow
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
- **Context teardown.** Destroying a context waits for its streams, unregisters its channels and VA space from UVM and
  frees its memory and RM objects. Streams that outlive their context become unusable but can still be destroyed.
  `cudaDeviceReset` uses this: it destroys the device's streams, events, modules and context, and the next API call
  creates a new context.
- **Robustness.** Channel errors (MMU faults, illegal instructions, ...) are detected through the channel error
  notifier and reported as `cudaErrorIllegalAddress` / `cudaErrorLaunchFailure` instead of hanging. An optional wait
  timeout turns GPU hangs into errors.

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
| `LIBRECUDA_COMPUTE_CHANNELS` | Number of compute channels streams are spread over (1–16, default 4). `1` runs all kernels one after another. |
| `LIBRECUDA_SMALL_ALLOC_POOL=0` | Disable the pool for small device allocations (every `cudaMalloc` becomes its own RM allocation). |
| `LIBRECUDA_ASYNC_COPY_ENGINES=0` | Use the graphics copy engine for all copies instead of two async copy engines (host→device and device→host). |
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

`tests/cudart` contains 17 self-checking CUDA programs:
- tiled and register-blocked SGEMM; reductions with warp shuffles and atomics,
- `__constant__`/`__device__` globals, function pointers and virtual functions,
- local memory and recursion; dynamic shared memory beyond 48 KiB,
- scan, radix and bitonic sort; n-body,
- streams and events; concurrent kernels on different streams; cooperative launches with `grid.sync()`; device side `printf`; memory operations; many small allocations; `cudaDeviceReset`; Thrust and CUB; PTX JIT.

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

## Limitations

- **Device side runtime features:** device side `malloc`/`free`, `assert` and dynamic parallelism are not supported. Thread
  block clusters, multi-device cooperative launches, CUDA graphs / stream capture, textures and surfaces are not
  supported either.
- **Separate compilation:** programs built with `-rdc=true` (`__cudaRegisterLinkedBinary`) are not supported.
- **Peer access:** not supported.
- **Managed memory:** device memory mapped into the CPU address space. CPU reads are uncached and slow, and there is no
  page migration.
- **Host functions:** `cudaLaunchHostFunc` and stream callbacks run synchronously when they are enqueued.
- **Host registration:** `cudaHostRegister` is accepted, but registered memory is still copied through staging buffers.
- **Threading:** the runtime serializes API calls with a global lock.
- **Stream concurrency:** with more streams than compute channels (4 by default), streams share channels, and
  kernels of streams on the same channel run one after another.
- **Clock rates:** the memory clock (`cudaDevAttrMemoryClockRate`) is read from the GPU's highest performance level.
  The core clock (`cudaDevAttrClockRate`) is a fixed 2 GHz estimate: RM doesn't report the GPC clock's maximum through a
  documented control.
- **Faulty PCIe links corrupt data silently.** On the test system's degraded Gen 4 links, `tests/cudart/streams_events`
  failed about 1 in 30 runs with blocks of NaN in data that crossed PCIe, some coinciding with PCIe completion
  timeouts. With the link forced to 2.5 GT/s (no PCIe errors at all), about 190 runs passed, so this is attributed to
  the link, not LibreCUDA. Check `lspci -vv` (UESta/CESta) when host transfers return wrong data.
