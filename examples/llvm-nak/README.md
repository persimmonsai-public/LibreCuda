# LLVM → SPIR-V → Mesa NAK → LibreCUDA (prototype)

CUDA C kernels (`kernels.cu`) compiled without any NVIDIA compiler and without reverse-engineered encodings, run on
LibreCUDA and checked against numpy (`run_cubin.py`): Mesa's open NVIDIA compiler (NAK, MIT licensed, part of the NVK
driver) does the instruction selection, register allocation and encoding for SM_120.

```
kernels.cu ─clang -x hip --offload=spirv64─► LLVM IR ─opt / llc─► SPIR-V ─spirv_to_nir.c (Mesa's spirv_to_nir,
   NIR lowering, LibreCUDA's kernel ABI)─► NIR ─NAK (tinymesa)─► SASS ─nak_compile.py─► cubin ─► LibreCUDA
```

- **clang** compiles CUDA C in HIP mode for SPIR-V; `prelude_spirv.h` maps `threadIdx`, `blockIdx`, `blockDim`,
  `gridDim` and `__syncthreads` to OpenCL built-ins, which LLVM's SPIR-V backend turns into SPIR-V built-ins.
  `build.sh` drops HIP's compilation unit id variable (Mesa rejects initialized global variables) and lets `opt`
  infer address spaces before `llc` writes SPIR-V.
- **spirv_to_nir.c** (`libspirv_to_nir.so`) runs Mesa's SPIR-V translator in the OpenCL environment and lowers the
  result like rusticl (Mesa's OpenCL) does; then kernel parameters, the block and the grid size become loads from
  constant bank 0 at the offsets LibreCUDA writes (`c[0][0x380+]`, `c[0][0x360]`, `c[0][0x370]` on SM_120). The NIR
  goes to NAK serialized: the NAK library (tinymesa) has no SPIR-V translator, but exports `nir_deserialize`.
- **NAK** (tinymesa 25.2.7, through tinygrad's ctypes bindings) compiles the NIR as a compute shader.
- **nak_compile.py** wraps the code in a minimal cubin (the format of `tools/nak_cubin.py`, plus `.nv.shared` and the
  parameter bank attribute LibreCUDA needs), one per kernel.

## Running it

```bash
./setup.sh                    # Mesa 25.2.7's NIR + SPIR-V libraries and the shim, tinymesa, tinygrad (in ./work)
CLANG=/path/to/llvm-21/bin/clang LIBRECUDA=../../build/lib CUDA_VISIBLE_DEVICES=0 ./build.sh
```

## Status (LLVM 21.1.8, Mesa / tinymesa 25.2.7, RTX 5060), compared with [OpenPTXas](https://github.com/garrick247/openptxas)

OpenPTXas (a PTX assembler) was tried on the same kernels: clang's PTX, normalized around its parser and
immediate-sign bugs, with NVIDIA's ptxas hidden from it.

| Kernel | Mesa NAK | OpenPTXas (garrick99 5f707ea / garrick247 91ed550) | NVIDIA ptxas (reference) |
|---|---|---|---|
| `vector_add` | pass | pass / pass | pass |
| `saxpy` | pass | pass / pass | pass |
| `reduce_sum` (shared memory, barrier in a loop) | pass | illegal address / illegal address | pass |
| `stencil` (if / else, neighbour loads) | pass | misaligned address / wrong results | pass |

NAK's code: 19 registers for every kernel, 384 to 1072 bytes; no local memory.

| | Mesa NAK | OpenPTXas |
|---|---|---|
| Input | NIR (here from SPIR-V) | PTX |
| Front end work | SPIR-V translation and ABI lowering (this example) | PTX normalization around its bugs |
| Licence | MIT | Business Source License 1.1 (non-production until 2029) |
| Origin of the encodings | Mesa's NVK driver (on LibreCUDA's list of allowed sources) | reverse-engineered from ptxas output |
| Uses NVIDIA's ptxas if present | no | yes (copies sections, or whole cubins) |

## Limitations of the prototype

- Kernels only use what the four examples need: no device function calls across modules, atomics, warp intrinsics,
  math library functions (CUDA's `__expf` & co. would need libdevice equivalents; OpenCL's would need libclc), or
  dynamic shared memory.
- The register limit assumes 1024 threads per block (NAK derives it from the block size, which is only known at
  launch), so a kernel never has more than 64 registers.
- `spirv_to_nir.c` must be built from the Mesa version of the NAK library (serialized NIR is version specific); tinygrad
  pins tinymesa 25.2.7.2.
