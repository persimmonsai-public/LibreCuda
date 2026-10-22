# LibreCUDA's cuBLAS / cuBLASLt (libcublas.so.13, libcublasLt.so.13): the subset of the BLAS API that PyTorch and
# similar libraries use, on CUTLASS GEMM kernels (third_party/cutlass) and our own. Built with nvcc against the
# LibreCUDA runtime; only the public cublas_api.h / cublasLt.h headers are used for types.

if (NOT CMAKE_CUDA_COMPILER)
    find_program(BLAS_NVCC nvcc HINTS $ENV{CUDA_HOME}/bin $ENV{CUDA_PATH}/bin /usr/local/cuda/bin)
    if (BLAS_NVCC)
        set(CMAKE_CUDA_COMPILER ${BLAS_NVCC})
    endif ()
endif ()
include(CheckLanguage)
check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER OR NOT TARGET librecudart)
    message(WARNING "nvcc or the LibreCUDA runtime not found, not building LibreCUDA's cuBLAS")
    return()
endif ()
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/cutlass/include/cutlass/cutlass.h")
    message(WARNING "CUTLASS not found (git submodule update --init third_party/cutlass), not building LibreCUDA's cuBLAS")
    return()
endif ()
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # the kernels use sm_80 instructions; build for the GPUs LibreCUDA supports (override with -DCMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif ()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

set(BLAS_INCLUDE ${CMAKE_SOURCE_DIR}/third_party/cutlass/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# weak stubs for the functions of the public headers that aren't implemented
foreach (lib cublas cublasLt)
    if (lib STREQUAL "cublas")
        set(header cublas_api.h)
    else ()
        set(header cublasLt.h)
    endif ()
    find_file(${lib}_HEADER ${header} PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} NO_DEFAULT_PATH)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${lib}_stubs.cpp
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${${lib}_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${lib}_stubs.cpp
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${${lib}_HEADER})
endforeach ()

add_library(librecublasLt SHARED
        src/cublasLt.cpp src/gemm.cu src/cutlass_half.cu src/cutlass_bfloat16.cu src/cutlass_float.cu
        ${CMAKE_CURRENT_BINARY_DIR}/cublasLt_stubs.cpp)
add_library(librecublas SHARED src/cublas.cpp ${CMAKE_CURRENT_BINARY_DIR}/cublas_stubs.cpp)

foreach (target librecublasLt librecublas)
    target_include_directories(${target} PRIVATE ${BLAS_INCLUDE} src)
    # the LibreCUDA runtime (same soname as NVIDIA's), found next to the library
    set_target_properties(${target} PROPERTIES
            CUDA_RUNTIME_LIBRARY None
            VERSION 13
            SOVERSION 13
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    target_link_libraries(${target} PRIVATE librecudart)
endforeach ()
target_compile_options(librecublasLt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -diag-suppress=20012,177,550>)
set_target_properties(librecublasLt PROPERTIES OUTPUT_NAME cublasLt)
set_target_properties(librecublas PROPERTIES OUTPUT_NAME cublas)
target_link_libraries(librecublas PRIVATE librecublasLt)
target_link_options(librecublasLt PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcublasLt.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
target_link_options(librecublas PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcublas.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
