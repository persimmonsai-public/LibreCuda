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

# weak stubs for the functions of the public headers that aren't implemented (macro, NOT_SUPPORTED status)
foreach (stub "cublas;cublas_api.h;CUBLASWINAPI;15" "cublasLt;cublasLt.h;CUBLASWINAPI;15"
              "cusolver;cusolverDn.h;CUSOLVERAPI;9" "cusparse;cusparse.h;CUSPARSEAPI;10")
    list(GET stub 0 lib)
    list(GET stub 1 header)
    list(GET stub 2 macro)
    list(GET stub 3 status)
    find_file(${lib}_HEADER ${header} PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} NO_DEFAULT_PATH)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${lib}_stubs.cpp
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${${lib}_HEADER} ${CMAKE_CURRENT_BINARY_DIR}/${lib}_stubs.cpp
                    ${macro} ${status}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${${lib}_HEADER})
endforeach ()

add_library(librecublasLt SHARED
        src/cublasLt.cpp src/gemm.cu src/cutlass_half.cu src/cutlass_bfloat16.cu src/cutlass_float.cu
        ${CMAKE_CURRENT_BINARY_DIR}/cublasLt_stubs.cpp)
add_library(librecublas SHARED src/cublas.cpp src/blas_routines.cu ${CMAKE_CURRENT_BINARY_DIR}/cublas_stubs.cpp)
add_library(librecusolver SHARED src/cusolver.cu ${CMAKE_CURRENT_BINARY_DIR}/cusolver_stubs.cpp)
add_library(librecusparse SHARED src/cusparse.cpp ${CMAKE_CURRENT_BINARY_DIR}/cusparse_stubs.cpp)

foreach (target librecublasLt librecublas librecusolver librecusparse)
    target_include_directories(${target} PRIVATE ${BLAS_INCLUDE} src)
    # the LibreCUDA runtime (same soname as NVIDIA's), found next to the library
    set_target_properties(${target} PROPERTIES
            CUDA_RUNTIME_LIBRARY None
            VERSION 13
            SOVERSION 13
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    target_link_libraries(${target} PRIVATE librecudart)
endforeach ()
foreach (target librecublasLt librecublas librecusolver)
    target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -diag-suppress=20012,177,550>)
endforeach ()
set_target_properties(librecublasLt PROPERTIES OUTPUT_NAME cublasLt)
set_target_properties(librecublas PROPERTIES OUTPUT_NAME cublas)
target_link_libraries(librecublas PRIVATE librecublasLt)
# libcusolver.so.12 (cuSOLVER of CUDA 13 keeps soname 12)
set_target_properties(librecusolver PROPERTIES OUTPUT_NAME cusolver VERSION 12 SOVERSION 12)
target_link_libraries(librecusolver PRIVATE librecublas librecublasLt)
target_link_options(librecusolver PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcusolver.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
target_link_options(librecublasLt PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcublasLt.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
target_link_options(librecublas PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcublas.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
# libcusparse.so.12: handles only (see src/cusparse.cpp)
set_target_properties(librecusparse PROPERTIES OUTPUT_NAME cusparse VERSION 12 SOVERSION 12)
target_link_options(librecusparse PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcusparse.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")

# libcudnn.so.9: convolutions (legacy and backend graph API), batch normalization, pooling, spatial transformer. Needs
# NVIDIA's public cuDNN headers (for the types): CUDNN_INCLUDE_DIR, or the nvidia-cudnn pip package.
file(GLOB CUDNN_HINTS $ENV{HOME}/.local/lib/python3*/site-packages/nvidia/cudnn/include
        /usr/lib/python3*/site-packages/nvidia/cudnn/include /usr/local/lib/python3*/site-packages/nvidia/cudnn/include
        /usr/lib64/python3*/site-packages/nvidia/cudnn/include)
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS $ENV{CUDNN_INCLUDE_DIR} ${CUDNN_HINTS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if (NOT CUDNN_INCLUDE_DIR)
    message(WARNING "cuDNN headers not found (set CUDNN_INCLUDE_DIR), not building LibreCUDA's cuDNN")
    return()
endif ()
set(CUDNN_STUBS)
foreach (header cudnn_graph_v9.h cudnn_ops_v9.h cudnn_cnn_v9.h cudnn_adv_v9.h)
    set(stub ${CMAKE_CURRENT_BINARY_DIR}/stubs_${header}.cpp)
    add_custom_command(
            OUTPUT ${stub}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${CUDNN_INCLUDE_DIR}/${header} ${stub} CUDNNWINAPI 3000
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_stubs.sh ${CUDNN_INCLUDE_DIR}/${header})
    list(APPEND CUDNN_STUBS ${stub})
endforeach ()
add_library(librecudnn SHARED src/cudnn_legacy.cpp src/cudnn_backend.cpp src/cudnn_conv.cu src/cudnn_ops.cu
        src/cudnn_rnn.cu
        ${CUDNN_STUBS})
target_include_directories(librecudnn PRIVATE ${CUDNN_INCLUDE_DIR} ${BLAS_INCLUDE} src)
set_target_properties(librecudnn PROPERTIES
        OUTPUT_NAME cudnn
        CUDA_RUNTIME_LIBRARY None
        VERSION 9
        SOVERSION 9
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
target_compile_options(librecudnn PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -diag-suppress=1444,20012 -Xcompiler=-Wno-deprecated-declarations>)
target_link_libraries(librecudnn PRIVATE librecudart librecublasLt)
target_link_options(librecudnn PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcudnn.map"
        -Wl,--no-undefined "-Wl,-rpath,$ORIGIN")
