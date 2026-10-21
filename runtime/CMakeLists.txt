# LibreCUDA runtime: implementation of the CUDA runtime API (libcudart) on top of the LibreCUDA driver API.
# Only the public CUDA toolkit headers are required (for type definitions).

find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
        HINTS $ENV{CUDA_HOME}/include $ENV{CUDA_PATH}/include /usr/local/cuda/include)
find_library(ZSTD_LIBRARY zstd)

if (NOT CUDA_RUNTIME_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
    message(WARNING "CUDA headers or libzstd not found, not building the LibreCUDA runtime")
    return()
endif ()

add_library(librecudart SHARED src/cudart.cpp)
target_include_directories(librecudart PRIVATE ${CUDA_RUNTIME_INCLUDE_DIR})
target_link_libraries(librecudart PRIVATE driverapi ${ZSTD_LIBRARY})
# drop-in replacement for libcudart.so.13: same soname and symbol version
set_target_properties(librecudart PROPERTIES
        OUTPUT_NAME cudart
        VERSION 13
        SOVERSION 13
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
target_link_options(librecudart PRIVATE
        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libcudart.map
        -Wl,--no-undefined)
