# LibreCUDA runtime: implementation of the CUDA runtime API (libcudart) on top of the LibreCUDA driver API.
# Only the public CUDA toolkit headers are required (for type definitions).

find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
        HINTS $ENV{CUDA_HOME}/include $ENV{CUDA_PATH}/include /usr/local/cuda/include)
find_library(ZSTD_LIBRARY zstd)

if (NOT CUDA_RUNTIME_INCLUDE_DIR OR NOT ZSTD_LIBRARY)
    message(WARNING "CUDA headers or libzstd not found, not building the LibreCUDA runtime")
    return()
endif ()

add_library(librecudart SHARED src/cudart.cpp src/cuda_driver.cpp)
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

# libcuda.so.1: the driver API of the runtime library, for programs and libraries that use or load libcuda (an ELF
# filter: its symbols are looked up in libcudart.so.13, so both share one set of contexts, streams and events). The
# filter's symbol table is generated from the driver API functions the runtime library exports.
set(LIBCUDA_STUBS ${CMAKE_CURRENT_BINARY_DIR}/libcuda_stubs.cpp)
add_custom_command(
        OUTPUT ${LIBCUDA_STUBS}
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_libcuda_stubs.sh $<TARGET_FILE:librecudart> ${LIBCUDA_STUBS}
        DEPENDS librecudart ${CMAKE_CURRENT_SOURCE_DIR}/gen_libcuda_stubs.sh)
add_library(librecuda_driver SHARED ${LIBCUDA_STUBS})
set_target_properties(librecuda_driver PROPERTIES
        OUTPUT_NAME cuda
        VERSION 1
        SOVERSION 1
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
target_link_options(librecuda_driver PRIVATE -Wl,--filter=libcudart.so.13 "-Wl,-rpath,$ORIGIN")
