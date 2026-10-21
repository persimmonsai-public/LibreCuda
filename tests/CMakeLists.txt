# The repository ships sm_80 cubins for the tests. To test on other GPUs, set LIBRECUDA_TEST_ARCH (e.g. sm_120) and
# the test kernels are compiled for it with nvcc at build time instead.
set(LIBRECUDA_TEST_ARCH "" CACHE STRING "GPU architecture to compile the test kernels for (e.g. sm_120); empty = use shipped cubins")
add_subdirectory(write_float)
add_subdirectory(memcopy)
add_subdirectory(dynamic_shared_mem)
add_subdirectory(compute_chronological_consistency)
add_subdirectory(async_kernels)
add_subdirectory(dma_chronological_consistency)
add_subdirectory(kernel_struct_param)
add_subdirectory(indexing)
add_subdirectory(stream_events)
add_subdirectory(many_kernels_launch)
add_subdirectory(ctx_recreate)
if (LIBRECUDA_TEST_ARCH)
    find_program(NVCC_EXECUTABLE nvcc HINTS $ENV{CUDA_HOME}/bin /usr/local/cuda/bin REQUIRED)
    set(test_cubins)
    file(GLOB test_kernel_sources "${CMAKE_CURRENT_LIST_DIR}/*/*.cu")
    foreach (source ${test_kernel_sources})
        get_filename_component(test_dir ${source} DIRECTORY)
        get_filename_component(test_name ${test_dir} NAME)
        get_filename_component(kernel_name ${source} NAME_WE)
        if (test_name STREQUAL "cudart")
            continue() # CUDA runtime tests are built by tests/cudart/run_tests.sh
        endif ()
        set(cubin "${CMAKE_BINARY_DIR}/tests/${test_name}/${kernel_name}.cubin")
        add_custom_command(OUTPUT ${cubin}
                COMMAND ${NVCC_EXECUTABLE} -cubin -arch=${LIBRECUDA_TEST_ARCH} -O3 ${source} -o ${cubin}
                DEPENDS ${source}
                COMMENT "Compiling ${test_name}/${kernel_name}.cu for ${LIBRECUDA_TEST_ARCH}")
        list(APPEND test_cubins ${cubin})
    endforeach ()
    add_custom_target(test_kernels ALL DEPENDS ${test_cubins})
endif ()
