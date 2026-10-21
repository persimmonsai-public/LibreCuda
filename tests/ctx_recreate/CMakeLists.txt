add_executable(
        test_ctx_recreate
        main.cpp
)
target_link_libraries(
        test_ctx_recreate
        PRIVATE
        driverapi
)

if (NOT LIBRECUDA_TEST_ARCH)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/fill.cubin" ${CMAKE_BINARY_DIR}/tests/ctx_recreate COPYONLY)
endif ()
