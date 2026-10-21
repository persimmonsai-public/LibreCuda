add_executable(
        test_indexing
        main.cpp
)
target_link_libraries(
        test_indexing
        PRIVATE
        driverapi
)

if (NOT LIBRECUDA_TEST_ARCH)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/write_float.cubin" ${CMAKE_BINARY_DIR}/tests/test_indexing/ COPYONLY)
endif ()