cmake_minimum_required(VERSION 3.16)

set(TINYEXR_DIR ${CMAKE_SOURCE_DIR}/external/tinyexr)

# miniz — ZIP/deflate decompression used by tinyexr
add_library(miniz STATIC ${TINYEXR_DIR}/deps/miniz/miniz.c)
target_include_directories(miniz PUBLIC ${TINYEXR_DIR}/deps/miniz)

# tinyexr — EXR image loading; tinyexr.cc defines TINYEXR_IMPLEMENTATION
add_library(tinyexr STATIC ${TINYEXR_DIR}/tinyexr.cc)
target_include_directories(tinyexr PUBLIC ${TINYEXR_DIR})
target_link_libraries(tinyexr PRIVATE miniz)

# Suppress warnings from third-party code
if(MSVC)
    target_compile_options(miniz   PRIVATE /W0)
    target_compile_options(tinyexr PRIVATE /W0)
else()
    target_compile_options(miniz   PRIVATE -w)
    target_compile_options(tinyexr PRIVATE -w)
endif()

set_target_properties(miniz tinyexr PROPERTIES FOLDER "External")
