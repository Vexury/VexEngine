add_executable(vex_tests
    test_main.cpp
    test_bvh.cpp
    test_bsdf.cpp
    test_primitives.cpp
    test_camera.cpp
    test_raytracer.cpp
)

target_include_directories(vex_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/external/doctest
)

target_link_libraries(vex_tests PRIVATE vex_core)
target_compile_features(vex_tests PRIVATE cxx_std_20)
set_target_properties(vex_tests PROPERTIES FOLDER "Tests")

add_test(NAME vex_tests COMMAND vex_tests)
