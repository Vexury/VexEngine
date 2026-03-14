add_library(vex_core STATIC
    src/core/engine.cpp
    src/core/window.cpp
    src/core/input.cpp
    src/core/log.cpp
    src/core/camera.cpp
    src/core/stb_impl.cpp
    src/core/tiny_obj_impl.cpp
    src/core/tiny_gltf_impl.cpp
    src/scene/mesh_data.cpp
    src/scene/gltf_loader.cpp
    src/scene/primitives.cpp
    src/ui/ui_layer.cpp
    src/raytracing/bvh.cpp
    src/raytracing/cpu_raytracer.cpp
)

# Suppress warnings from the tinygltf implementation unit (third-party code)
if (MSVC)
    set_source_files_properties(src/core/tiny_gltf_impl.cpp PROPERTIES COMPILE_FLAGS "/W0")
else()
    set_source_files_properties(src/core/tiny_gltf_impl.cpp PROPERTIES COMPILE_FLAGS "-w")
endif()

target_include_directories(vex_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vex_core
    PUBLIC
        vex_compiler_settings
        glfw
        glm::glm
        imgui
        stb
)

set_target_properties(vex_core PROPERTIES FOLDER "Engine")
