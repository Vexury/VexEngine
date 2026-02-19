add_library(vex_core STATIC
    src/core/engine.cpp
    src/core/window.cpp
    src/core/input.cpp
    src/core/log.cpp
    src/core/camera.cpp
    src/core/stb_impl.cpp
    src/core/tiny_obj_impl.cpp
    src/scene/mesh_data.cpp
    src/ui/ui_layer.cpp
    src/raytracing/bvh.cpp
    src/raytracing/cpu_raytracer.cpp
)

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
