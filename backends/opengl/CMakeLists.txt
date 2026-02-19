add_library(vex_opengl STATIC
    src/gl_context.cpp
    src/gl_shader.cpp
    src/gl_buffer.cpp
    src/gl_texture.cpp
    src/gl_framebuffer.cpp
    src/gl_mesh.cpp
    src/gl_skybox.cpp
    src/gl_gpu_raytracer.cpp
)

target_include_directories(vex_opengl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vex_opengl
    PUBLIC
        vex_core
        glad
        ${OPENGL_LIBRARIES}
)

set_target_properties(vex_opengl PROPERTIES FOLDER "Engine")
