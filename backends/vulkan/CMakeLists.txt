add_library(vex_vulkan STATIC
    src/vk_context.cpp
    src/vk_init.cpp
    src/vk_shader.cpp
    src/vk_buffer.cpp
    src/vk_texture.cpp
    src/vk_framebuffer.cpp
    src/vk_mesh.cpp
    src/vk_skybox.cpp
    src/vk_gpu_raytracer.cpp
)

target_include_directories(vex_vulkan
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vex_vulkan
    PUBLIC
        vex_core
        volk
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
)

target_compile_definitions(vex_vulkan PUBLIC VK_NO_PROTOTYPES)

set_target_properties(vex_vulkan PROPERTIES FOLDER "Engine")
