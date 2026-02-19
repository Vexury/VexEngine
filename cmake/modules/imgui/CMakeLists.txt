cmake_minimum_required(VERSION 3.16)
project(imgui CXX)

set(IMGUI_SOURCES
    ${IMGUI_SOURCE_DIR}/imgui.cpp
    ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
    ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
    ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
    ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)

if(VEX_BACKEND STREQUAL "OpenGL")
    list(APPEND IMGUI_SOURCES
        ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
elseif(VEX_BACKEND STREQUAL "Vulkan")
    list(APPEND IMGUI_SOURCES
        ${IMGUI_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )
endif()

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui
    PUBLIC
        ${IMGUI_SOURCE_DIR}
        ${IMGUI_SOURCE_DIR}/backends
)

target_link_libraries(imgui PUBLIC glfw)

if(VEX_BACKEND STREQUAL "OpenGL")
    target_link_libraries(imgui PUBLIC glad)
elseif(VEX_BACKEND STREQUAL "Vulkan")
    target_compile_definitions(imgui PUBLIC VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_USE_VOLK)
    target_link_libraries(imgui PUBLIC volk)
endif()
