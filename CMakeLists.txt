cmake_minimum_required(VERSION 3.16)
project(VexEngine VERSION 0.1.0 LANGUAGES CXX C)

# --- Options ---
set(VEX_BACKEND "OpenGL" CACHE STRING "Graphics backend (OpenGL or Vulkan)")
set_property(CACHE VEX_BACKEND PROPERTY STRINGS "OpenGL" "Vulkan")
option(VEX_BUILD_APP "Build the demo application" ON)

# --- Global settings ---
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Compiler settings ---
include(cmake/CompilerSettings.cmake)

# --- Third-party dependencies ---
# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# GLAD
add_subdirectory(external/glad)

# GLM
add_subdirectory(external/glm)

# ImGui
set(IMGUI_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
add_subdirectory(cmake/modules/imgui)

# STB
set(STB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/stb)
add_subdirectory(cmake/modules/stb)

# Vulkan-only dependencies
if(VEX_BACKEND STREQUAL "Vulkan")
    set(VOLK_STATIC_DEFINES "" CACHE STRING "" FORCE)
    add_subdirectory(external/volk)

    set(VK_BOOTSTRAP_WERROR OFF CACHE BOOL "" FORCE)
    set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/vk-bootstrap)

    add_subdirectory(external/VulkanMemoryAllocator)
endif()

# --- IDE folder organization ---
foreach(TARGET glfw glad glm imgui stb)
    if(TARGET ${TARGET})
        set_target_properties(${TARGET} PROPERTIES FOLDER "External")
    endif()
endforeach()

# --- Engine core ---
add_subdirectory(engine)

# --- Backend ---
if(VEX_BACKEND STREQUAL "OpenGL")
    find_package(OpenGL REQUIRED)
    add_subdirectory(backends/opengl)
elseif(VEX_BACKEND STREQUAL "Vulkan")
    add_subdirectory(backends/vulkan)
else()
    message(FATAL_ERROR "Unknown VEX_BACKEND: ${VEX_BACKEND}. Must be 'OpenGL' or 'Vulkan'.")
endif()

# --- Application ---
if(VEX_BUILD_APP)
    add_subdirectory(app)
endif()

# --- Summary ---
message(STATUS "=== VexEngine Configuration ===")
message(STATUS "  Backend:    ${VEX_BACKEND}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "===============================")
