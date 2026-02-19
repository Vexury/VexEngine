# Minimum compiler version checks
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "GCC >= 10.0 required for C++20. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Clang >= 10.0 required for C++20. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.28")
        message(FATAL_ERROR "MSVC >= 19.28 required for C++20. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

# Common settings interface library
add_library(vex_compiler_settings INTERFACE)

target_compile_features(vex_compiler_settings INTERFACE cxx_std_20)

target_compile_options(vex_compiler_settings INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /MP /permissive- /Zc:__cplusplus
    >
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall -Wextra -Wpedantic -fPIC
    >
)

target_compile_definitions(vex_compiler_settings INTERFACE
    $<$<CONFIG:Debug>:VEX_DEBUG>
    $<$<BOOL:${WIN32}>:VEX_PLATFORM_WINDOWS>
    $<$<PLATFORM_ID:Linux>:VEX_PLATFORM_LINUX>
    $<$<PLATFORM_ID:Darwin>:VEX_PLATFORM_MACOS>
)

# Set backend define based on VEX_BACKEND option
if(VEX_BACKEND STREQUAL "OpenGL")
    target_compile_definitions(vex_compiler_settings INTERFACE VEX_BACKEND_OPENGL)
elseif(VEX_BACKEND STREQUAL "Vulkan")
    target_compile_definitions(vex_compiler_settings INTERFACE VEX_BACKEND_VULKAN)
endif()
