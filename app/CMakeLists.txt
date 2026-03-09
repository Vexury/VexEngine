add_executable(vex_app
    src/main.cpp
    src/app.cpp
    src/scene.cpp
    src/scene_renderer.cpp
    src/editor_ui.cpp
    src/file_dialog.cpp
    src/denoiser.cpp
    src/command.cpp
)

if(VEX_BACKEND STREQUAL "OpenGL")
    target_link_libraries(vex_app PRIVATE vex_opengl)
elseif(VEX_BACKEND STREQUAL "Vulkan")
    target_link_libraries(vex_app PRIVATE vex_vulkan)
endif()

target_link_libraries(vex_app PRIVATE tinyexr nfd)

if(OpenImageDenoise_FOUND)
    target_link_libraries(vex_app PRIVATE OpenImageDenoise)
    target_compile_definitions(vex_app PRIVATE VEX_HAS_OIDN)
    set(_oidn_base "${CMAKE_SOURCE_DIR}/external/${OIDN_PLATFORM_DIR}")
    if(WIN32)
        set(_oidn_runtime_files
            "${_oidn_base}/bin/OpenImageDenoise.dll"
            "${_oidn_base}/bin/OpenImageDenoise_core.dll"
            "${_oidn_base}/bin/OpenImageDenoise_device_cpu.dll"
            "${_oidn_base}/bin/tbb12.dll"
        )
    elseif(APPLE)
        file(GLOB _oidn_runtime_files
            "${_oidn_base}/lib/libOpenImageDenoise*.dylib"
            "${_oidn_base}/lib/libtbb*.dylib"
        )
    else()
        file(GLOB _oidn_runtime_files
            "${_oidn_base}/lib/libOpenImageDenoise*.so*"
            "${_oidn_base}/lib/libtbb*.so*"
        )
    endif()
    if(_oidn_runtime_files)
        add_custom_command(TARGET vex_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_oidn_runtime_files}
                $<TARGET_FILE_DIR:vex_app>
            COMMENT "Copying OIDN runtime libraries"
        )
    endif()
endif()

# Compile Vulkan shaders to SPIR-V
if(VEX_BACKEND STREQUAL "Vulkan")
    include(${CMAKE_SOURCE_DIR}/cmake/modules/ShaderCompilation.cmake)
    compile_shaders(vex_app "${CMAKE_SOURCE_DIR}/shaders/vulkan")
endif()

# Copy assets and shaders to build output
add_custom_command(TARGET vex_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/VexAssetsCC0
        $<TARGET_FILE_DIR:vex_app>/VexAssetsCC0
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:vex_app>/shaders
    COMMENT "Copying assets and shaders to build directory"
)

if(MSVC)
    set_target_properties(vex_app PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT vex_app)
endif()

set_target_properties(vex_app PROPERTIES FOLDER "App")
