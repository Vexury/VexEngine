add_executable(vex_app
    src/main.cpp
    src/app.cpp
    src/scene.cpp
    src/scene_renderer.cpp
    src/editor_ui.cpp
    src/file_dialog.cpp
)

if(VEX_BACKEND STREQUAL "OpenGL")
    target_link_libraries(vex_app PRIVATE vex_opengl)
elseif(VEX_BACKEND STREQUAL "Vulkan")
    target_link_libraries(vex_app PRIVATE vex_vulkan)
endif()

target_link_libraries(vex_app PRIVATE tinyexr)

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
