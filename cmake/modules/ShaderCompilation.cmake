# Compile GLSL shaders to SPIR-V using glslc from the Vulkan SDK
# Outputs .spv files alongside the .glsl sources so they're found
# regardless of working directory (VS debugger or build output).
function(compile_shaders TARGET SHADER_DIR)
    find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
    if(NOT GLSLC)
        message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or add glslc to PATH.")
    endif()

    file(GLOB SHADER_SOURCES
        "${SHADER_DIR}/*.vert"
        "${SHADER_DIR}/*.frag"
        "${SHADER_DIR}/*.comp"
    )

    set(SPV_OUTPUTS "")
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE "${SHADER_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} "${SHADER}" -o "${SPV_FILE}"
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
            VERBATIM
        )
        list(APPEND SPV_OUTPUTS ${SPV_FILE})
    endforeach()

    add_custom_target(${TARGET}_shaders ALL DEPENDS ${SPV_OUTPUTS})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()
