# Compile GLSL shaders to SPIR-V using glslc from the Vulkan SDK
# Outputs .spv files alongside the .glsl sources so they're found
# regardless of working directory (VS debugger or build output).
function(compile_shaders TARGET SHADER_DIR)
    find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
    if(NOT GLSLC)
        message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or add glslc to PATH.")
    endif()

    # Regular graphics / compute shaders
    file(GLOB SHADER_SOURCES
        "${SHADER_DIR}/*.vert"
        "${SHADER_DIR}/*.frag"
        "${SHADER_DIR}/*.comp"
    )

    # Ray tracing shader stages
    file(GLOB RT_SHADER_SOURCES
        "${SHADER_DIR}/*.rgen"
        "${SHADER_DIR}/*.rmiss"
        "${SHADER_DIR}/*.rchit"
        "${SHADER_DIR}/*.rahit"
        "${SHADER_DIR}/*.rint"
        "${SHADER_DIR}/*.rcall"
    )

    # Include files that RT shaders depend on (via GL_GOOGLE_include_directive)
    file(GLOB RT_SHADER_INCLUDES "${SHADER_DIR}/*.glsl")

    set(SPV_OUTPUTS "")

    # Compile regular shaders
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE "${SHADER_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} -I "${SHADER_DIR}" "${SHADER}" -o "${SPV_FILE}"
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
            VERBATIM
        )
        list(APPEND SPV_OUTPUTS ${SPV_FILE})
    endforeach()

    # Compile RT shaders with Vulkan 1.2 target (required for ray tracing extensions)
    foreach(SHADER ${RT_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_FILE "${SHADER_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPV_FILE}
            COMMAND ${GLSLC} --target-env=vulkan1.2 -I "${SHADER_DIR}" "${SHADER}" -o "${SPV_FILE}"
            DEPENDS ${SHADER} ${RT_SHADER_INCLUDES}
            COMMENT "Compiling RT shader ${SHADER_NAME} -> SPIR-V"
            VERBATIM
        )
        list(APPEND SPV_OUTPUTS ${SPV_FILE})
    endforeach()

    add_custom_target(${TARGET}_shaders ALL DEPENDS ${SPV_OUTPUTS})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()
