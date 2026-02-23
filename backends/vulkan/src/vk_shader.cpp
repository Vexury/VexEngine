#include <vex/vulkan/vk_shader.h>
#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_texture.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <fstream>
#include <cstring>

namespace vex
{

std::unique_ptr<Shader> Shader::create()
{
    return std::make_unique<VKShader>();
}

std::string Shader::shaderDir()
{
    return "shaders/vulkan/";
}

std::string Shader::shaderExt()
{
    return ".spv";
}

VKShader::~VKShader()
{
    auto device = VKContext::get().getDevice();
    auto allocator = VKContext::get().getAllocator();

    if (m_wireframePipeline)
        vkDestroyPipeline(device, m_wireframePipeline, nullptr);
    if (m_pipeline)
        vkDestroyPipeline(device, m_pipeline, nullptr);
    if (m_pipelineLayout)
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    if (m_externalTexturePool)
        vkDestroyDescriptorPool(device, m_externalTexturePool, nullptr);
    if (m_textureDescriptorPool)
        vkDestroyDescriptorPool(device, m_textureDescriptorPool, nullptr);
    if (m_descriptorPool)
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    if (m_textureSetLayout)
        vkDestroyDescriptorSetLayout(device, m_textureSetLayout, nullptr);
    if (m_descriptorSetLayout)
        vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
    if (m_vertModule)
        vkDestroyShaderModule(device, m_vertModule, nullptr);
    if (m_fragModule)
        vkDestroyShaderModule(device, m_fragModule, nullptr);

    for (auto& frame : m_frameUBOs)
    {
        if (frame.buffer)
            vmaDestroyBuffer(allocator, frame.buffer, frame.allocation);
    }
}

VkShaderModule VKShader::loadShaderModule(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        Log::error("Failed to open shader file: " + path);
        return VK_NULL_HANDLE;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = buffer.data();

    VkShaderModule module;
    if (vkCreateShaderModule(VKContext::get().getDevice(), &createInfo, nullptr, &module) != VK_SUCCESS)
    {
        Log::error("Failed to create shader module: " + path);
        return VK_NULL_HANDLE;
    }

    return module;
}

void VKShader::buildUniformMap()
{
    m_uniformOffsets["u_view"]       = offsetof(MeshUBO, view);
    m_uniformOffsets["u_projection"] = offsetof(MeshUBO, projection);
    m_uniformOffsets["u_cameraPos"]  = offsetof(MeshUBO, cameraPos);
    m_uniformOffsets["u_lightPos"]      = offsetof(MeshUBO, lightPos);
    m_uniformOffsets["u_lightColor"]    = offsetof(MeshUBO, lightColor);
    m_uniformOffsets["u_sunDirection"]  = offsetof(MeshUBO, sunDirection);
    m_uniformOffsets["u_sunColor"]      = offsetof(MeshUBO, sunColor);
    m_uniformOffsets["u_envColor"]           = offsetof(MeshUBO, envColor);
    m_uniformOffsets["u_enableEnvLighting"]  = offsetof(MeshUBO, enableEnvLighting);
    m_uniformOffsets["u_envLightMultiplier"] = offsetof(MeshUBO, envLightMultiplier);
    m_uniformOffsets["u_hasEnvMap"]          = offsetof(MeshUBO, hasEnvMap);
    m_uniformOffsets["u_shadowViewProj"]     = offsetof(MeshUBO, sunShadowVP);
    m_uniformOffsets["u_enableShadows"]      = offsetof(MeshUBO, enableShadows);
    m_uniformOffsets["u_shadowNormalBias"]   = offsetof(MeshUBO, shadowNormalBias);
}

bool VKShader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath)
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto allocator = ctx.getAllocator();

    m_vertModule = loadShaderModule(vertexPath);
    m_fragModule = loadShaderModule(fragmentPath);

    if (!m_vertModule || !m_fragModule)
        return false;

    // Set 0: UBO descriptor set layout
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
    {
        Log::error("Failed to create descriptor set layout");
        return false;
    }

    // Set 1: Texture descriptor set layout (combined image sampler)
    VkDescriptorSetLayoutBinding texBinding{};
    texBinding.binding = 0;
    texBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texBinding.descriptorCount = 1;
    texBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo texLayoutInfo{};
    texLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    texLayoutInfo.bindingCount = 1;
    texLayoutInfo.pBindings = &texBinding;

    if (vkCreateDescriptorSetLayout(device, &texLayoutInfo, nullptr, &m_textureSetLayout) != VK_SUCCESS)
    {
        Log::error("Failed to create texture descriptor set layout");
        return false;
    }

    // Pipeline layout with 8 set layouts + push constant
    // Set 0: UBO, Set 1: diffuse, Set 2: normal, Set 3: roughness, Set 4: metallic,
    // Set 5: emissive, Set 6: env map, Set 7: shadow map
    VkDescriptorSetLayout setLayouts[] = {
        m_descriptorSetLayout, m_textureSetLayout, m_textureSetLayout,
        m_textureSetLayout, m_textureSetLayout, m_textureSetLayout,
        m_textureSetLayout, m_textureSetLayout
    };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(MeshPushConstant);

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 8;
    plInfo.pSetLayouts = setLayouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
    {
        Log::error("Failed to create pipeline layout");
        return false;
    }

    // UBO descriptor pool (for per-frame UBO sets)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo dpInfo{};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &dpInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
    {
        Log::error("Failed to create descriptor pool");
        return false;
    }

    // Texture descriptor pool (for per-material texture sets — diffuse + normal maps)
    VkDescriptorPoolSize texPoolSize{};
    texPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texPoolSize.descriptorCount = 4096;

    VkDescriptorPoolCreateInfo texDpInfo{};
    texDpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    texDpInfo.maxSets = 4096;
    texDpInfo.poolSizeCount = 1;
    texDpInfo.pPoolSizes = &texPoolSize;

    if (vkCreateDescriptorPool(device, &texDpInfo, nullptr, &m_textureDescriptorPool) != VK_SUCCESS)
    {
        Log::error("Failed to create texture descriptor pool");
        return false;
    }

    // Small pool for external image views (e.g. RT output, shadow map). Kept separate so it
    // can be reset on resize without disturbing material texture descriptor sets.
    VkDescriptorPoolSize extPoolSize{};
    extPoolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    extPoolSize.descriptorCount = 8;

    VkDescriptorPoolCreateInfo extDpInfo{};
    extDpInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    extDpInfo.maxSets      = 8;
    extDpInfo.poolSizeCount = 1;
    extDpInfo.pPoolSizes   = &extPoolSize;

    if (vkCreateDescriptorPool(device, &extDpInfo, nullptr, &m_externalTexturePool) != VK_SUCCESS)
    {
        Log::error("Failed to create external texture descriptor pool");
        return false;
    }

    // Create per-frame UBOs and descriptor sets
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        // Create UBO buffer
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = m_uboSize;
        bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo vmaAllocInfo{};
        vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocInfo{};
        if (vmaCreateBuffer(allocator, &bufInfo, &vmaAllocInfo,
                           &m_frameUBOs[i].buffer, &m_frameUBOs[i].allocation,
                           &allocInfo) != VK_SUCCESS)
        {
            Log::error("Failed to create UBO buffer");
            return false;
        }
        m_frameUBOs[i].mapped = allocInfo.pMappedData;

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo dsAllocInfo{};
        dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAllocInfo.descriptorPool = m_descriptorPool;
        dsAllocInfo.descriptorSetCount = 1;
        dsAllocInfo.pSetLayouts = &m_descriptorSetLayout;

        vkAllocateDescriptorSets(device, &dsAllocInfo, &m_frameUBOs[i].descriptorSet);

        // Write descriptor
        VkDescriptorBufferInfo descBufInfo{};
        descBufInfo.buffer = m_frameUBOs[i].buffer;
        descBufInfo.offset = 0;
        descBufInfo.range = m_uboSize;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_frameUBOs[i].descriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &descBufInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    buildUniformMap();

    return true;
}

void VKShader::preparePipeline(const Framebuffer& fb)
{
    auto* vkFB = static_cast<const VKFramebuffer*>(&fb);
    m_cachedRenderPass = vkFB->getRenderPass();
    createPipeline(m_cachedRenderPass, true, true, true, VK_POLYGON_MODE_FILL);

    // Create wireframe pipeline if device supports it
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(VKContext::get().getPhysicalDevice(), &features);
    if (features.fillModeNonSolid)
    {
        createPipeline(m_cachedRenderPass, true, true, true, VK_POLYGON_MODE_LINE);
    }
}

void VKShader::createPipeline(VkRenderPass renderPass, bool depthTest,
                               bool depthWrite, bool hasVertexInput,
                               VkPolygonMode polygonMode, bool depthOnly)
{
    auto device = VKContext::get().getDevice();

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_fragModule;
    stages[1].pName = "main";

    // Vertex input
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[6]{};
    // position
    attrDescs[0].binding = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[0].offset = offsetof(Vertex, position);
    // normal
    attrDescs[1].binding = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[1].offset = offsetof(Vertex, normal);
    // color
    attrDescs[2].binding = 0;
    attrDescs[2].location = 2;
    attrDescs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[2].offset = offsetof(Vertex, color);
    // emissive
    attrDescs[3].binding = 0;
    attrDescs[3].location = 3;
    attrDescs[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[3].offset = offsetof(Vertex, emissive);
    // uv
    attrDescs[4].binding = 0;
    attrDescs[4].location = 4;
    attrDescs[4].format = VK_FORMAT_R32G32_SFLOAT;
    attrDescs[4].offset = offsetof(Vertex, uv);
    // tangent
    attrDescs[5].binding = 0;
    attrDescs[5].location = 5;
    attrDescs[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrDescs[5].offset = offsetof(Vertex, tangent);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (hasVertexInput)
    {
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &bindingDesc;
        vertexInput.vertexAttributeDescriptionCount = 6;
        vertexInput.pVertexAttributeDescriptions = attrDescs;
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport/scissor
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = polygonMode;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_TRUE; // values set dynamically via vkCmdSetDepthBias

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blending{};
    blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    if (depthOnly)
    {
        blending.attachmentCount = 0;
        blending.pAttachments = nullptr;
    }
    else
    {
        blending.attachmentCount = 1;
        blending.pAttachments = &blendAttachment;
    }

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    VkPipeline newPipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
    {
        Log::error("Failed to create graphics pipeline");
        return;
    }

    if (polygonMode == VK_POLYGON_MODE_LINE)
        m_wireframePipeline = newPipeline;
    else
        m_pipeline = newPipeline;
}

void VKShader::bind()
{
    auto& ctx = VKContext::get();
    auto cmd = ctx.getCurrentCommandBuffer();
    uint32_t frame = ctx.getCurrentFrameIndex();

    // Flush UBO data
    std::memcpy(m_frameUBOs[frame].mapped, &m_uboData, m_uboSize);

    // Bind pipeline (wireframe or fill) and descriptor set 0 (UBO)
    VkPipeline activePipeline = (m_wireframeActive && m_wireframePipeline)
                                ? m_wireframePipeline : m_pipeline;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_frameUBOs[frame].descriptorSet, 0, nullptr);

    // Push all constants
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(MeshPushConstant), &m_pushData);
}

void VKShader::unbind()
{
    // No-op in Vulkan
}

void VKShader::setWireframe(bool enabled)
{
    if (m_wireframeActive == enabled)
        return;
    m_wireframeActive = enabled;

    // Re-bind the appropriate pipeline
    if (m_wireframePipeline)
    {
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        VkPipeline activePipeline = enabled ? m_wireframePipeline : m_pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);
    }
}

void VKShader::setInt(const std::string& name, int value)
{
    if (name == "u_debugMode")
        m_pushData.debugMode = value;
    else if (name == "u_materialType")
        m_pushData.materialType = value;
}

void VKShader::setFloat(const std::string& name, float value)
{
    if (name == "u_nearPlane")
        m_pushData.nearPlane = value;
    else if (name == "u_farPlane")
        m_pushData.farPlane = value;
    else if (name == "u_roughness")
        m_pushData.roughness = value;
    else if (name == "u_metallic")
        m_pushData.metallic = value;
    else if (name == "u_sampleCount")
        m_pushData.sampleCount = value;
    else if (name == "u_exposure")
        m_pushData.exposure = value;
    else if (name == "u_gamma")
        m_pushData.gamma = value;
    else
    {
        auto it = m_uniformOffsets.find(name);
        if (it != m_uniformOffsets.end())
            std::memcpy(reinterpret_cast<char*>(&m_uboData) + it->second, &value, sizeof(float));
    }
}

void VKShader::setBool(const std::string& name, bool value)
{
    if (name == "u_alphaClip")
    {
        m_pushData.alphaClip = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_hasNormalMap")
    {
        m_pushData.hasNormalMap = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_hasRoughnessMap")
    {
        m_pushData.hasRoughnessMap = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_hasMetallicMap")
    {
        m_pushData.hasMetallicMap = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_flipV")
    {
        m_pushData.flipV = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_enableACES")
    {
        m_pushData.enableACES = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_hasEmissiveMap")
    {
        m_pushData.hasEmissiveMap = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else if (name == "u_enableOutline")
    {
        m_pushData.enableOutline = value ? 1u : 0u;
        auto cmd = VKContext::get().getCurrentCommandBuffer();
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(MeshPushConstant), &m_pushData);
    }
    else
    {
        auto it = m_uniformOffsets.find(name);
        if (it != m_uniformOffsets.end())
        {
            uint32_t v = value ? 1u : 0u;
            std::memcpy(reinterpret_cast<char*>(&m_uboData) + it->second, &v, sizeof(uint32_t));
        }
    }
}

void VKShader::setVec3(const std::string& name, const glm::vec3& value)
{
    auto it = m_uniformOffsets.find(name);
    if (it != m_uniformOffsets.end())
    {
        std::memcpy(reinterpret_cast<char*>(&m_uboData) + it->second, &value, sizeof(glm::vec3));
    }
}

void VKShader::setVec4(const std::string& name, const glm::vec4& value)
{
    auto it = m_uniformOffsets.find(name);
    if (it != m_uniformOffsets.end())
    {
        std::memcpy(reinterpret_cast<char*>(&m_uboData) + it->second, &value, sizeof(glm::vec4));
    }
}

void VKShader::setMat4(const std::string& name, const glm::mat4& value)
{
    auto it = m_uniformOffsets.find(name);
    if (it != m_uniformOffsets.end())
    {
        std::memcpy(reinterpret_cast<char*>(&m_uboData) + it->second, &value, sizeof(glm::mat4));
    }
}

void VKShader::setTexture(uint32_t slot, Texture2D* tex)
{
    auto* vkTex = static_cast<VKTexture2D*>(tex);
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto cmd = ctx.getCurrentCommandBuffer();

    // Look up or create descriptor set for this texture
    auto it = m_textureDescriptorSets.find(vkTex);
    VkDescriptorSet texSet;

    if (it != m_textureDescriptorSets.end())
    {
        texSet = it->second;
    }
    else
    {
        // Allocate a new descriptor set from the texture pool
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_textureDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_textureSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &texSet) != VK_SUCCESS)
        {
            Log::error("Failed to allocate texture descriptor set");
            return;
        }

        // Write the image descriptor
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = vkTex->getSampler();
        imageInfo.imageView = vkTex->getImageView();
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = texSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        m_textureDescriptorSets[vkTex] = texSet;
    }

    // Bind to set = 1 + slot (set 1 = diffuse, set 2 = normal map)
    uint32_t setIndex = 1 + slot;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            setIndex, 1, &texSet, 0, nullptr);
}

void VKShader::clearExternalTextureCache()
{
    m_externalTextureDescSets.clear();
    if (m_externalTexturePool)
        vkResetDescriptorPool(VKContext::get().getDevice(), m_externalTexturePool, 0);
}

void VKShader::setExternalTextureVK(uint32_t slot, VkImageView view, VkSampler sampler, VkImageLayout layout)
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto cmd = ctx.getCurrentCommandBuffer();

    auto it = m_externalTextureDescSets.find(view);
    VkDescriptorSet texSet;

    if (it != m_externalTextureDescSets.end())
    {
        texSet = it->second;
    }
    else
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_externalTexturePool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_textureSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &texSet) != VK_SUCCESS)
        {
            Log::error("Failed to allocate external texture descriptor set");
            return;
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler     = sampler;
        imageInfo.imageView   = view;
        imageInfo.imageLayout = layout;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = texSet;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo      = &imageInfo;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        m_externalTextureDescSets[view] = texSet;
    }

    uint32_t setIndex = 1 + slot;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            setIndex, 1, &texSet, 0, nullptr);
}

} // namespace vex
