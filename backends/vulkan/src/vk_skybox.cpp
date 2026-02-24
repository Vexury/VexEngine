#include <vex/vulkan/vk_skybox.h>
#include <vex/vulkan/vk_context.h>
#include <vex/vulkan/vk_framebuffer.h>
#include <vex/core/log.h>

#include <stb_image.h>

#include <fstream>
#include <cstring>
#include <vector>

namespace vex
{

std::unique_ptr<Skybox> Skybox::create()
{
    return std::make_unique<VKSkybox>();
}

VKSkybox::~VKSkybox()
{
    destroyResources();
}

void VKSkybox::destroyResources()
{
    auto device = VKContext::get().getDevice();
    auto allocator = VKContext::get().getAllocator();

    if (m_pipeline)            vkDestroyPipeline(device, m_pipeline, nullptr);
    if (m_pipelineLayout)      vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    if (m_descriptorPool)      vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    if (m_descriptorSetLayout) vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
    if (m_vertModule)          vkDestroyShaderModule(device, m_vertModule, nullptr);
    if (m_fragModule)          vkDestroyShaderModule(device, m_fragModule, nullptr);

    if (m_textureSampler)    vkDestroySampler(device, m_textureSampler, nullptr);
    if (m_textureImageView)  vkDestroyImageView(device, m_textureImageView, nullptr);
    if (m_textureImage)      vmaDestroyImage(allocator, m_textureImage, m_textureAllocation);

    if (m_vertexBuffer) vmaDestroyBuffer(allocator, m_vertexBuffer, m_vertexAllocation);

    for (auto& frame : m_frames)
    {
        if (frame.uboBuffer)
            vmaDestroyBuffer(allocator, frame.uboBuffer, frame.uboAllocation);
    }

    m_pipeline = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_loaded = false;
}

void VKSkybox::createQuad()
{
    float verts[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f,
    };

    auto& ctx = VKContext::get();
    auto allocator = ctx.getAllocator();

    // Create staging
    VkDeviceSize size = sizeof(verts);

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = size;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, verts, sizeof(verts));
    vmaUnmapMemory(allocator, stagingAlloc);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo gpuAllocInfo{};
    gpuAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateBuffer(allocator, &bufInfo, &gpuAllocInfo,
                    &m_vertexBuffer, &m_vertexAllocation, nullptr);

    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, m_vertexBuffer, 1, &copy);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
}

bool VKSkybox::load(const std::string& equirectPath)
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto allocator = ctx.getAllocator();

    // If reloading, destroy old texture
    if (m_textureImage)
    {
        vkDeviceWaitIdle(device);
        if (m_textureSampler)   vkDestroySampler(device, m_textureSampler, nullptr);
        if (m_textureImageView) vkDestroyImageView(device, m_textureImageView, nullptr);
        vmaDestroyImage(allocator, m_textureImage, m_textureAllocation);
        m_textureSampler = VK_NULL_HANDLE;
        m_textureImageView = VK_NULL_HANDLE;
        m_textureImage = VK_NULL_HANDLE;
    }

    // Load image (HDR as float, LDR as uint8)
    int w, h, ch;
    stbi_set_flip_vertically_on_load(false);

    VkDeviceSize imageSize;
    void* pixelData;
    std::vector<float> floatBuf; // owns HDR pixel data until upload

    if (stbi_is_hdr(equirectPath.c_str()))
    {
        float* data = stbi_loadf(equirectPath.c_str(), &w, &h, &ch, 4);
        if (!data)
        {
            Log::error("Failed to load envmap: " + equirectPath);
            return false;
        }
        imageSize = static_cast<VkDeviceSize>(w) * h * 4 * sizeof(float);
        floatBuf.assign(data, data + w * h * 4);
        stbi_image_free(data);
        pixelData = floatBuf.data();
        m_textureFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    else
    {
        unsigned char* data = stbi_load(equirectPath.c_str(), &w, &h, &ch, 4);
        if (!data)
        {
            Log::error("Failed to load envmap: " + equirectPath);
            return false;
        }
        imageSize = static_cast<VkDeviceSize>(w) * h * 4;
        // Wrap in vector for uniform cleanup
        floatBuf.resize(imageSize); // reuse as byte storage (just need lifetime)
        std::memcpy(floatBuf.data(), data, static_cast<size_t>(imageSize));
        stbi_image_free(data);
        pixelData = floatBuf.data();
        m_textureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    }

    // Staging buffer
    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, pixelData, static_cast<size_t>(imageSize));
    vmaUnmapMemory(allocator, stagingAlloc);

    // Create image
    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = m_textureFormat;
    imgInfo.extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(allocator, &imgInfo, &imgAllocInfo,
                   &m_textureImage, &m_textureAllocation, nullptr);

    // Transfer
    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = m_textureImage;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };

        vkCmdCopyBufferToImage(cmd, stagingBuffer, m_textureImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_textureImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_textureFormat;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(device, &viewInfo, nullptr, &m_textureImageView);

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkCreateSampler(device, &samplerInfo, nullptr, &m_textureSampler);

    // Create quad if needed
    if (!m_vertexBuffer)
        createQuad();

    // Update descriptor sets with new texture
    if (m_descriptorPool)
    {
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorImageInfo imgDescInfo{};
            imgDescInfo.sampler = m_textureSampler;
            imgDescInfo.imageView = m_textureImageView;
            imgDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_frames[i].descriptorSet;
            write.dstBinding = 1;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imgDescInfo;

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    m_loaded = true;
    Log::info("Loaded envmap: " + equirectPath);
    return true;
}

void VKSkybox::preparePipeline(const Framebuffer& fb)
{
    auto* vkFB = static_cast<const VKFramebuffer*>(&fb);
    createPipeline(vkFB->getRenderPass());
}

void VKSkybox::createPipeline(VkRenderPass renderPass)
{
    auto& ctx = VKContext::get();
    auto device = ctx.getDevice();
    auto allocator = ctx.getAllocator();

    // Load shader modules
    auto loadModule = [&](const std::string& path) -> VkShaderModule
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            Log::error("Failed to open shader: " + path);
            return VK_NULL_HANDLE;
        }
        size_t size = static_cast<size_t>(file.tellg());
        std::vector<uint32_t> code(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = size;
        ci.pCode = code.data();
        VkShaderModule module;
        vkCreateShaderModule(device, &ci, nullptr, &module);
        return module;
    };

    m_vertModule = loadModule("shaders/vulkan/envmap.vert.spv");
    m_fragModule = loadModule("shaders/vulkan/envmap.frag.spv");

    if (!m_vertModule || !m_fragModule)
    {
        Log::error("Failed to load envmap shaders");
        return;
    }

    // Descriptor set layout: binding 0 = UBO (inverseVP), binding 1 = sampler
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 2;
    dslInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &m_descriptorSetLayout);

    // Pipeline layout
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_descriptorSetLayout;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &m_pipelineLayout);

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo dpInfo{};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    dpInfo.poolSizeCount = 2;
    dpInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device, &dpInfo, nullptr, &m_descriptorPool);

    // Per-frame UBO + descriptor sets
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = sizeof(glm::mat4);
        bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo vmaInfo{};
        vmaInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        vmaInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo allocResult{};
        vmaCreateBuffer(allocator, &bufInfo, &vmaInfo,
                       &m_frames[i].uboBuffer, &m_frames[i].uboAllocation, &allocResult);
        m_frames[i].uboMapped = allocResult.pMappedData;

        VkDescriptorSetAllocateInfo dsAllocInfo{};
        dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAllocInfo.descriptorPool = m_descriptorPool;
        dsAllocInfo.descriptorSetCount = 1;
        dsAllocInfo.pSetLayouts = &m_descriptorSetLayout;
        vkAllocateDescriptorSets(device, &dsAllocInfo, &m_frames[i].descriptorSet);

        // Write UBO binding
        VkDescriptorBufferInfo bufDescInfo{};
        bufDescInfo.buffer = m_frames[i].uboBuffer;
        bufDescInfo.range = sizeof(glm::mat4);

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_frames[i].descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bufDescInfo;

        uint32_t writeCount = 1;

        // Write texture binding if texture is loaded
        VkDescriptorImageInfo imgDescInfo{};
        if (m_textureImage)
        {
            imgDescInfo.sampler = m_textureSampler;
            imgDescInfo.imageView = m_textureImageView;
            imgDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = m_frames[i].descriptorSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &imgDescInfo;
            writeCount = 2;
        }

        vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
    }

    // Create graphics pipeline
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_fragModule;
    stages[1].pName = "main";

    // Vertex input: vec2 position
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.stride = 2 * sizeof(float);

    VkVertexInputAttributeDescription attrDesc{};
    attrDesc.format = VK_FORMAT_R32G32_SFLOAT;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attrDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blending{};
    blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blending.attachmentCount = 1;
    blending.pAttachments = &blendAttach;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &blending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = renderPass;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS)
    {
        Log::error("Failed to create envmap pipeline");
    }
}

void VKSkybox::draw(const glm::mat4& inverseVP) const
{
    if (!m_loaded || !m_pipeline) return;

    auto& ctx = VKContext::get();
    auto cmd = ctx.getCurrentCommandBuffer();
    uint32_t frame = ctx.getCurrentFrameIndex();

    // Update UBO
    std::memcpy(m_frames[frame].uboMapped, &inverseVP, sizeof(glm::mat4));

    // Bind pipeline and descriptors
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_frames[frame].descriptorSet, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer, &offset);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace vex
