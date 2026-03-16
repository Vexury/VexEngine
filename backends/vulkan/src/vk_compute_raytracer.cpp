#include <vex/vulkan/vk_compute_raytracer.h>
#include <vex/vulkan/vk_context.h>
#include <vex/core/log.h>

#include <fstream>
#include <cstring>
#include <algorithm>

namespace vex
{

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

VkShaderModule VKComputeRaytracer::loadShader(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        Log::error("VKComputeRaytracer: failed to open shader: " + path);
        return VK_NULL_HANDLE;
    }

    size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode    = code.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(VKContext::get().getDevice(), &info, nullptr, &mod) != VK_SUCCESS)
    {
        Log::error("VKComputeRaytracer: failed to create shader module: " + path);
        return VK_NULL_HANDLE;
    }
    return mod;
}

void VKComputeRaytracer::createAndUploadBuffer(const void* data, VkDeviceSize size,
                                                VkBufferUsageFlags extraUsage,
                                                VkBuffer& outBuf, VmaAllocation& outAlloc)
{
    auto& ctx       = VKContext::get();
    auto  allocator = ctx.getAllocator();

    // Staging buffer
    VkBuffer      stagingBuf;
    VmaAllocation stagingAlloc;
    {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size  = size;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(allocator, &info, &ai, &stagingBuf, &stagingAlloc, nullptr);
    }

    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, stagingAlloc);

    // Device buffer
    {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size  = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                   | extraUsage;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateBuffer(allocator, &info, &ai, &outBuf, &outAlloc, nullptr);
    }

    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy copy{ 0, 0, size };
        vkCmdCopyBuffer(cmd, stagingBuf, outBuf, 1, &copy);
    });

    vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
}

void VKComputeRaytracer::destroyBuffer(VkBuffer& buf, VmaAllocation& alloc)
{
    if (buf)
    {
        vmaDestroyBuffer(VKContext::get().getAllocator(), buf, alloc);
        buf  = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool VKComputeRaytracer::init()
{
    if (!createPipeline())
        return false;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter    = VK_FILTER_LINEAR;
    samplerInfo.minFilter    = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(VKContext::get().getDevice(), &samplerInfo, nullptr, &m_displaySampler) != VK_SUCCESS)
    {
        Log::error("VKComputeRaytracer: failed to create display sampler");
        return false;
    }

    Log::info("VK compute path tracer initialised");
    return true;
}

void VKComputeRaytracer::shutdown()
{
    auto& ctx       = VKContext::get();
    auto  device    = ctx.getDevice();
    auto  allocator = ctx.getAllocator();

    // Ensure no GPU work is in-flight
    vkDeviceWaitIdle(device);

    // Readback + output image
    destroyBuffer(m_readbackBuffer, m_readbackAlloc);
    if (m_outputImageView) { vkDestroyImageView(device, m_outputImageView, nullptr); m_outputImageView = VK_NULL_HANDLE; }
    if (m_outputImage)     { vmaDestroyImage(allocator, m_outputImage, m_outputAlloc); m_outputImage = VK_NULL_HANDLE; }

    // Aux images (albedo, normal)
    if (m_albedoImageView) { vkDestroyImageView(device, m_albedoImageView, nullptr); m_albedoImageView = VK_NULL_HANDLE; }
    if (m_albedoImage)     { vmaDestroyImage(allocator, m_albedoImage, m_albedoAlloc); m_albedoImage = VK_NULL_HANDLE; }
    if (m_normalImageView) { vkDestroyImageView(device, m_normalImageView, nullptr); m_normalImageView = VK_NULL_HANDLE; }
    if (m_normalImage)     { vmaDestroyImage(allocator, m_normalImage, m_normalAlloc); m_normalImage = VK_NULL_HANDLE; }

    // Scene SSBOs
    destroyBuffer(m_envCdfBuffer,    m_envCdfAlloc);
    destroyBuffer(m_envMapBuffer,    m_envMapAlloc);
    destroyBuffer(m_triShadingBuffer, m_triShadingAlloc);
    destroyBuffer(m_texDataBuffer,   m_texDataAlloc);
    destroyBuffer(m_lightsBuffer,    m_lightsAlloc);
    destroyBuffer(m_triVertsBuffer,  m_triVertsAlloc);
    destroyBuffer(m_bvhBuffer,       m_bvhAlloc);

    // UBO
    destroyBuffer(m_uboBuffer, m_uboAlloc);
    m_uboMapped = nullptr;

    // Pipeline
    if (m_descPool)       { vkDestroyDescriptorPool(device, m_descPool, nullptr);      m_descPool  = VK_NULL_HANDLE; }
    if (m_pipeline)       { vkDestroyPipeline(device, m_pipeline, nullptr);             m_pipeline  = VK_NULL_HANDLE; }
    if (m_pipelineLayout) { vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr); m_pipelineLayout = VK_NULL_HANDLE; }
    if (m_descSetLayout)  { vkDestroyDescriptorSetLayout(device, m_descSetLayout, nullptr); m_descSetLayout = VK_NULL_HANDLE; }

    if (m_displaySampler) { vkDestroySampler(device, m_displaySampler, nullptr); m_displaySampler = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

bool VKComputeRaytracer::createPipeline()
{
    auto& ctx    = VKContext::get();
    auto  device = ctx.getDevice();
    auto  alloc  = ctx.getAllocator();

    // ── Compute shader ───────────────────────────────────────────────────────
    VkShaderModule compMod = loadShader("shaders/vulkan/pathtracer.comp.spv");
    if (!compMod) return false;

    // ── Descriptor set layout (11 bindings, all COMPUTE) ────────────────────
    VkDescriptorSetLayoutBinding bindings[11]{};
    // binding 0: UBO
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    // bindings 1-7: SSBOs
    for (uint32_t i = 1; i <= 7; ++i)
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    // binding 8: accumulation storage image
    bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    // binding 9: albedo aux image; binding 10: normal aux image
    bindings[9]  = { 9,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = 11;
    setLayoutInfo.pBindings    = bindings;
    vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &m_descSetLayout);

    // ── Pipeline layout ──────────────────────────────────────────────────────
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts    = &m_descSetLayout;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout);

    // ── Compute pipeline ─────────────────────────────────────────────────────
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = compMod;
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage  = stageInfo;
    pipelineInfo.layout = m_pipelineLayout;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(device, compMod, nullptr);

    if (result != VK_SUCCESS)
    {
        Log::error("VKComputeRaytracer: failed to create compute pipeline");
        return false;
    }

    // ── Descriptor pool ──────────────────────────────────────────────────────
    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 };
    poolSizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  3 }; // main + albedo + normal

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes    = poolSizes;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descPool);

    // ── UBO (always-mapped, persistent) ─────────────────────────────────────
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = sizeof(VKComputeUniforms);
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo vmaInfo{};
        vmaCreateBuffer(alloc, &bi, &ai, &m_uboBuffer, &m_uboAlloc, &vmaInfo);
        m_uboMapped = static_cast<VKComputeUniforms*>(vmaInfo.pMappedData);
        std::memset(m_uboMapped, 0, sizeof(VKComputeUniforms));
    }

    return true;
}

// ---------------------------------------------------------------------------
// Geometry upload
// ---------------------------------------------------------------------------

void VKComputeRaytracer::uploadGeometry(
    const std::vector<CPURaytracer::Triangle>& triangles,
    const BVH& bvh,
    const std::vector<uint32_t>& lightIndices,
    const std::vector<float>& lightCDF,
    float totalLightArea,
    const std::vector<CPURaytracer::TextureData>& textures)
{
    // Wait for any in-flight compute work before destroying old SSBOs
    vkDeviceWaitIdle(VKContext::get().getDevice());

    destroyBuffer(m_bvhBuffer,        m_bvhAlloc);
    destroyBuffer(m_triVertsBuffer,   m_triVertsAlloc);
    destroyBuffer(m_lightsBuffer,     m_lightsAlloc);
    destroyBuffer(m_texDataBuffer,    m_texDataAlloc);
    destroyBuffer(m_triShadingBuffer, m_triShadingAlloc);

    m_triangleCount = static_cast<uint32_t>(triangles.size());
    m_bvhNodeCount  = static_cast<uint32_t>(bvh.nodeCount());

    // ── BVH nodes (8 floats / 32 bytes per node) ─────────────────────────────
    {
        struct GPUBVHNode {
            float    minX, minY, minZ;
            uint32_t leftFirst;
            float    maxX, maxY, maxZ;
            uint32_t triCount;
        };
        const auto& nodes = bvh.nodes();
        std::vector<GPUBVHNode> gpuNodes(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            gpuNodes[i].minX      = nodes[i].bounds.min.x;
            gpuNodes[i].minY      = nodes[i].bounds.min.y;
            gpuNodes[i].minZ      = nodes[i].bounds.min.z;
            gpuNodes[i].leftFirst = nodes[i].leftFirst;
            gpuNodes[i].maxX      = nodes[i].bounds.max.x;
            gpuNodes[i].maxY      = nodes[i].bounds.max.y;
            gpuNodes[i].maxZ      = nodes[i].bounds.max.z;
            gpuNodes[i].triCount  = nodes[i].triCount;
        }
        if (!gpuNodes.empty())
            createAndUploadBuffer(gpuNodes.data(), gpuNodes.size() * sizeof(GPUBVHNode),
                                  0, m_bvhBuffer, m_bvhAlloc);
        else
        {
            static const uint32_t kDummy = 0;
            createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, m_bvhBuffer, m_bvhAlloc);
        }
    }

    // ── Triangle vertices hot (3 vec4s = 12 floats per tri) ──────────────────
    {
        std::vector<float> vertsBuffer(triangles.size() * 12);
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            const auto& tri = triangles[i];
            float* p = &vertsBuffer[i * 12];
            p[0]  = tri.v0.x; p[1]  = tri.v0.y; p[2]  = tri.v0.z; p[3]  = 0.0f;
            p[4]  = tri.v1.x; p[5]  = tri.v1.y; p[6]  = tri.v1.z; p[7]  = 0.0f;
            p[8]  = tri.v2.x; p[9]  = tri.v2.y; p[10] = tri.v2.z; p[11] = 0.0f;
        }
        if (!vertsBuffer.empty())
            createAndUploadBuffer(vertsBuffer.data(), vertsBuffer.size() * sizeof(float),
                                  0, m_triVertsBuffer, m_triVertsAlloc);
        else
        {
            static const uint32_t kDummy = 0;
            createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, m_triVertsBuffer, m_triVertsAlloc);
        }
    }

    // ── Lights (headerless: [indices...][CDF float-bits...]) ─────────────────
    {
        uint32_t lc = static_cast<uint32_t>(lightIndices.size());
        std::vector<uint32_t> lightBuf(lc * 2);
        for (uint32_t i = 0; i < lc; ++i)
            lightBuf[i] = lightIndices[i];
        for (uint32_t i = 0; i < lc; ++i)
        {
            float cdf = lightCDF[i];
            std::memcpy(&lightBuf[lc + i], &cdf, sizeof(float));
        }
        if (!lightBuf.empty())
        {
            createAndUploadBuffer(lightBuf.data(), lightBuf.size() * sizeof(uint32_t),
                                  0, m_lightsBuffer, m_lightsAlloc);
        }
        else
        {
            static const uint32_t kDummy = 0;
            createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, m_lightsBuffer, m_lightsAlloc);
        }
        (void)totalLightArea; // stored in uniforms, not SSBO header
    }

    // ── Texture data ──────────────────────────────────────────────────────────
    {
        uint32_t texCount  = static_cast<uint32_t>(textures.size());
        uint32_t headerSz  = 1 + texCount * 4;
        uint32_t totalPx   = 0;
        for (const auto& tex : textures)
            totalPx += static_cast<uint32_t>(tex.width) * static_cast<uint32_t>(tex.height);

        std::vector<uint32_t> texBuf(headerSz + totalPx, 0);
        texBuf[0] = texCount;
        uint32_t curOffset = headerSz;
        for (uint32_t i = 0; i < texCount; ++i)
        {
            const auto& tex = textures[i];
            uint32_t px = static_cast<uint32_t>(tex.width) * static_cast<uint32_t>(tex.height);
            texBuf[1 + i * 4 + 0] = curOffset;
            texBuf[1 + i * 4 + 1] = static_cast<uint32_t>(tex.width);
            texBuf[1 + i * 4 + 2] = static_cast<uint32_t>(tex.height);
            texBuf[1 + i * 4 + 3] = 0;
            std::memcpy(&texBuf[curOffset], tex.pixels.data(), px * 4);
            curOffset += px;
        }
        if (!texBuf.empty())
            createAndUploadBuffer(texBuf.data(), texBuf.size() * sizeof(uint32_t),
                                  0, m_texDataBuffer, m_texDataAlloc);
    }

    // ── Triangle shading cold (10 vec4s = 40 floats per tri) ─────────────────
    {
        std::vector<float> shadingBuffer(triangles.size() * 40);
        for (size_t i = 0; i < triangles.size(); ++i)
        {
            const auto& tri = triangles[i];
            float* p = &shadingBuffer[i * 40];
            // vec4 0: n0 + roughnessTextureIndex
            p[0] = tri.n0.x; p[1] = tri.n0.y; p[2] = tri.n0.z;
            { uint32_t b; std::memcpy(&b, &tri.roughnessTextureIndex, sizeof(int)); std::memcpy(&p[3], &b, 4); }
            // vec4 1: n1 + metallicTextureIndex
            p[4] = tri.n1.x; p[5] = tri.n1.y; p[6] = tri.n1.z;
            { uint32_t b; std::memcpy(&b, &tri.metallicTextureIndex, sizeof(int)); std::memcpy(&p[7], &b, 4); }
            // vec4 2: n2 + emissiveStrength
            p[8] = tri.n2.x; p[9] = tri.n2.y; p[10] = tri.n2.z; p[11] = tri.emissiveStrength;
            // vec4 3: uv0.xy, uv1.xy
            p[12] = tri.uv0.x; p[13] = tri.uv0.y; p[14] = tri.uv1.x; p[15] = tri.uv1.y;
            // vec4 4: uv2.xy, roughness, metallic
            p[16] = tri.uv2.x; p[17] = tri.uv2.y; p[18] = tri.roughness; p[19] = tri.metallic;
            // vec4 5: color + textureIndex
            p[20] = tri.color.x; p[21] = tri.color.y; p[22] = tri.color.z;
            { uint32_t b; std::memcpy(&b, &tri.textureIndex, sizeof(int)); std::memcpy(&p[23], &b, 4); }
            // vec4 6: emissive + area
            p[24] = tri.emissive.x; p[25] = tri.emissive.y; p[26] = tri.emissive.z; p[27] = tri.area;
            // vec4 7: geometricNormal + normalMapTextureIndex
            p[28] = tri.geometricNormal.x; p[29] = tri.geometricNormal.y; p[30] = tri.geometricNormal.z;
            { uint32_t b; std::memcpy(&b, &tri.normalMapTextureIndex, sizeof(int)); std::memcpy(&p[31], &b, 4); }
            // vec4 8: alphaClip, materialType, ior, emissiveTextureIndex
            p[32] = tri.alphaClip ? 1.0f : 0.0f;
            p[33] = static_cast<float>(tri.materialType);
            p[34] = tri.ior;
            { uint32_t b; std::memcpy(&b, &tri.emissiveTextureIndex, sizeof(int)); std::memcpy(&p[35], &b, 4); }
            // vec4 9: tangent.xyz + bitangentSign
            p[36] = tri.tangent.x; p[37] = tri.tangent.y; p[38] = tri.tangent.z; p[39] = tri.bitangentSign;
        }
        if (!shadingBuffer.empty())
            createAndUploadBuffer(shadingBuffer.data(), shadingBuffer.size() * sizeof(float),
                                  0, m_triShadingBuffer, m_triShadingAlloc);
        else
        {
            static const uint32_t kDummy = 0;
            createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, m_triShadingBuffer, m_triShadingAlloc);
        }
    }

    Log::info("VKComputeRaytracer: uploaded " + std::to_string(m_triangleCount) +
              " triangles, " + std::to_string(m_bvhNodeCount) + " BVH nodes");

    if (m_outputImage)
        writeAllDescriptors();
}

// ---------------------------------------------------------------------------
// Environment map
// ---------------------------------------------------------------------------

void VKComputeRaytracer::uploadEnvironmentMap(const std::vector<float>& data, int w, int h,
                                               const std::vector<float>& cdf)
{
    vkDeviceWaitIdle(VKContext::get().getDevice());

    destroyBuffer(m_envMapBuffer, m_envMapAlloc);
    destroyBuffer(m_envCdfBuffer, m_envCdfAlloc);

    if (!data.empty())
        createAndUploadBuffer(data.data(), data.size() * sizeof(float),
                              0, m_envMapBuffer, m_envMapAlloc);
    else
    {
        static const float kDummy[3] = { 0.0f, 0.0f, 0.0f };
        createAndUploadBuffer(kDummy, sizeof(kDummy), 0, m_envMapBuffer, m_envMapAlloc);
    }

    if (!cdf.empty())
        createAndUploadBuffer(cdf.data(), cdf.size() * sizeof(float),
                              0, m_envCdfBuffer, m_envCdfAlloc);
    else
    {
        static const float kDummy = 0.0f;
        createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, m_envCdfBuffer, m_envCdfAlloc);
    }

    (void)w; (void)h; // stored in uniforms

    if (m_outputImage)
        writeEnvDescriptors();
}

void VKComputeRaytracer::clearEnvironmentMap()
{
    vkDeviceWaitIdle(VKContext::get().getDevice());

    destroyBuffer(m_envMapBuffer, m_envMapAlloc);
    destroyBuffer(m_envCdfBuffer, m_envCdfAlloc);

    static const float kDummyMap[3] = { 0.0f, 0.0f, 0.0f };
    static const float kDummyCdf    = 0.0f;
    createAndUploadBuffer(kDummyMap, sizeof(kDummyMap), 0, m_envMapBuffer, m_envMapAlloc);
    createAndUploadBuffer(&kDummyCdf, sizeof(kDummyCdf), 0, m_envCdfBuffer, m_envCdfAlloc);

    if (m_outputImage)
        writeEnvDescriptors();
}

// ---------------------------------------------------------------------------
// Output image
// ---------------------------------------------------------------------------

bool VKComputeRaytracer::createOutputImage(uint32_t width, uint32_t height)
{
    auto& ctx       = VKContext::get();
    auto  device    = ctx.getDevice();
    auto  allocator = ctx.getAllocator();

    destroyBuffer(m_readbackBuffer, m_readbackAlloc);
    if (m_outputImageView) { vkDestroyImageView(device, m_outputImageView, nullptr); m_outputImageView = VK_NULL_HANDLE; }
    if (m_outputImage)     { vmaDestroyImage(allocator, m_outputImage, m_outputAlloc); m_outputImage = VK_NULL_HANDLE; }
    if (m_albedoImageView) { vkDestroyImageView(device, m_albedoImageView, nullptr); m_albedoImageView = VK_NULL_HANDLE; }
    if (m_albedoImage)     { vmaDestroyImage(allocator, m_albedoImage, m_albedoAlloc); m_albedoImage = VK_NULL_HANDLE; }
    if (m_normalImageView) { vkDestroyImageView(device, m_normalImageView, nullptr); m_normalImageView = VK_NULL_HANDLE; }
    if (m_normalImage)     { vmaDestroyImage(allocator, m_normalImage, m_normalAlloc); m_normalImage = VK_NULL_HANDLE; }

    m_width  = width;
    m_height = height;

    // Create rgba32f storage + sampled image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.extent        = { width, height, 1 };
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_outputImage, &m_outputAlloc, nullptr);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image            = m_outputImage;
    viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(device, &viewInfo, nullptr, &m_outputImageView);

    // Transition UNDEFINED → GENERAL
    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = m_outputImage;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask       = 0;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    // Aux images (albedo + normal) — rgba32f, write-only from shader (no SAMPLED bit needed)
    {
        VkImageCreateInfo auxInfo{};
        auxInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        auxInfo.imageType     = VK_IMAGE_TYPE_2D;
        auxInfo.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        auxInfo.extent        = { width, height, 1 };
        auxInfo.mipLevels     = 1;
        auxInfo.arrayLayers   = 1;
        auxInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        auxInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        auxInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        auxInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo auxAllocInfo{};
        auxAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateImage(allocator, &auxInfo, &auxAllocInfo, &m_albedoImage, &m_albedoAlloc, nullptr);
        vmaCreateImage(allocator, &auxInfo, &auxAllocInfo, &m_normalImage, &m_normalAlloc, nullptr);

        VkImageViewCreateInfo auxViewInfo{};
        auxViewInfo.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        auxViewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        auxViewInfo.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
        auxViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        auxViewInfo.image = m_albedoImage;
        vkCreateImageView(device, &auxViewInfo, nullptr, &m_albedoImageView);
        auxViewInfo.image = m_normalImage;
        vkCreateImageView(device, &auxViewInfo, nullptr, &m_normalImageView);

        // Transition both aux images UNDEFINED → GENERAL
        ctx.immediateSubmit([&](VkCommandBuffer cmd)
        {
            VkImageMemoryBarrier barriers[2]{};
            for (auto& b : barriers)
            {
                b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                b.srcAccessMask       = 0;
                b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            }
            barriers[0].image = m_albedoImage;
            barriers[1].image = m_normalImage;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 2, barriers);
        });
    }

    // Readback staging buffer (4 floats per pixel)
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(allocator, &bi, &ai, &m_readbackBuffer, &m_readbackAlloc, nullptr);
    }

    // (Re-)allocate descriptor set
    vkDeviceWaitIdle(device);
    vkResetDescriptorPool(device, m_descPool, 0);
    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool     = m_descPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts        = &m_descSetLayout;
    vkAllocateDescriptorSets(device, &setInfo, &m_descSet);

    writeAllDescriptors();
    return true;
}

// ---------------------------------------------------------------------------
// Descriptor writes
// ---------------------------------------------------------------------------

void VKComputeRaytracer::writeAllDescriptors()
{
    if (!m_descSet) return;

    auto device = VKContext::get().getDevice();

    // Binding 0: UBO
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = m_uboBuffer;
    uboInfo.offset = 0;
    uboInfo.range  = sizeof(VKComputeUniforms);

    // Bindings 1-7: SSBOs
    VkBuffer ssboBuffers[7] = {
        m_bvhBuffer, m_triVertsBuffer, m_lightsBuffer, m_texDataBuffer,
        m_envMapBuffer, m_envCdfBuffer, m_triShadingBuffer
    };
    VkDescriptorBufferInfo ssboInfos[7]{};
    for (int i = 0; i < 7; ++i)
    {
        ssboInfos[i].buffer = ssboBuffers[i];
        ssboInfos[i].offset = 0;
        ssboInfos[i].range  = VK_WHOLE_SIZE;
    }

    // Binding 8: accumulation storage image
    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = m_outputImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Binding 9: albedo aux image; binding 10: normal aux image
    VkDescriptorImageInfo albedoImgInfo{};
    albedoImgInfo.imageView   = m_albedoImageView;
    albedoImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo normalImgInfo{};
    normalImgInfo.imageView   = m_normalImageView;
    normalImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[11]{};
    for (auto& w : writes) w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    writes[0].dstSet          = m_descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo     = &uboInfo;

    for (int i = 0; i < 7; ++i)
    {
        writes[1 + i].dstSet          = m_descSet;
        writes[1 + i].dstBinding      = static_cast<uint32_t>(1 + i);
        writes[1 + i].descriptorCount = 1;
        writes[1 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1 + i].pBufferInfo     = &ssboInfos[i];
    }

    writes[8].dstSet          = m_descSet;
    writes[8].dstBinding      = 8;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[8].pImageInfo      = &imgInfo;

    writes[9].dstSet          = m_descSet;
    writes[9].dstBinding      = 9;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[9].pImageInfo      = &albedoImgInfo;

    writes[10].dstSet          = m_descSet;
    writes[10].dstBinding      = 10;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[10].pImageInfo      = &normalImgInfo;

    // Only submit writes for non-null resources
    uint32_t writeCount = 0;
    VkWriteDescriptorSet validWrites[11]{};

    if (m_uboBuffer)        validWrites[writeCount++] = writes[0];
    if (m_bvhBuffer)        validWrites[writeCount++] = writes[1];
    if (m_triVertsBuffer)   validWrites[writeCount++] = writes[2];
    if (m_lightsBuffer)     validWrites[writeCount++] = writes[3];
    if (m_texDataBuffer)    validWrites[writeCount++] = writes[4];
    if (m_envMapBuffer)     validWrites[writeCount++] = writes[5];
    if (m_envCdfBuffer)     validWrites[writeCount++] = writes[6];
    if (m_triShadingBuffer) validWrites[writeCount++] = writes[7];
    if (m_outputImageView)  validWrites[writeCount++] = writes[8];
    if (m_albedoImageView)  validWrites[writeCount++] = writes[9];
    if (m_normalImageView)  validWrites[writeCount++] = writes[10];

    if (writeCount > 0)
        vkUpdateDescriptorSets(device, writeCount, validWrites, 0, nullptr);
}

void VKComputeRaytracer::writeEnvDescriptors()
{
    if (!m_descSet) return;

    auto device = VKContext::get().getDevice();

    VkDescriptorBufferInfo envMapInfo{};
    envMapInfo.buffer = m_envMapBuffer;
    envMapInfo.offset = 0;
    envMapInfo.range  = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo envCdfInfo{};
    envCdfInfo.buffer = m_envCdfBuffer;
    envCdfInfo.offset = 0;
    envCdfInfo.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = m_descSet;
    writes[0].dstBinding      = 5;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &envMapInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = m_descSet;
    writes[1].dstBinding      = 6;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo     = &envCdfInfo;

    uint32_t count = 0;
    VkWriteDescriptorSet valid[2]{};
    if (m_envMapBuffer) valid[count++] = writes[0];
    if (m_envCdfBuffer) valid[count++] = writes[1];
    if (count > 0)
        vkUpdateDescriptorSets(device, count, valid, 0, nullptr);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void VKComputeRaytracer::setUniforms(const VKComputeUniforms& u)
{
    if (m_uboMapped)
        std::memcpy(m_uboMapped, &u, sizeof(VKComputeUniforms));
}

void VKComputeRaytracer::traceSample(VkCommandBuffer cmd)
{
    if (!m_outputImage || !m_descSet) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);
    vkCmdDispatch(cmd, (m_width + 7) / 8, (m_height + 7) / 8, 1);
}

void VKComputeRaytracer::postTraceBarrier(VkCommandBuffer cmd)
{
    if (!m_outputImage) return;

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = m_outputImage;
    barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VKComputeRaytracer::reset()
{
    if (!m_outputImage) return;
    VKContext::get().immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkClearColorValue clear{};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_outputImage, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        if (m_albedoImage)
            vkCmdClearColorImage(cmd, m_albedoImage, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        if (m_normalImage)
            vkCmdClearColorImage(cmd, m_normalImage, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
    });
}

void VKComputeRaytracer::readbackLinearHDR(std::vector<float>& outRGB)
{
    if (!m_outputImage || !m_readbackBuffer) return;

    VKContext::get().immediateSubmit([&](VkCommandBuffer cmd)
    {
        // GENERAL → TRANSFER_SRC
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = m_outputImage;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy copy{};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent      = { m_width, m_height, 1 };
        vkCmdCopyImageToBuffer(cmd, m_outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               m_readbackBuffer, 1, &copy);

        // TRANSFER_SRC → GENERAL
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    float* pixels = nullptr;
    vmaMapMemory(VKContext::get().getAllocator(), m_readbackAlloc,
                 reinterpret_cast<void**>(&pixels));
    outRGB.resize(m_width * m_height * 3);
    for (uint32_t i = 0; i < m_width * m_height; ++i)
    {
        float s   = pixels[i * 4 + 3];
        float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
        outRGB[i * 3 + 0] = pixels[i * 4 + 0] * inv;
        outRGB[i * 3 + 1] = pixels[i * 4 + 1] * inv;
        outRGB[i * 3 + 2] = pixels[i * 4 + 2] * inv;
    }
    vmaUnmapMemory(VKContext::get().getAllocator(), m_readbackAlloc);
}

void VKComputeRaytracer::readbackAuxBuffers(std::vector<float>& outAlbedo, std::vector<float>& outNormal)
{
    if (!m_albedoImage || !m_normalImage || !m_readbackBuffer) return;

    auto& ctx = VKContext::get();
    auto  allocator = ctx.getAllocator();

    auto readbackOne = [&](VkImage img, std::vector<float>& out)
    {
        ctx.immediateSubmit([&](VkCommandBuffer cmd)
        {
            VkImageMemoryBarrier barrier{};
            barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image               = img;
            barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy copy{};
            copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copy.imageExtent      = { m_width, m_height, 1 };
            vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   m_readbackBuffer, 1, &copy);

            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        });

        float* pixels = nullptr;
        vmaMapMemory(allocator, m_readbackAlloc, reinterpret_cast<void**>(&pixels));
        out.resize(m_width * m_height * 3);
        for (uint32_t i = 0; i < m_width * m_height; ++i)
        {
            out[i * 3 + 0] = pixels[i * 4 + 0];
            out[i * 3 + 1] = pixels[i * 4 + 1];
            out[i * 3 + 2] = pixels[i * 4 + 2];
        }
        vmaUnmapMemory(allocator, m_readbackAlloc);
    };

    readbackOne(m_albedoImage, outAlbedo);
    readbackOne(m_normalImage, outNormal);
}

} // namespace vex
