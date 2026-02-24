#include <vex/vulkan/vk_gpu_raytracer.h>
#include <vex/vulkan/vk_context.h>
#include <vex/core/log.h>

#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace vex
{

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static uint32_t alignUp(uint32_t val, uint32_t alignment)
{
    return (val + alignment - 1) & ~(alignment - 1);
}

static VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(VKContext::get().getDevice(), &info);
}

// Allocate a GPU-only buffer with device address support (for AS scratch/storage).
static void createASBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkBuffer& outBuffer, VmaAllocation& outAlloc)
{
    auto allocator = VKContext::get().getAllocator();

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &outBuffer, &outAlloc, nullptr);
}

VkShaderModule VKGpuRaytracer::loadShader(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        Log::error("Failed to open RT shader: " + path);
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
        Log::error("Failed to create shader module: " + path);
        return VK_NULL_HANDLE;
    }
    return mod;
}

void VKGpuRaytracer::createAndUploadBuffer(const void* data, VkDeviceSize size,
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

void VKGpuRaytracer::destroyBuffer(VkBuffer& buf, VmaAllocation& alloc)
{
    if (buf)
    {
        vmaDestroyBuffer(VKContext::get().getAllocator(), buf, alloc);
        buf  = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// VKBlas / VKTlas cleanup
// ---------------------------------------------------------------------------

void VKBlas::destroy()
{
    auto& ctx = VKContext::get();
    if (handle) { vkDestroyAccelerationStructureKHR(ctx.getDevice(), handle, nullptr); handle = VK_NULL_HANDLE; }
    if (buffer) { vmaDestroyBuffer(ctx.getAllocator(), buffer, allocation);             buffer = VK_NULL_HANDLE; }
    deviceAddress = 0;
}

void VKTlas::destroy()
{
    auto& ctx = VKContext::get();
    if (handle) { vkDestroyAccelerationStructureKHR(ctx.getDevice(), handle, nullptr); handle = VK_NULL_HANDLE; }
    if (buffer) { vmaDestroyBuffer(ctx.getAllocator(), buffer, allocation);             buffer = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool VKGpuRaytracer::init()
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
        Log::error("VKGpuRaytracer: failed to create display sampler");
        return false;
    }

    return true;
}

void VKGpuRaytracer::shutdown()
{
    auto& ctx       = VKContext::get();
    auto  device    = ctx.getDevice();
    auto  allocator = ctx.getAllocator();

    clearAccelerationStructures();

    // Readback + output image
    destroyBuffer(m_readbackBuffer, m_readbackAlloc);
    if (m_outputImageView) { vkDestroyImageView(device, m_outputImageView, nullptr); m_outputImageView = VK_NULL_HANDLE; }
    if (m_outputImage)     { vmaDestroyImage(allocator, m_outputImage, m_outputAlloc); m_outputImage = VK_NULL_HANDLE; }

    // Scene SSBOs
    destroyBuffer(m_instanceOffsetsBuffer, m_instanceOffsetsAlloc);
    destroyBuffer(m_envCdfBuffer,          m_envCdfAlloc);
    destroyBuffer(m_envMapBuffer,          m_envMapAlloc);
    destroyBuffer(m_texDataBuffer,         m_texDataAlloc);
    destroyBuffer(m_lightsBuffer,          m_lightsAlloc);
    destroyBuffer(m_triShadingBuffer,      m_triShadingAlloc);

    // UBO
    destroyBuffer(m_uboBuffer, m_uboAlloc);
    m_uboMapped = nullptr;

    // SBT + pipeline
    if (m_sbtBuffer)       { vmaDestroyBuffer(allocator, m_sbtBuffer, m_sbtAllocation); m_sbtBuffer = VK_NULL_HANDLE; }
    if (m_descPool)        { vkDestroyDescriptorPool(device, m_descPool, nullptr);      m_descPool  = VK_NULL_HANDLE; }
    if (m_pipeline)        { vkDestroyPipeline(device, m_pipeline, nullptr);             m_pipeline  = VK_NULL_HANDLE; }
    if (m_pipelineLayout)  { vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr); m_pipelineLayout = VK_NULL_HANDLE; }
    if (m_descSetLayout)   { vkDestroyDescriptorSetLayout(device, m_descSetLayout, nullptr); m_descSetLayout = VK_NULL_HANDLE; }

    if (m_displaySampler)  { vkDestroySampler(device, m_displaySampler, nullptr); m_displaySampler = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// Acceleration structures
// ---------------------------------------------------------------------------

void VKGpuRaytracer::clearAccelerationStructures()
{
    // The previous frame's trace command may still be executing on the GPU and
    // referencing these acceleration structures. Wait for all GPU work to finish
    // before destroying them. This is an infrequent operation (geometry changes only).
    vkDeviceWaitIdle(VKContext::get().getDevice());

    m_tlas.destroy();
    for (auto& blas : m_blases) blas.destroy();
    m_blases.clear();
}

void VKGpuRaytracer::addBlas(VkBuffer vertexBuffer, uint32_t vertexCount,
                               VkDeviceSize vertexStride,
                               VkBuffer indexBuffer,  uint32_t indexCount)
{
    // Just stage the build — no GPU submission yet. Call commitBlasBuild() when done.
    auto& ctx    = VKContext::get();
    auto  device = ctx.getDevice();

    PendingBlas pb{};

    pb.geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    pb.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // No VK_GEOMETRY_OPAQUE_BIT_KHR — allows the any-hit shader to fire for alpha clipping
    auto& tri        = pb.geometry.geometry.triangles;
    tri.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = getBufferDeviceAddress(vertexBuffer);
    tri.vertexStride = vertexStride;
    tri.maxVertex    = vertexCount - 1;
    tri.indexType    = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress = getBufferDeviceAddress(indexBuffer);

    pb.primitiveCount = indexCount / 3;

    pb.buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    pb.buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    pb.buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    pb.buildInfo.geometryCount = 1;
    pb.buildInfo.pGeometries   = &pb.geometry; // re-pointed in commitBlasBuild after potential realloc

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &pb.buildInfo, &pb.primitiveCount, &sizeInfo);

    createASBuffer(sizeInfo.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                   pb.blas.buffer, pb.blas.allocation);

    VkAccelerationStructureCreateInfoKHR asInfo{};
    asInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asInfo.buffer = pb.blas.buffer;
    asInfo.size   = sizeInfo.accelerationStructureSize;
    asInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device, &asInfo, nullptr, &pb.blas.handle);

    createASBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   pb.scratchBuffer, pb.scratchAlloc);

    pb.buildInfo.dstAccelerationStructure  = pb.blas.handle;
    pb.buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(pb.scratchBuffer);

    pb.rangeInfo.primitiveCount = pb.primitiveCount;

    m_pendingBlases.push_back(std::move(pb));
}

void VKGpuRaytracer::commitBlasBuild()
{
    if (m_pendingBlases.empty()) return;

    auto& ctx    = VKContext::get();
    auto  device = ctx.getDevice();
    uint32_t count = static_cast<uint32_t>(m_pendingBlases.size());

    // Re-point pGeometries after any vector reallocs during addBlas calls.
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(count);
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pRanges(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        buildInfos[i]             = m_pendingBlases[i].buildInfo;
        buildInfos[i].pGeometries = &m_pendingBlases[i].geometry;
        pRanges[i]                = &m_pendingBlases[i].rangeInfo;
    }

    // One submission for all BLASes — eliminates N-1 round-trips to the GPU.
    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        vkCmdBuildAccelerationStructuresKHR(cmd, count, buildInfos.data(), pRanges.data());
    });

    // Query device addresses and free scratch buffers.
    for (auto& pb : m_pendingBlases)
    {
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = pb.blas.handle;
        pb.blas.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

        vmaDestroyBuffer(ctx.getAllocator(), pb.scratchBuffer, pb.scratchAlloc);
        m_blases.push_back(std::move(pb.blas));
    }
    m_pendingBlases.clear();
}

void VKGpuRaytracer::buildTlas()
{
    // Note: we intentionally allow an empty TLAS (zero instances).
    // With no geometry every ray misses and the miss shader returns the
    // sky/environment colour, which is the correct result for an empty scene.

    auto& ctx    = VKContext::get();
    auto  device = ctx.getDevice();

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(m_blases.size());

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_blases.size()); ++i)
    {
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0]                = 1.0f;
        inst.transform.matrix[1][1]                = 1.0f;
        inst.transform.matrix[2][2]                = 1.0f;
        inst.instanceCustomIndex                    = i;         // maps to gl_InstanceCustomIndexEXT
        inst.mask                                   = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference         = m_blases[i].deviceAddress;
        instances.push_back(inst);
    }

    uint32_t instanceCount = static_cast<uint32_t>(instances.size());

    // Vulkan does not allow size-0 buffers; use a 16-byte minimum.
    // The GPU won't access the buffer when primitiveCount == 0.
    VkDeviceSize instancesSize = std::max(
        instanceCount * static_cast<VkDeviceSize>(sizeof(VkAccelerationStructureInstanceKHR)),
        VkDeviceSize(16));

    VkBuffer      instanceBuffer;
    VmaAllocation instanceAlloc;
    {
        auto allocator = ctx.getAllocator();

        VkBuffer stagingBuf; VmaAllocation stagingAlloc;
        VkBufferCreateInfo si{}; si.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        si.size = instancesSize; si.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo sai{}; sai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(allocator, &si, &sai, &stagingBuf, &stagingAlloc, nullptr);
        if (instanceCount > 0)
        {
            void* mp; vmaMapMemory(allocator, stagingAlloc, &mp);
            std::memcpy(mp, instances.data(), instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
            vmaUnmapMemory(allocator, stagingAlloc);
        }

        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = instancesSize;
        bi.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo gai{}; gai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateBuffer(allocator, &bi, &gai, &instanceBuffer, &instanceAlloc, nullptr);

        ctx.immediateSubmit([&](VkCommandBuffer cmd)
        {
            VkBufferCopy copy{ 0, 0, instancesSize };
            vkCmdCopyBuffer(cmd, stagingBuf, instanceBuffer, 1, &copy);
        });
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
    }

    VkAccelerationStructureGeometryInstancesDataKHR instanceData{};
    instanceData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instanceData.data.deviceAddress = getBufferDeviceAddress(instanceBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instanceData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    m_tlas.destroy();
    createASBuffer(sizeInfo.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                   m_tlas.buffer, m_tlas.allocation);

    VkAccelerationStructureCreateInfoKHR asInfo{};
    asInfo.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asInfo.buffer = m_tlas.buffer;
    asInfo.size   = sizeInfo.accelerationStructureSize;
    asInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device, &asInfo, nullptr, &m_tlas.handle);

    VkBuffer scratchBuffer; VmaAllocation scratchAlloc;
    createASBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   scratchBuffer, scratchAlloc);

    buildInfo.dstAccelerationStructure  = m_tlas.handle;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;

    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    });

    vmaDestroyBuffer(ctx.getAllocator(), scratchBuffer, scratchAlloc);
    vmaDestroyBuffer(ctx.getAllocator(), instanceBuffer, instanceAlloc);

    Log::info("TLAS built: " + std::to_string(m_blases.size()) + " instance(s)");
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

bool VKGpuRaytracer::createPipeline()
{
    auto& ctx    = VKContext::get();
    auto  device = ctx.getDevice();
    auto  alloc  = ctx.getAllocator();

    // ── Shader stages (5 shaders) ────────────────────────────────────────────
    VkShaderModule rgenMod        = loadShader("shaders/vulkan/rt.rgen.spv");
    VkShaderModule missMod        = loadShader("shaders/vulkan/rt.rmiss.spv");
    VkShaderModule shadowMissMod  = loadShader("shaders/vulkan/rt.shadow.rmiss.spv");
    VkShaderModule chitMod        = loadShader("shaders/vulkan/rt.rchit.spv");
    VkShaderModule ahitMod        = loadShader("shaders/vulkan/rt.rahit.spv");

    if (!rgenMod || !missMod || !shadowMissMod || !chitMod || !ahitMod)
        return false;

    // Stage index: 0=rgen, 1=miss, 2=shadowmiss, 3=rchit, 4=rahit
    VkPipelineShaderStageCreateInfo stages[5]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,      rgenMod,       "main" };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR,        missMod,       "main" };
    stages[2] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR,        shadowMissMod, "main" };
    stages[3] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chitMod,       "main" };
    stages[4] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_ANY_HIT_BIT_KHR,     ahitMod,       "main" };

    // ── Shader groups ────────────────────────────────────────────────────────
    // Group 0: rgen  (general,    stage 0)
    // Group 1: miss  (general,    stage 1) — primary miss, missIndex=0
    // Group 2: smiss (general,    stage 2) — shadow miss,  missIndex=1
    // Group 3: hit   (triangles,  stage 3+4)
    VkRayTracingShaderGroupCreateInfoKHR groups[4]{};
    for (auto& g : groups)
    {
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = VK_SHADER_UNUSED_KHR;
        g.anyHitShader       = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;

    groups[1].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;

    groups[2].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;

    groups[3].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader     = 4;

    // ── Descriptor set layout (9 bindings) ───────────────────────────────────
    constexpr VkShaderStageFlags kAllRT =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR |
        VK_SHADER_STAGE_MISS_BIT_KHR   |
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    VkDescriptorSetLayoutBinding bindings[9]{};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1, kAllRT,                         nullptr };
    bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };
    bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };
    bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };
    bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };
    bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };
    bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, kAllRT,                         nullptr };

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = 9;
    setLayoutInfo.pBindings    = bindings;
    vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &m_descSetLayout);

    // ── Pipeline layout ──────────────────────────────────────────────────────
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts    = &m_descSetLayout;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout);

    // ── RT pipeline ──────────────────────────────────────────────────────────
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount                   = 5;
    pipelineInfo.pStages                      = stages;
    pipelineInfo.groupCount                   = 4;
    pipelineInfo.pGroups                      = groups;
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout                       = m_pipelineLayout;

    VkResult result = vkCreateRayTracingPipelinesKHR(device,
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);

    vkDestroyShaderModule(device, rgenMod,       nullptr);
    vkDestroyShaderModule(device, missMod,       nullptr);
    vkDestroyShaderModule(device, shadowMissMod, nullptr);
    vkDestroyShaderModule(device, chitMod,       nullptr);
    vkDestroyShaderModule(device, ahitMod,       nullptr);

    if (result != VK_SUCCESS)
    {
        Log::error("Failed to create RT pipeline");
        return false;
    }

    // ── Descriptor pool ──────────────────────────────────────────────────────
    VkDescriptorPoolSize poolSizes[4]{};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1 };
    poolSizes[2] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1 };
    poolSizes[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             6 };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 4;
    poolInfo.pPoolSizes    = poolSizes;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descPool);

    // ── UBO (always-mapped, persistent) ─────────────────────────────────────
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = sizeof(RTUniforms);
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo vmaInfo{};
        vmaCreateBuffer(alloc, &bi, &ai, &m_uboBuffer, &m_uboAlloc, &vmaInfo);
        m_uboMapped = static_cast<RTUniforms*>(vmaInfo.pMappedData);
        std::memset(m_uboMapped, 0, sizeof(RTUniforms));
    }

    buildSBT();

    Log::info("RT pipeline created");
    return true;
}

void VKGpuRaytracer::buildSBT()
{
    auto& ctx     = VKContext::get();
    auto  device  = ctx.getDevice();
    auto& rtProps = ctx.getRTProperties();

    const uint32_t handleSize      = rtProps.shaderGroupHandleSize;
    const uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    const uint32_t baseAlignment   = rtProps.shaderGroupBaseAlignment;

    // Each entry padded to handleAlignment; each region start padded to baseAlignment.
    const uint32_t stride = alignUp(handleSize, handleAlignment);

    // SBT layout:
    //   [rgen @ raygenOff]
    //   [miss0 @ missOff][miss1 @ missOff+stride]   — 2 miss entries
    //   [hit0  @ hitOff]
    const uint32_t raygenOff = 0;
    const uint32_t missOff   = alignUp(raygenOff + stride, baseAlignment);
    const uint32_t hitOff    = alignUp(missOff + 2u * stride, baseAlignment);
    const uint32_t sbtSize   = hitOff + stride;

    // Retrieve raw handles (4 groups: rgen, miss, shadowmiss, hit)
    constexpr uint32_t kGroupCount = 4;
    std::vector<uint8_t> handles(kGroupCount * handleSize);
    vkGetRayTracingShaderGroupHandlesKHR(device, m_pipeline, 0, kGroupCount,
                                          handles.size(), handles.data());

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size  = sbtSize;
    bufInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo vmaInfo{};
    vmaCreateBuffer(ctx.getAllocator(), &bufInfo, &allocInfo,
                    &m_sbtBuffer, &m_sbtAllocation, &vmaInfo);

    auto* data = static_cast<uint8_t*>(vmaInfo.pMappedData);
    std::memcpy(data + raygenOff,          handles.data() + 0 * handleSize, handleSize); // rgen
    std::memcpy(data + missOff,            handles.data() + 1 * handleSize, handleSize); // primary miss
    std::memcpy(data + missOff + stride,   handles.data() + 2 * handleSize, handleSize); // shadow miss
    std::memcpy(data + hitOff,             handles.data() + 3 * handleSize, handleSize); // hit group

    const VkDeviceAddress base = getBufferDeviceAddress(m_sbtBuffer);
    m_rgenRegion     = { base + raygenOff, stride, stride };
    m_missRegion     = { base + missOff,   stride, 2u * stride }; // size covers both miss shaders
    m_hitRegion      = { base + hitOff,    stride, stride };
    m_callableRegion = {};
}

// ---------------------------------------------------------------------------
// Scene data upload
// ---------------------------------------------------------------------------

void VKGpuRaytracer::uploadSceneData(
    const std::vector<float>&    triShading,
    const std::vector<uint32_t>& lightsData,
    const std::vector<uint32_t>& texData,
    const std::vector<float>&    envMapData,
    const std::vector<float>&    envCdfData,
    const std::vector<uint32_t>& instanceOffsets)
{
    // Ensure the previous frame's RT dispatch has finished before destroying
    // the SSBOs it was reading. The BLAS/TLAS rebuild already called
    // vkDeviceWaitIdle, but that was before the last trace submission.
    vkDeviceWaitIdle(VKContext::get().getDevice());

    // Destroy old SSBOs
    destroyBuffer(m_triShadingBuffer,      m_triShadingAlloc);
    destroyBuffer(m_lightsBuffer,          m_lightsAlloc);
    destroyBuffer(m_texDataBuffer,         m_texDataAlloc);
    destroyBuffer(m_envMapBuffer,          m_envMapAlloc);
    destroyBuffer(m_envCdfBuffer,          m_envCdfAlloc);
    destroyBuffer(m_instanceOffsetsBuffer, m_instanceOffsetsAlloc);

    // Helper: upload or create a 1-element dummy if empty
    auto upload = [&](const void* data, size_t bytes, VkBuffer& buf, VmaAllocation& alloc)
    {
        if (bytes == 0)
        {
            static const uint32_t kDummy = 0;
            createAndUploadBuffer(&kDummy, sizeof(kDummy), 0, buf, alloc);
        }
        else
        {
            createAndUploadBuffer(data, static_cast<VkDeviceSize>(bytes), 0, buf, alloc);
        }
    };

    upload(triShading.data(),       triShading.size()       * sizeof(float),    m_triShadingBuffer,      m_triShadingAlloc);
    upload(lightsData.data(),       lightsData.size()       * sizeof(uint32_t), m_lightsBuffer,          m_lightsAlloc);
    upload(texData.data(),          texData.size()          * sizeof(uint32_t), m_texDataBuffer,         m_texDataAlloc);
    upload(envMapData.data(),       envMapData.size()       * sizeof(float),    m_envMapBuffer,          m_envMapAlloc);
    upload(envCdfData.data(),       envCdfData.size()       * sizeof(float),    m_envCdfBuffer,          m_envCdfAlloc);
    upload(instanceOffsets.data(),  instanceOffsets.size()  * sizeof(uint32_t), m_instanceOffsetsBuffer, m_instanceOffsetsAlloc);

    Log::info("RT scene data uploaded: " + std::to_string(triShading.size() / 52) + " triangles");
}

// ---------------------------------------------------------------------------
// Output image + descriptor writes
// ---------------------------------------------------------------------------

void VKGpuRaytracer::writeDescriptors()
{
    if (!m_descSet) return;

    auto device = VKContext::get().getDevice();

    // Binding 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR tlasWrite{};
    tlasWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    tlasWrite.accelerationStructureCount = 1;
    tlasWrite.pAccelerationStructures    = &m_tlas.handle;

    // Binding 1: storage image
    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = m_outputImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Binding 2: UBO
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = m_uboBuffer;
    uboInfo.offset = 0;
    uboInfo.range  = sizeof(RTUniforms);

    // Bindings 3-8: SSBOs
    VkDescriptorBufferInfo ssboInfos[6]{};
    VkBuffer ssboBuffers[6] = {
        m_triShadingBuffer, m_lightsBuffer, m_texDataBuffer,
        m_envMapBuffer,     m_envCdfBuffer, m_instanceOffsetsBuffer
    };
    for (int i = 0; i < 6; ++i)
    {
        ssboInfos[i].buffer = ssboBuffers[i];
        ssboInfos[i].offset = 0;
        ssboInfos[i].range  = VK_WHOLE_SIZE;
    }

    VkWriteDescriptorSet writes[9]{};
    for (auto& w : writes) w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    writes[0].pNext           = &tlasWrite;
    writes[0].dstSet          = m_descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1].dstSet          = m_descSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &imgInfo;

    writes[2].dstSet          = m_descSet;
    writes[2].dstBinding      = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo     = &uboInfo;

    for (int i = 0; i < 6; ++i)
    {
        writes[3 + i].dstSet          = m_descSet;
        writes[3 + i].dstBinding      = static_cast<uint32_t>(3 + i);
        writes[3 + i].descriptorCount = 1;
        writes[3 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3 + i].pBufferInfo     = &ssboInfos[i];
    }

    // Only write non-null bindings
    uint32_t writeCount = 0;
    VkWriteDescriptorSet validWrites[9]{};

    if (m_tlas.handle)          validWrites[writeCount++] = writes[0];
    if (m_outputImageView)      validWrites[writeCount++] = writes[1];
    if (m_uboBuffer)            validWrites[writeCount++] = writes[2];
    if (m_triShadingBuffer)     validWrites[writeCount++] = writes[3];
    if (m_lightsBuffer)         validWrites[writeCount++] = writes[4];
    if (m_texDataBuffer)        validWrites[writeCount++] = writes[5];
    if (m_envMapBuffer)         validWrites[writeCount++] = writes[6];
    if (m_envCdfBuffer)         validWrites[writeCount++] = writes[7];
    if (m_instanceOffsetsBuffer)validWrites[writeCount++] = writes[8];

    if (writeCount > 0)
        vkUpdateDescriptorSets(device, writeCount, validWrites, 0, nullptr);
}

bool VKGpuRaytracer::createOutputImage(uint32_t width, uint32_t height)
{
    auto& ctx       = VKContext::get();
    auto  device    = ctx.getDevice();
    auto  allocator = ctx.getAllocator();

    // Destroy previous resources
    destroyBuffer(m_readbackBuffer, m_readbackAlloc);
    if (m_outputImageView) { vkDestroyImageView(device, m_outputImageView, nullptr); m_outputImageView = VK_NULL_HANDLE; }
    if (m_outputImage)     { vmaDestroyImage(allocator, m_outputImage, m_outputAlloc); m_outputImage = VK_NULL_HANDLE; }

    m_width  = width;
    m_height = height;

    // Create rgba32f storage image
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
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    // Create readback staging buffer
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size  = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(allocator, &bi, &ai, &m_readbackBuffer, &m_readbackAlloc, nullptr);
    }

    // (Re-)allocate descriptor set. The previous frame's RT dispatch may still
    // be in-flight referencing the current descriptor set, so wait before reset.
    vkDeviceWaitIdle(device);
    vkResetDescriptorPool(device, m_descPool, 0);
    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool     = m_descPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts        = &m_descSetLayout;
    vkAllocateDescriptorSets(device, &setInfo, &m_descSet);

    writeDescriptors();
    return true;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void VKGpuRaytracer::setUniforms(const RTUniforms& u)
{
    std::memcpy(m_uboMapped, &u, sizeof(RTUniforms));
}

void VKGpuRaytracer::trace(VkCommandBuffer cmd)
{
    if (!m_tlas.handle || !m_outputImage) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                             m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);
    vkCmdTraceRaysKHR(cmd,
                      &m_rgenRegion, &m_missRegion,
                      &m_hitRegion,  &m_callableRegion,
                      m_width, m_height, 1);
}

void VKGpuRaytracer::postTraceBarrier(VkCommandBuffer cmd)
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
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VKGpuRaytracer::reset()
{
    if (!m_outputImage) return;
    VKContext::get().immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkClearColorValue clear{};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_outputImage, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
    });
}

void VKGpuRaytracer::readbackRGBA8(std::vector<uint8_t>& out)
{
    if (!m_outputImage || !m_readbackBuffer) return;

    VKContext::get().immediateSubmit([&](VkCommandBuffer cmd)
    {
        // GENERAL -> TRANSFER_SRC
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
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkBufferImageCopy copy{};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent      = { m_width, m_height, 1 };
        vkCmdCopyImageToBuffer(cmd, m_outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               m_readbackBuffer, 1, &copy);

        // TRANSFER_SRC -> GENERAL
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    });

    // Map and tone-map to RGBA8
    float* pixels = nullptr;
    vmaMapMemory(VKContext::get().getAllocator(), m_readbackAlloc,
                 reinterpret_cast<void**>(&pixels));

    out.resize(m_width * m_height * 4);
    for (uint32_t i = 0; i < m_width * m_height; ++i)
    {
        float r = pixels[i * 4 + 0];
        float g = pixels[i * 4 + 1];
        float b = pixels[i * 4 + 2];
        float s = pixels[i * 4 + 3]; // sample count stored in alpha

        if (s > 0.0f) { r /= s; g /= s; b /= s; }

        auto tonemapGamma = [](float c) -> uint8_t {
            c = c / (1.0f + c);                          // Reinhard
            c = std::pow(std::max(c, 0.0f), 1.0f / 2.2f); // gamma
            return static_cast<uint8_t>(std::min(c * 255.0f + 0.5f, 255.0f));
        };

        out[i * 4 + 0] = tonemapGamma(r);
        out[i * 4 + 1] = tonemapGamma(g);
        out[i * 4 + 2] = tonemapGamma(b);
        out[i * 4 + 3] = 255;
    }

    vmaUnmapMemory(VKContext::get().getAllocator(), m_readbackAlloc);
}

} // namespace vex
