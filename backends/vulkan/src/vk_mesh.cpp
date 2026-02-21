#include <vex/vulkan/vk_mesh.h>
#include <vex/vulkan/vk_context.h>
#include <vex/scene/mesh_data.h>

#include <cstring>

namespace vex
{

std::unique_ptr<Mesh> Mesh::create()
{
    return std::make_unique<VKMesh>();
}

VKMesh::~VKMesh()
{
    auto allocator = VKContext::get().getAllocator();
    if (m_vertexBuffer)
        vmaDestroyBuffer(allocator, m_vertexBuffer, m_vertexAllocation);
    if (m_indexBuffer)
        vmaDestroyBuffer(allocator, m_indexBuffer, m_indexAllocation);
}

static void createGPUBuffer(const void* data, VkDeviceSize size,
                              VkBufferUsageFlags usage,
                              VkBuffer& outBuffer, VmaAllocation& outAlloc)
{
    auto& ctx = VKContext::get();
    auto allocator = ctx.getAllocator();

    // Staging buffer
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
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, stagingAlloc);

    // GPU buffer
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo gpuAllocInfo{};
    gpuAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateBuffer(allocator, &bufInfo, &gpuAllocInfo,
                    &outBuffer, &outAlloc, nullptr);

    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, outBuffer, 1, &copy);
    });

    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
}

void VKMesh::upload(const MeshData& data)
{
    m_vertexCount = static_cast<uint32_t>(data.vertices.size());
    m_indexCount  = static_cast<uint32_t>(data.indices.size());

    VkDeviceSize vertexSize = static_cast<VkDeviceSize>(data.vertices.size() * sizeof(Vertex));
    VkDeviceSize indexSize  = static_cast<VkDeviceSize>(data.indices.size()  * sizeof(uint32_t));

    // AS build input flags required for BLAS construction
    constexpr VkBufferUsageFlags kASInputFlags =
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    createGPUBuffer(data.vertices.data(), vertexSize,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | kASInputFlags,
                    m_vertexBuffer, m_vertexAllocation);

    createGPUBuffer(data.indices.data(), indexSize,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | kASInputFlags,
                    m_indexBuffer, m_indexAllocation);
}

void VKMesh::draw() const
{
    auto cmd = VKContext::get().getCurrentCommandBuffer();

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
}

} // namespace vex
