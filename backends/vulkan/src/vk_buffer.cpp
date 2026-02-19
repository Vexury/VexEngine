#include <vex/vulkan/vk_buffer.h>
#include <vex/vulkan/vk_context.h>
#include <vex/core/log.h>

#include <cstring>

namespace vex
{

// Factories
std::unique_ptr<VertexBuffer> VertexBuffer::create(const void* data, size_t size, BufferUsage usage)
{
    return std::make_unique<VKVertexBuffer>(data, size, usage);
}

std::unique_ptr<IndexBuffer> IndexBuffer::create(const uint32_t* indices, uint32_t count)
{
    return std::make_unique<VKIndexBuffer>(indices, count);
}

// Helper: create a GPU buffer via staging
static void createBufferWithStaging(const void* data, VkDeviceSize size,
                                     VkBufferUsageFlags usage,
                                     VkBuffer& outBuffer, VmaAllocation& outAllocation)
{
    auto& ctx = VKContext::get();
    auto allocator = ctx.getAllocator();

    // Create staging buffer
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

    // Copy data to staging
    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, stagingAlloc);

    // Create GPU buffer
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo gpuAllocInfo{};
    gpuAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateBuffer(allocator, &bufInfo, &gpuAllocInfo,
                    &outBuffer, &outAllocation, nullptr);

    // Copy via command buffer
    ctx.immediateSubmit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, outBuffer, 1, &copy);
    });

    // Destroy staging
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
}

// --- VKVertexBuffer ---

VKVertexBuffer::VKVertexBuffer(const void* data, size_t size, BufferUsage /*usage*/)
    : m_size(size)
{
    if (data && size > 0)
    {
        createBufferWithStaging(data, static_cast<VkDeviceSize>(size),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                m_buffer, m_allocation);
    }
}

VKVertexBuffer::~VKVertexBuffer()
{
    if (m_buffer)
        vmaDestroyBuffer(VKContext::get().getAllocator(), m_buffer, m_allocation);
}

void VKVertexBuffer::bind()
{
    auto cmd = VKContext::get().getCurrentCommandBuffer();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_buffer, &offset);
}

void VKVertexBuffer::unbind()
{
    // No-op in Vulkan
}

void VKVertexBuffer::setData(const void* data, size_t size)
{
    if (m_buffer)
        vmaDestroyBuffer(VKContext::get().getAllocator(), m_buffer, m_allocation);

    m_size = size;
    createBufferWithStaging(data, static_cast<VkDeviceSize>(size),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            m_buffer, m_allocation);
}

// --- VKIndexBuffer ---

VKIndexBuffer::VKIndexBuffer(const uint32_t* indices, uint32_t count)
    : m_count(count)
{
    VkDeviceSize size = static_cast<VkDeviceSize>(count) * sizeof(uint32_t);
    createBufferWithStaging(indices, size,
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            m_buffer, m_allocation);
}

VKIndexBuffer::~VKIndexBuffer()
{
    if (m_buffer)
        vmaDestroyBuffer(VKContext::get().getAllocator(), m_buffer, m_allocation);
}

void VKIndexBuffer::bind()
{
    auto cmd = VKContext::get().getCurrentCommandBuffer();
    vkCmdBindIndexBuffer(cmd, m_buffer, 0, VK_INDEX_TYPE_UINT32);
}

void VKIndexBuffer::unbind()
{
    // No-op
}

} // namespace vex
