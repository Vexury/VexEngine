#pragma once

#include <vex/graphics/buffer.h>
#include <volk.h>
#include <vk_mem_alloc.h>

namespace vex
{

class VKVertexBuffer : public VertexBuffer
{
public:
    VKVertexBuffer(const void* data, size_t size, BufferUsage usage);
    ~VKVertexBuffer() override;

    void bind() override;
    void unbind() override;
    void setData(const void* data, size_t size) override;

    VkBuffer getBuffer() const { return m_buffer; }

private:
    VkBuffer      m_buffer     = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    size_t        m_size       = 0;
};

class VKIndexBuffer : public IndexBuffer
{
public:
    VKIndexBuffer(const uint32_t* indices, uint32_t count);
    ~VKIndexBuffer() override;

    void bind() override;
    void unbind() override;
    uint32_t getCount() const override { return m_count; }

    VkBuffer getBuffer() const { return m_buffer; }

private:
    VkBuffer      m_buffer     = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    uint32_t      m_count      = 0;
};

} // namespace vex
