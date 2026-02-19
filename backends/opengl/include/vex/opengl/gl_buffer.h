#pragma once

#include <vex/graphics/buffer.h>
#include <cstdint>

namespace vex
{

class GLVertexBuffer : public VertexBuffer
{
public:
    GLVertexBuffer(const void* data, size_t size, BufferUsage usage);
    ~GLVertexBuffer() override;

    void bind() override;
    void unbind() override;
    void setData(const void* data, size_t size) override;

private:
    uint32_t m_id = 0;
};

class GLIndexBuffer : public IndexBuffer
{
public:
    GLIndexBuffer(const uint32_t* indices, uint32_t count);
    ~GLIndexBuffer() override;

    void bind() override;
    void unbind() override;
    uint32_t getCount() const override { return m_count; }

private:
    uint32_t m_id = 0;
    uint32_t m_count = 0;
};

} // namespace vex
