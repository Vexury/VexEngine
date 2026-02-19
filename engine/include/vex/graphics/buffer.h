#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vex
{

enum class BufferUsage { Static, Dynamic, Stream };

class VertexBuffer
{
public:
    virtual ~VertexBuffer() = default;

    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setData(const void* data, size_t size) = 0;

    static std::unique_ptr<VertexBuffer> create(const void* data, size_t size,
                                                BufferUsage usage = BufferUsage::Static);
};

class IndexBuffer
{
public:
    virtual ~IndexBuffer() = default;

    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual uint32_t getCount() const = 0;

    static std::unique_ptr<IndexBuffer> create(const uint32_t* indices, uint32_t count);
};

} // namespace vex
