#pragma once

#include <vex/graphics/mesh.h>
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>

namespace vex
{

class VKMesh : public Mesh
{
public:
    VKMesh() = default;
    ~VKMesh() override;

    VKMesh(const VKMesh&) = delete;
    VKMesh& operator=(const VKMesh&) = delete;

    void upload(const MeshData& data) override;
    void draw() const override;

    VkBuffer  getVertexBuffer() const { return m_vertexBuffer; }
    VkBuffer  getIndexBuffer()  const { return m_indexBuffer; }
    uint32_t  getVertexCount()  const { return m_vertexCount; }
    uint32_t  getIndexCount()   const { return m_indexCount; }

private:
    VkBuffer      m_vertexBuffer     = VK_NULL_HANDLE;
    VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;
    VkBuffer      m_indexBuffer      = VK_NULL_HANDLE;
    VmaAllocation m_indexAllocation  = VK_NULL_HANDLE;
    uint32_t      m_vertexCount      = 0;
    uint32_t      m_indexCount       = 0;
};

} // namespace vex
