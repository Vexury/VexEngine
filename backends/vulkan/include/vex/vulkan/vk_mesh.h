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

private:
    VkBuffer      m_vertexBuffer     = VK_NULL_HANDLE;
    VmaAllocation m_vertexAllocation = VK_NULL_HANDLE;
    VkBuffer      m_indexBuffer      = VK_NULL_HANDLE;
    VmaAllocation m_indexAllocation  = VK_NULL_HANDLE;
    uint32_t      m_indexCount       = 0;
};

} // namespace vex
