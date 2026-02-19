#pragma once

#include <vex/graphics/mesh.h>
#include <cstdint>

namespace vex
{

class GLMesh : public Mesh
{
public:
    GLMesh() = default;
    ~GLMesh() override;

    GLMesh(const GLMesh&) = delete;
    GLMesh& operator=(const GLMesh&) = delete;

    void upload(const MeshData& data) override;
    void draw() const override;

private:
    uint32_t m_vao = 0;
    uint32_t m_vbo = 0;
    uint32_t m_ibo = 0;
    uint32_t m_indexCount = 0;
};

} // namespace vex
