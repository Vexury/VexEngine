#pragma once

#include <vex/graphics/skybox.h>
#include <cstdint>
#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace vex
{

class Shader;

class GLSkybox : public Skybox
{
public:
    GLSkybox() = default;
    ~GLSkybox() override;

    GLSkybox(const GLSkybox&) = delete;
    GLSkybox& operator=(const GLSkybox&) = delete;

    bool load(const std::string& equirectPath) override;
    void draw(const glm::mat4& inverseVP) const override;

private:
    uint32_t m_textureId = 0;
    uint32_t m_vao = 0;
    uint32_t m_vbo = 0;
    std::unique_ptr<Shader> m_shader;

    void createQuad();
};

} // namespace vex
