#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace vex
{

class Framebuffer;

class Skybox
{
public:
    virtual ~Skybox() = default;

    virtual bool load(const std::string& equirectPath) = 0;
    virtual void draw(const glm::mat4& inverseVP) const = 0;

    virtual void preparePipeline(const Framebuffer& /*fb*/) {}

    void  setEnvRotation(float r) { m_envRotation = r; }
    float getEnvRotation() const  { return m_envRotation; }

    static std::unique_ptr<Skybox> create();

protected:
    float m_envRotation = 0.0f;
};

} // namespace vex
