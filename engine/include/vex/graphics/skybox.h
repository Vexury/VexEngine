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

    static std::unique_ptr<Skybox> create();
};

} // namespace vex
