#pragma once

#include <memory>
#include <string>
#include <cstdint>
#include <glm/glm.hpp>

namespace vex
{

class Framebuffer;
class Texture2D;

class Shader
{
public:
    virtual ~Shader() = default;

    virtual bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) = 0;
    virtual void bind() = 0;
    virtual void unbind() = 0;

    virtual void setInt(const std::string& name, int value) = 0;
    virtual void setFloat(const std::string& name, float value) = 0;
    virtual void setBool(const std::string& name, bool value) = 0;
    virtual void setVec3(const std::string& name, const glm::vec3& value) = 0;
    virtual void setVec4(const std::string& name, const glm::vec4& value) = 0;
    virtual void setMat4(const std::string& name, const glm::mat4& value) = 0;

    virtual void setTexture(uint32_t slot, Texture2D* tex) = 0;

    virtual void setWireframe(bool /*enabled*/) {}
    virtual void preparePipeline(const Framebuffer& /*fb*/) {}

    static std::unique_ptr<Shader> create();
    static std::string shaderDir();
    static std::string shaderExt();
};

} // namespace vex
