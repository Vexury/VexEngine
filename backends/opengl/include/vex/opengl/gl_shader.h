#pragma once

#include <vex/graphics/shader.h>
#include <cstdint>

namespace vex
{

class GLShader : public Shader
{
public:
    GLShader() = default;
    ~GLShader() override;

    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) override;
    void bind() override;
    void unbind() override;

    void setInt(const std::string& name, int value) override;
    void setFloat(const std::string& name, float value) override;
    void setBool(const std::string& name, bool value) override;
    void setVec3(const std::string& name, const glm::vec3& value) override;
    void setVec4(const std::string& name, const glm::vec4& value) override;
    void setMat4(const std::string& name, const glm::mat4& value) override;

    void setTexture(uint32_t slot, Texture2D* tex) override;

    void setWireframe(bool enabled) override;

private:
    uint32_t m_programId = 0;

    uint32_t compileShader(uint32_t type, const std::string& source);
    int getUniformLocation(const std::string& name);
};

} // namespace vex
