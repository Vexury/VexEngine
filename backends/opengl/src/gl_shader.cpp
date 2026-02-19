#include <vex/opengl/gl_shader.h>
#include <vex/graphics/texture.h>
#include <vex/core/log.h>

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>

namespace vex
{

std::unique_ptr<Shader> Shader::create()
{
    return std::make_unique<GLShader>();
}

std::string Shader::shaderDir()
{
    return "shaders/opengl/";
}

std::string Shader::shaderExt()
{
    return "";
}

GLShader::~GLShader()
{
    if (m_programId)
        glDeleteProgram(m_programId);
}

uint32_t GLShader::compileShader(uint32_t type, const std::string& source)
{
    uint32_t id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    GLint result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        GLint length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog(static_cast<size_t>(length), '\0');
        glGetShaderInfoLog(id, length, &length, infoLog.data());
        Log::error("Shader compile error: " + infoLog);
        glDeleteShader(id);
        return 0;
    }

    return id;
}

bool GLShader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath)
{
    auto readFile = [](const std::string& path) -> std::string
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            Log::error("Failed to open shader: " + path);
            return "";
        }
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
    };

    std::string vertSrc = readFile(vertexPath);
    std::string fragSrc = readFile(fragmentPath);
    if (vertSrc.empty() || fragSrc.empty())
        return false;

    m_programId = glCreateProgram();

    uint32_t vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    uint32_t fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs)
        return false;

    glAttachShader(m_programId, vs);
    glAttachShader(m_programId, fs);
    glLinkProgram(m_programId);
    glValidateProgram(m_programId);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return true;
}

void GLShader::bind()   { glUseProgram(m_programId); }
void GLShader::unbind() { glUseProgram(0); }

int GLShader::getUniformLocation(const std::string& name)
{
    return glGetUniformLocation(m_programId, name.c_str());
}

void GLShader::setInt(const std::string& name, int value)
{
    glUniform1i(getUniformLocation(name), value);
}

void GLShader::setFloat(const std::string& name, float value)
{
    glUniform1f(getUniformLocation(name), value);
}

void GLShader::setBool(const std::string& name, bool value)
{
    glUniform1i(getUniformLocation(name), static_cast<int>(value));
}

void GLShader::setVec3(const std::string& name, const glm::vec3& value)
{
    glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void GLShader::setVec4(const std::string& name, const glm::vec4& value)
{
    glUniform4fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void GLShader::setMat4(const std::string& name, const glm::mat4& value)
{
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void GLShader::setWireframe(bool enabled)
{
    glPolygonMode(GL_FRONT_AND_BACK, enabled ? GL_LINE : GL_FILL);
}

void GLShader::setTexture(uint32_t slot, Texture2D* tex)
{
    tex->bind(slot);
    switch (slot)
    {
        case 0: setInt("u_diffuseMap",   0); break;
        case 1: setInt("u_normalMap",    1); break;
        case 2: setInt("u_roughnessMap", 2); break;
        case 3: setInt("u_metallicMap",  3); break;
        case 4: setInt("u_emissiveMap",  4); break;
        // slot 5 (envMap) is bound manually via glActiveTexture in the renderer
        default: break;
    }
}

} // namespace vex
