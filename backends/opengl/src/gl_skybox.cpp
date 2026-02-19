#include <vex/opengl/gl_skybox.h>
#include <vex/graphics/shader.h>
#include <vex/core/log.h>

#include <glad/glad.h>
#include <stb_image.h>

namespace vex
{

std::unique_ptr<Skybox> Skybox::create()
{
    return std::make_unique<GLSkybox>();
}

GLSkybox::~GLSkybox()
{
    if (m_textureId) glDeleteTextures(1, &m_textureId);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
}

void GLSkybox::createQuad()
{
    // Fullscreen triangle (covers screen with one triangle, no index buffer needed)
    float verts[] = {
        -1.0f, -1.0f,
         3.0f, -1.0f,
        -1.0f,  3.0f,
    };

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

    glBindVertexArray(0);
}

bool GLSkybox::load(const std::string& equirectPath)
{
    // Load HDR or LDR equirectangular image
    int w, h, ch;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(equirectPath.c_str(), &w, &h, &ch, 3);
    if (!data)
    {
        Log::error("Failed to load envmap: " + equirectPath);
        return false;
    }

    glGenTextures(1, &m_textureId);
    glBindTexture(GL_TEXTURE_2D, m_textureId);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    stbi_image_free(data);

    createQuad();

    m_shader = Shader::create();
    if (!m_shader->loadFromFiles("shaders/opengl/envmap.vert", "shaders/opengl/envmap.frag"))
    {
        Log::error("Failed to load envmap shaders");
        return false;
    }

    Log::info("Loaded envmap: " + equirectPath);
    return true;
}

void GLSkybox::draw(const glm::mat4& inverseVP) const
{
    if (!m_textureId || !m_shader) return;

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);

    m_shader->bind();
    m_shader->setMat4("u_inverseVP", inverseVP);
    m_shader->setInt("u_envmap", 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_textureId);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    m_shader->unbind();

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

} // namespace vex
