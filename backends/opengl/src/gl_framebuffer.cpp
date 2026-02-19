#include <vex/opengl/gl_framebuffer.h>
#include <vex/core/log.h>

#include <glad/glad.h>

#include <algorithm>
#include <vector>

namespace vex
{

// Factory
std::unique_ptr<Framebuffer> Framebuffer::create(const FramebufferSpec& spec)
{
    return std::make_unique<GLFramebuffer>(spec);
}

GLFramebuffer::GLFramebuffer(const FramebufferSpec& spec)
    : m_spec(spec)
{
    create();
}

GLFramebuffer::~GLFramebuffer()
{
    destroy();
}

void GLFramebuffer::create()
{
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    // Color attachment
    glGenTextures(1, &m_colorAttachment);
    glBindTexture(GL_TEXTURE_2D, m_colorAttachment);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                 static_cast<GLsizei>(m_spec.width),
                 static_cast<GLsizei>(m_spec.height),
                 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorAttachment, 0);

    // Depth attachment
    if (m_spec.hasDepth)
    {
        glGenTextures(1, &m_depthAttachment);
        glBindTexture(GL_TEXTURE_2D, m_depthAttachment);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8,
                       static_cast<GLsizei>(m_spec.width),
                       static_cast<GLsizei>(m_spec.height));
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_depthAttachment, 0);
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        Log::error("Framebuffer incomplete");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLFramebuffer::destroy()
{
    if (m_fbo)
    {
        glDeleteFramebuffers(1, &m_fbo);
        m_fbo = 0;
    }
    if (m_colorAttachment)
    {
        glDeleteTextures(1, &m_colorAttachment);
        m_colorAttachment = 0;
    }
    if (m_depthAttachment)
    {
        glDeleteTextures(1, &m_depthAttachment);
        m_depthAttachment = 0;
    }
}

void GLFramebuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0,
               static_cast<GLsizei>(m_spec.width),
               static_cast<GLsizei>(m_spec.height));
}

void GLFramebuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLFramebuffer::clear(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

int GLFramebuffer::readPixel(int x, int y) const
{
    int readY = static_cast<int>(m_spec.height) - 1 - y;
    unsigned char pixel[4] = {};
    glReadPixels(x, readY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    return static_cast<int>(pixel[0]);
}

std::vector<uint8_t> GLFramebuffer::readPixels() const
{
    uint32_t w = m_spec.width;
    uint32_t h = m_spec.height;
    size_t rowBytes = static_cast<size_t>(w) * 4;
    std::vector<uint8_t> pixels(rowBytes * h);

    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    // FBO uses GL_RGBA32F, read as float then convert to u8
    std::vector<float> floatPixels(static_cast<size_t>(w) * h * 4);
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h),
                 GL_RGBA, GL_FLOAT, floatPixels.data());

    // Convert float [0,1] -> uint8 and flip vertically (OpenGL bottom-to-top -> top-to-bottom)
    for (uint32_t y = 0; y < h; ++y)
    {
        uint32_t srcRow = h - 1 - y;
        const float* src = floatPixels.data() + srcRow * w * 4;
        uint8_t* dst = pixels.data() + y * rowBytes;
        for (uint32_t x = 0; x < w * 4; ++x)
            dst[x] = static_cast<uint8_t>(std::clamp(src[x], 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return pixels;
}

void GLFramebuffer::resize(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) return;
    m_spec.width = width;
    m_spec.height = height;
    destroy();
    create();
}

} // namespace vex
