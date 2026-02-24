#include <vex/opengl/gl_texture.h>
#include <vex/core/log.h>

#include <glad/glad.h>
#include <stb_image.h>
#include <tinyexr.h>

#include <algorithm>
#include <cstring>

namespace vex
{

// Factory methods
std::unique_ptr<Texture2D> Texture2D::create(uint32_t width, uint32_t height, uint32_t channels)
{
    return std::make_unique<GLTexture2D>(width, height, channels);
}

std::unique_ptr<Texture2D> Texture2D::createFromFile(const std::string& path)
{
    // EXR path
    if (path.size() >= 4 &&
        (path.compare(path.size() - 4, 4, ".exr") == 0 ||
         path.compare(path.size() - 4, 4, ".EXR") == 0))
    {
        float* exrRGBA = nullptr;
        int w = 0, h = 0;
        const char* err = nullptr;
        if (LoadEXR(&exrRGBA, &w, &h, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            std::string errMsg = err ? err : "";
            if (errMsg == "Unknown compression type.")
                errMsg += " (DWAA/DWAB not supported — re-export with ZIP or PIZ compression)";
            Log::error("Failed to load EXR texture: " + path +
                       (errMsg.empty() ? "" : " (" + errMsg + ")"));
            FreeEXRErrorMessage(err);
            return nullptr;
        }

        // Convert float RGBA → uint8, then flip vertically
        std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; ++y)
        {
            int srcRow = (h - 1 - y); // flip
            for (int x = 0; x < w; ++x)
            {
                const float* src = exrRGBA + (srcRow * w + x) * 4;
                unsigned char* dst = pixels.data() + (y * w + x) * 4;
                for (int c = 0; c < 4; ++c)
                    dst[c] = static_cast<unsigned char>(
                        std::clamp(src[c], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        free(exrRGBA);

        auto tex = std::make_unique<GLTexture2D>(
            static_cast<uint32_t>(w), static_cast<uint32_t>(h), 4);
        tex->setData(pixels.data(), static_cast<uint32_t>(w),
                     static_cast<uint32_t>(h), 4);
        return tex;
    }

    int w, h, ch;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!data)
    {
        Log::error("Failed to load texture: " + path);
        return nullptr;
    }

    auto tex = std::make_unique<GLTexture2D>(
        static_cast<uint32_t>(w), static_cast<uint32_t>(h), 4);
    tex->setData(data, static_cast<uint32_t>(w), static_cast<uint32_t>(h), 4);
    stbi_image_free(data);
    return tex;
}

static GLenum channelsToFormat(uint32_t channels)
{
    switch (channels)
    {
        case 1: return GL_RED;
        case 3: return GL_RGB;
        case 4: return GL_RGBA;
        default: return GL_RGBA;
    }
}

static GLenum channelsToInternalFormat(uint32_t channels)
{
    switch (channels)
    {
        case 1: return GL_R8;
        case 3: return GL_RGB8;
        case 4: return GL_RGBA8;
        default: return GL_RGBA8;
    }
}

GLTexture2D::GLTexture2D(uint32_t width, uint32_t height, uint32_t channels)
    : m_width(width), m_height(height), m_channels(channels)
{
    glGenTextures(1, &m_id);
    glBindTexture(GL_TEXTURE_2D, m_id);

    glTexImage2D(GL_TEXTURE_2D, 0,
                 static_cast<GLint>(channelsToInternalFormat(channels)),
                 static_cast<GLsizei>(width), static_cast<GLsizei>(height),
                 0, channelsToFormat(channels), GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

GLTexture2D::~GLTexture2D()
{
    if (m_id) glDeleteTextures(1, &m_id);
}

void GLTexture2D::bind(uint32_t slot)
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, m_id);
}

void GLTexture2D::unbind()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLTexture2D::setData(const void* data, uint32_t width, uint32_t height, uint32_t channels)
{
    m_width = width;
    m_height = height;
    m_channels = channels;

    glBindTexture(GL_TEXTURE_2D, m_id);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 static_cast<GLint>(channelsToInternalFormat(channels)),
                 static_cast<GLsizei>(width), static_cast<GLsizei>(height),
                 0, channelsToFormat(channels), GL_UNSIGNED_BYTE, data);
}

} // namespace vex
