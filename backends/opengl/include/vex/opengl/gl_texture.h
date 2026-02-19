#pragma once

#include <vex/graphics/texture.h>
#include <cstdint>

namespace vex
{

class GLTexture2D : public Texture2D
{
public:
    GLTexture2D(uint32_t width, uint32_t height, uint32_t channels);
    ~GLTexture2D() override;

    void bind(uint32_t slot = 0) override;
    void unbind() override;
    uint32_t getWidth() const override { return m_width; }
    uint32_t getHeight() const override { return m_height; }
    uintptr_t getNativeHandle() const override { return static_cast<uintptr_t>(m_id); }

    void setData(const void* data, uint32_t width, uint32_t height, uint32_t channels) override;

private:
    uint32_t m_id = 0;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_channels = 4;
};

} // namespace vex
