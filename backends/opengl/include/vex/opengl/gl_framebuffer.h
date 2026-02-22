#pragma once

#include <vex/graphics/framebuffer.h>
#include <cstdint>

namespace vex
{

class GLFramebuffer : public Framebuffer
{
public:
    explicit GLFramebuffer(const FramebufferSpec& spec);
    ~GLFramebuffer() override;

    void bind() override;
    void unbind() override;
    void resize(uint32_t width, uint32_t height) override;

    void clear(float r, float g, float b, float a = 1.0f) override;
    int readPixel(int x, int y) const override;
    std::vector<uint8_t> readPixels() const override;
    bool flipsUV() const override { return true; }

    uintptr_t getColorAttachmentHandle() const override
    {
        return static_cast<uintptr_t>(m_colorAttachment);
    }
    uint32_t getDepthAttachment() const { return m_depthAttachment; }
    // Toggle compare mode so the depth texture can be sampled as a plain float in ImGui
    void prepareDepthForDisplay();
    void restoreDepthForSampling();
    const FramebufferSpec& getSpec() const override { return m_spec; }

private:
    uint32_t m_fbo = 0;
    uint32_t m_colorAttachment = 0;
    uint32_t m_depthAttachment = 0;
    FramebufferSpec m_spec;

    void create();
    void destroy();
};

} // namespace vex
