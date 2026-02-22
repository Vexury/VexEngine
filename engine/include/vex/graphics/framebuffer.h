#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace vex
{

struct FramebufferSpec
{
    uint32_t width = 1280;
    uint32_t height = 720;
    bool hasDepth = false;
    bool depthOnly = false; // depth-only framebuffer (for shadow maps)
};

class Framebuffer
{
public:
    virtual ~Framebuffer() = default;

    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;
    virtual void setClearColor(float r, float g, float b, float a = 1.0f) { (void)r; (void)g; (void)b; (void)a; }
    virtual void clear(float r, float g, float b, float a = 1.0f) { setClearColor(r, g, b, a); }
    virtual int readPixel(int x, int y) const { (void)x; (void)y; return -1; }
    virtual std::vector<uint8_t> readPixels() const { return {}; }
    virtual bool flipsUV() const { return false; }

    virtual uintptr_t getColorAttachmentHandle() const = 0;
    virtual const FramebufferSpec& getSpec() const = 0;

    static std::unique_ptr<Framebuffer> create(const FramebufferSpec& spec);
};

} // namespace vex
