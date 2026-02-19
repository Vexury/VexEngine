#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace vex
{

class Texture2D
{
public:
    virtual ~Texture2D() = default;

    virtual void bind(uint32_t slot = 0) = 0;
    virtual void unbind() = 0;

    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;
    virtual uintptr_t getNativeHandle() const = 0;

    virtual void setData(const void* data, uint32_t width, uint32_t height, uint32_t channels) = 0;

    static std::unique_ptr<Texture2D> create(uint32_t width, uint32_t height, uint32_t channels = 4);
    static std::unique_ptr<Texture2D> createFromFile(const std::string& path);
};

} // namespace vex
