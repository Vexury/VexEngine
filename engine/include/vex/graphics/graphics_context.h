#pragma once

#include <functional>
#include <memory>
#include <string_view>

struct GLFWwindow;

namespace vex
{

class Window;

struct MemoryStats
{
    float usedMB = 0.0f;
    float budgetMB = 0.0f;
    bool available = false;
};

class GraphicsContext
{
public:
    virtual ~GraphicsContext() = default;

    virtual bool init(Window& window) = 0;
    virtual void shutdown() = 0;

    virtual void beginFrame() = 0;
    virtual void endFrame() = 0;

    virtual std::string_view backendName() const = 0;
    virtual std::function<void()> getWindowHints() const = 0;
    virtual MemoryStats getMemoryStats() const { return {}; }
    virtual void waitIdle() {}

    // ImGui backend integration — implemented by each backend
    virtual void imguiInit(GLFWwindow* window) = 0;
    virtual void imguiShutdown() = 0;
    virtual void imguiNewFrame() = 0;
    virtual void imguiRenderDrawData() = 0;

    static std::unique_ptr<GraphicsContext> create();
};

} // namespace vex
