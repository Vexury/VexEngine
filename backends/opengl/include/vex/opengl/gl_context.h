#pragma once

#include <vex/graphics/graphics_context.h>

namespace vex
{

class GLContext : public GraphicsContext
{
public:
    bool init(Window& window) override;
    void shutdown() override;
    void beginFrame() override;
    void endFrame() override;
    std::string_view backendName() const override { return "OpenGL"; }
    std::function<void()> getWindowHints() const override;

    void imguiInit(GLFWwindow* window) override;
    void imguiShutdown() override;
    void imguiNewFrame() override;
    void imguiRenderDrawData() override;
    void setVSync(bool v) override;
    bool getVSync() const override { return m_vsync; }

private:
    Window* m_window = nullptr;
    bool    m_vsync  = true;
};

} // namespace vex
