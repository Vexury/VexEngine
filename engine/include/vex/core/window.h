#pragma once

#include <cstdint>
#include <functional>
#include <string>

struct GLFWwindow;

namespace vex
{

struct WindowConfig
{
    uint32_t width = 1280;
    uint32_t height = 720;
    std::string title = "VexEngine";
    bool maximized = true;
    bool vsync = true;
};

class Window
{
public:
    Window() = default;
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool init(const WindowConfig& config, std::function<void()> preCreateHints = nullptr);
    void shutdown();

    bool shouldClose() const;
    void pollEvents();
    void swapBuffers();

    GLFWwindow* getNativeWindow() const { return m_window; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    using ResizeCallback = std::function<void(uint32_t, uint32_t)>;
    using ScrollCallback = std::function<void(double)>;

    void setResizeCallback(ResizeCallback cb) { m_resizeCallback = std::move(cb); }
    void setScrollCallback(ScrollCallback cb) { m_scrollCallback = std::move(cb); }

private:
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    ResizeCallback m_resizeCallback;
    ScrollCallback m_scrollCallback;

    static void onWindowSizeCallback(GLFWwindow* window, int width, int height);
    static void onScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};

} // namespace vex
