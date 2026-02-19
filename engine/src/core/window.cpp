#include <vex/core/window.h>
#include <vex/core/log.h>
#include <GLFW/glfw3.h>

namespace vex
{

Window::~Window()
{
    shutdown();
}

bool Window::init(const WindowConfig& config, std::function<void()> preCreateHints)
{
    if (!glfwInit())
    {
        Log::error("Failed to initialize GLFW");
        return false;
    }

    if (preCreateHints)
        preCreateHints();

    if (config.maximized)
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    m_window = glfwCreateWindow(
        static_cast<int>(config.width),
        static_cast<int>(config.height),
        config.title.c_str(),
        nullptr, nullptr
    );

    if (!m_window)
    {
        Log::error("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    int w, h;
    glfwGetWindowSize(m_window, &w, &h);
    m_width = static_cast<uint32_t>(w);
    m_height = static_cast<uint32_t>(h);

    glfwSetWindowUserPointer(m_window, this);
    glfwSetWindowSizeCallback(m_window, onWindowSizeCallback);
    glfwSetScrollCallback(m_window, onScrollCallback);

    if (config.vsync)
        glfwSwapInterval(1);

    return true;
}

void Window::shutdown()
{
    if (m_window)
    {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool Window::shouldClose() const
{
    return m_window && glfwWindowShouldClose(m_window);
}

void Window::pollEvents()
{
    glfwPollEvents();
}

void Window::swapBuffers()
{
    if (m_window)
        glfwSwapBuffers(m_window);
}

void Window::onWindowSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self)
        return;

    self->m_width = static_cast<uint32_t>(width);
    self->m_height = static_cast<uint32_t>(height);

    if (self->m_resizeCallback)
        self->m_resizeCallback(self->m_width, self->m_height);
}

void Window::onScrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self && self->m_scrollCallback)
        self->m_scrollCallback(yoffset);
}

} // namespace vex
