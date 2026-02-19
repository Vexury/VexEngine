#include <vex/opengl/gl_context.h>
#include <vex/core/window.h>
#include <vex/core/log.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace vex
{

std::unique_ptr<GraphicsContext> GraphicsContext::create()
{
    return std::make_unique<GLContext>();
}

std::function<void()> GLContext::getWindowHints() const
{
    return []()
    {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    };
}

bool GLContext::init(Window& window)
{
    m_window = &window;

    glfwMakeContextCurrent(window.getNativeWindow());

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        Log::error("Failed to initialize GLAD");
        return false;
    }

    glEnable(GL_DEPTH_TEST);

    Log::info(std::string("OpenGL Renderer: ")
              + reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    Log::info(std::string("OpenGL Version: ")
              + reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    return true;
}

void GLContext::beginFrame()
{
    int w = static_cast<int>(m_window->getWidth());
    int h = static_cast<int>(m_window->getHeight());
    glViewport(0, 0, w, h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void GLContext::endFrame()
{
    m_window->swapBuffers();
    m_window->pollEvents();
}

void GLContext::shutdown()
{
    m_window = nullptr;
}

void GLContext::imguiInit(GLFWwindow* window)
{
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
}

void GLContext::imguiShutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
}

void GLContext::imguiNewFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
}

void GLContext::imguiRenderDrawData()
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup);
    }
}

} // namespace vex
