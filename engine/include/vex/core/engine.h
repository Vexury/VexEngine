#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vex
{

class Window;
class GraphicsContext;
class UILayer;

struct EngineConfig
{
    uint32_t windowWidth = 1280;
    uint32_t windowHeight = 720;
    std::string title = "VexEngine";
    bool headless = false;
    bool vsync = true;
};

class Engine
{
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    using ContextFactory = std::function<std::unique_ptr<GraphicsContext>()>;
    bool init(const EngineConfig& config, ContextFactory contextFactory);
    void shutdown();

    void beginFrame();
    void endFrame();
    bool isRunning() const;

    // Asks the main loop to stop. Kept separate from m_running so that a later
    // shutdown() still tears the window and context down.
    void requestExit() { m_exitRequested = true; }

    Window& getWindow() { return *m_window; }
    const Window& getWindow() const { return *m_window; }
    GraphicsContext& getGraphicsContext() { return *m_context; }
    const GraphicsContext& getGraphicsContext() const { return *m_context; }
    UILayer& getUILayer() { return *m_uiLayer; }

private:
    std::unique_ptr<Window> m_window;
    std::unique_ptr<GraphicsContext> m_context;
    std::unique_ptr<UILayer> m_uiLayer;
    bool m_running = false;
    bool m_headless = false;
    bool m_exitRequested = false;
};

} // namespace vex
