#pragma once

#include <cstdint>
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

    bool init(const EngineConfig& config);
    void shutdown();

    void beginFrame();
    void endFrame();
    bool isRunning() const;

    Window& getWindow() { return *m_window; }
    const Window& getWindow() const { return *m_window; }
    GraphicsContext& getGraphicsContext() { return *m_context; }
    const GraphicsContext& getGraphicsContext() const { return *m_context; }

private:
    std::unique_ptr<Window> m_window;
    std::unique_ptr<GraphicsContext> m_context;
    std::unique_ptr<UILayer> m_uiLayer;
    bool m_running = false;
    bool m_headless = false;
};

} // namespace vex
