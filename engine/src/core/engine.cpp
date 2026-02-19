#include <vex/core/engine.h>
#include <vex/core/window.h>
#include <vex/core/log.h>
#include <vex/graphics/graphics_context.h>
#include <vex/ui/ui_layer.h>

namespace vex
{

Engine::Engine() = default;

Engine::~Engine()
{
    shutdown();
}

bool Engine::init(const EngineConfig& config)
{
    m_headless = config.headless;

    if (m_headless)
    {
        Log::info("Initializing VexEngine in headless mode");
        m_running = true;
        return true;
    }

    m_context = GraphicsContext::create();
    if (!m_context)
    {
        Log::error("Failed to create graphics context");
        return false;
    }

    Log::info(std::string("Using backend: ") + std::string(m_context->backendName()));

    m_window = std::make_unique<Window>();

    WindowConfig windowConfig;
    windowConfig.width = config.windowWidth;
    windowConfig.height = config.windowHeight;
    windowConfig.title = config.title;
    windowConfig.vsync = config.vsync;

    if (!m_window->init(windowConfig, m_context->getWindowHints()))
    {
        Log::error("Failed to initialize window");
        return false;
    }

    if (!m_context->init(*m_window))
    {
        Log::error("Failed to initialize graphics context");
        return false;
    }

    m_uiLayer = std::make_unique<UILayer>();
    if (!m_uiLayer->init(*m_window, *m_context))
    {
        Log::error("Failed to initialize UI layer");
        return false;
    }

    m_running = true;
    Log::info("VexEngine initialized successfully");
    return true;
}

void Engine::beginFrame()
{
    if (m_headless)
        return;

    m_context->beginFrame();
    m_uiLayer->beginFrame();
}

void Engine::endFrame()
{
    if (m_headless)
        return;

    m_uiLayer->endFrame();
    m_context->endFrame();
}

bool Engine::isRunning() const
{
    if (m_headless)
        return m_running;

    return m_running && !m_window->shouldClose();
}

void Engine::shutdown()
{
    if (!m_running)
        return;

    m_running = false;

    if (m_uiLayer)
    {
        m_uiLayer->shutdown();
        m_uiLayer.reset();
    }

    if (m_context)
    {
        m_context->shutdown();
        m_context.reset();
    }

    if (m_window)
    {
        m_window->shutdown();
        m_window.reset();
    }

    Log::info("VexEngine shut down");
}

} // namespace vex
