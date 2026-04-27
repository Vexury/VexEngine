#pragma once

#include <functional>

namespace vex
{

class Window;
class GraphicsContext;

class UILayer
{
public:
    bool init(Window& window, GraphicsContext& context);
    void shutdown();

    void beginFrame();
    void endFrame();

    void setMenuBarCallback(std::function<void()> cb) { m_menuBarCallback = std::move(cb); }

private:
    GraphicsContext* m_context = nullptr;
    bool m_firstFrame = true;
    std::function<void()> m_menuBarCallback;
};

} // namespace vex
