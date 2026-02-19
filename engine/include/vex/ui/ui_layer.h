#pragma once

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

private:
    GraphicsContext* m_context = nullptr;
    bool m_firstFrame = true;
};

} // namespace vex
