#pragma once

namespace vex { class GraphicsContext; }

struct Scene;
class SceneRenderer;

enum class RenderMode;
enum class Selection { None, Mesh, Skybox, Light, Sun, Camera };

class EditorUI
{
public:
    void renderViewport(SceneRenderer& renderer);
    void renderHierarchy(Scene& scene, SceneRenderer& renderer);
    void renderInspector(Scene& scene, SceneRenderer& renderer);
    void renderSettings(SceneRenderer& renderer);
    void renderConsole();
    void renderStats(SceneRenderer& renderer, Scene& scene, vex::GraphicsContext& ctx);

    Selection getSelectionType() const { return m_selectionType; }
    int       getSelectionIndex() const { return m_selectionIndex; }

    void clearSelection()
    {
        m_selectionType  = Selection::None;
        m_submeshIndex   = -1;
    }

    void setSelection(Selection type, int index = 0, int submesh = -1)
    {
        m_selectionType  = type;
        m_selectionIndex = index;
        m_submeshIndex   = submesh;
    }

    int getSelectedMeshGroup() const
    {
        return (m_selectionType == Selection::Mesh) ? m_selectionIndex : -1;
    }

    int getSelectedSubmesh() const { return m_submeshIndex; }

    bool isViewportHovered() const { return m_viewportHovered; }
    int  getRenderModeIndex() const { return m_renderModeIndex; }
    int  getDebugModeIndex() const { return m_debugModeIndex; }

    bool consumePickRequest(int& outX, int& outY);

private:
    Selection m_selectionType  = Selection::None;
    int       m_selectionIndex = 0;
    int       m_submeshIndex   = -1;

    bool m_viewportHovered = false;

    bool m_pickRequested = false;
    int  m_pickX = 0;
    int  m_pickY = 0;

    int m_renderModeIndex = 0;
    int m_debugModeIndex = 0;
    int m_prevEnvmapForRevert = 0;
};
