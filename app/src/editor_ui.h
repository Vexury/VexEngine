#pragma once

#include <glm/glm.hpp>
#include <imgui.h>
#include <string>

namespace vex { class GraphicsContext; }

struct Scene;
class SceneRenderer;

enum class RenderMode;
enum class Selection { None, Mesh, Skybox, Light, Sun, Camera };

class EditorUI
{
public:
    void renderViewport(SceneRenderer& renderer, Scene& scene);
    void renderHierarchy(Scene& scene, SceneRenderer& renderer);
    void renderInspector(Scene& scene, SceneRenderer& renderer);
    void renderSettings(SceneRenderer& renderer);
    void renderConsole();
    void renderStats(SceneRenderer& renderer, Scene& scene, vex::GraphicsContext& ctx);

    Selection getSelectionType() const { return m_selectionType; }
    int       getSelectionIndex() const { return m_selectionIndex; }

    void clearSelection()
    {
        m_selectionType      = Selection::None;
        m_submeshIndex       = -1;
        m_selectedObjectName.clear();
    }

    void setSelection(Selection type, int index = 0, int submesh = -1)
    {
        m_selectionType  = type;
        m_selectionIndex = index;
        m_submeshIndex   = submesh;
        m_selectedObjectName.clear();
    }

    // Called after a viewport pick so we can resolve the object name from mesh data.
    void setSelectedObjectName(const std::string& name) { m_selectedObjectName = name; }
    const std::string& getSelectedObjectName() const    { return m_selectedObjectName; }

    int getSelectedMeshGroup() const
    {
        return (m_selectionType == Selection::Mesh) ? m_selectionIndex : -1;
    }

    int getSelectedSubmesh() const { return m_submeshIndex; }

    bool isViewportHovered() const { return m_viewportHovered; }
    int  getRenderModeIndex() const { return m_renderModeIndex; }
    int  getDebugModeIndex() const { return m_debugModeIndex; }

    bool consumePickRequest(int& outX, int& outY);

    void setGizmoMode(int mode) { m_gizmoMode = mode; }

    // Deferred import: the Import OBJ button stores the path here instead of blocking
    // the current frame. App::run() calls consumePendingImport() between frames.
    bool consumePendingImport(std::string& outPath, std::string& outName);

    // Loading overlay: called by App::runImport() to show progress between frames.
    void setLoadingState(const std::string& stage, float progress);
    void clearLoadingState();
    void renderLoadingOverlay();

private:
    Selection   m_selectionType       = Selection::None;
    int         m_selectionIndex      = 0;
    int         m_submeshIndex        = -1;  // -1 = object/group level, >=0 = specific submesh
    std::string m_selectedObjectName;         // object (shape) selected within a group

    bool m_viewportHovered = false;

    bool m_pickRequested = false;
    int  m_pickX = 0;
    int  m_pickY = 0;

    int m_renderModeIndex = 0;
    int m_debugModeIndex = 0;
    int m_prevEnvmapForRevert = 0;

    // Gizmo state
    int       m_gizmoMode     = 0;   // 0=Translate  1=Rotate  2=Scale
    int       m_gizmoAxis     = -1;  // 0=X 1=Y 2=Z  -1=none active
    bool      m_gizmoDragging = false;
    ImVec2    m_gizmoDragStart   = {};
    glm::mat4 m_gizmoMatStart    = glm::mat4(1.f);
    glm::vec3 m_gizmoRotRef      = glm::vec3(0.f);
    bool      m_gizmoRotRefSet   = false;

    bool drawGizmo(Scene& scene, ImDrawList* dl, ImVec2 vpOrigin, ImVec2 vpSize);

    // Pending import (set by Import OBJ button, consumed by App between frames)
    std::string m_pendingImportPath;
    std::string m_pendingImportName;

    // Loading overlay state
    std::string m_loadingStage;
    float       m_loadingProgress = 0.f;
};
