#pragma once

#include "selection.h"

#include <glm/glm.hpp>
#include <imgui.h>
#include <string>

namespace vex { class GraphicsContext; }

struct Scene;
class SceneRenderer;

enum class RenderMode;

class EditorUI
{
public:
    void renderViewport(SceneRenderer& renderer, Scene& scene);
    void renderHierarchy(Scene& scene, SceneRenderer& renderer);
    void renderInspector(Scene& scene, SceneRenderer& renderer);
    void renderSettings(SceneRenderer& renderer);
    void renderConsole();
    void renderStats(SceneRenderer& renderer, Scene& scene, vex::GraphicsContext& ctx);

    void init(SelectionState& sel) { m_selection = &sel; }

    Selection getSelectionType() const { return m_selection->type; }
    int       getSelectionIndex() const { return m_selection->index; }

    void clearSelection() { m_selection->clear(); }

    void setSelection(Selection type, int index = 0, int submesh = -1)
    {
        m_selection->set(type, index, submesh);
    }

    // Called after a viewport pick so we can resolve the object name from mesh data.
    void setSelectedObjectName(const std::string& name) { m_selection->objectName = name; }
    const std::string& getSelectedObjectName() const    { return m_selection->objectName; }

    int getSelectedMeshGroup() const
    {
        return (m_selection->type == Selection::Mesh) ? m_selection->index : -1;
    }

    int getSelectedSubmesh() const { return m_selection->submeshIdx; }

    bool isViewportHovered() const { return m_viewportHovered; }
    int  getRenderModeIndex() const { return m_renderModeIndex; }
    int  getDebugModeIndex() const { return m_debugModeIndex; }

    bool consumePickRequest(int& outX, int& outY);

    void setGizmoMode(int mode)    { m_gizmoMode  = mode;  }
    void toggleGizmoLocal()        { m_gizmoLocal = !m_gizmoLocal; }
    bool isGizmoLocal() const      { return m_gizmoLocal; }

    // Deferred import: the Import OBJ button stores the path here instead of blocking
    // the current frame. App::run() calls consumePendingImport() between frames.
    bool consumePendingImport(std::string& outPath, std::string& outName);

    // Deferred GLTF import (same pattern as OBJ)
    bool consumePendingGltfImport(std::string& outPath, std::string& outName);

    // Deferred primitive creation
    enum class PrimitiveType { None, Plane, Cube, Sphere, Cylinder };
    bool consumePendingPrimitive(PrimitiveType& outType);

    // Deferred volume add (routed through App so it can push an undo command)
    bool consumePendingAddVolume();

    // Deferred duplicate (set by Duplicate button or Ctrl+D)
    bool consumePendingDuplicate();

    // Gizmo drag-end commit — consumed by App to push CmdSetTransform
    struct TransformCommit {
        int       nodeIdx;
        glm::mat4 before;  // localMatrix at drag start
        glm::mat4 after;   // localMatrix at drag end
    };
    bool consumeTransformCommit(TransformCommit& out);

    // Deferred reparent (set by hierarchy drag-and-drop, consumed by App between frames)
    struct PendingReparent { int nodeIdx; int newParentIdx; };  // newParentIdx=-1 = make root
    bool consumePendingReparent(PendingReparent& out);

    // Deferred env map load — must happen before beginFrame() to avoid destroying
    // a VkSampler while the current frame's command buffer has it bound.
    bool consumePendingEnvLoad(std::string& outPath);

    // Ancestor check (used to prevent parenting a node to one of its own descendants)
    bool isAncestorOf(const Scene& scene, int potentialAncestor, int node) const;

    // Loading overlay: called by App::runImport() to show progress between frames.
    void setLoadingState(const std::string& stage, float progress);
    void clearLoadingState();
    void renderLoadingOverlay();

private:
    SelectionState* m_selection = nullptr;

    bool m_viewportHovered = false;

    bool m_pickRequested = false;
    int  m_pickX = 0;
    int  m_pickY = 0;

    int m_renderModeIndex = 0;
    int m_debugModeIndex = 0;
    int m_prevEnvmapForRevert = 0;

    // Gizmo state
    int       m_gizmoMode     = 0;     // 0=Translate  1=Rotate  2=Scale
    int       m_gizmoAxis     = -1;    // 0=X 1=Y 2=Z  -1=none active
    bool      m_gizmoLocal    = true;  // true=local space, false=world space
    bool      m_gizmoDragging = false;
    ImVec2    m_gizmoDragStart   = {};
    glm::mat4 m_gizmoMatStart    = glm::mat4(1.f);
    glm::vec3 m_gizmoPivot       = glm::vec3(0.f); // world-space pivot frozen at drag start
    glm::vec3 m_gizmoRotRef      = glm::vec3(0.f);
    bool      m_gizmoRotRefSet   = false;

    bool drawGizmo(Scene& scene, ImDrawList* dl, ImVec2 vpOrigin, ImVec2 vpSize);
    void drawHierarchyNode(int nodeIdx, Scene& scene);

    // Pending import (set by Import OBJ button, consumed by App between frames)
    std::string m_pendingImportPath;
    std::string m_pendingImportName;

    // Pending GLTF import (set by Import GLTF button, consumed by App between frames)
    std::string m_pendingGltfImportPath;
    std::string m_pendingGltfImportName;

    // Pending deferred actions
    PrimitiveType m_pendingPrimitive  = PrimitiveType::None;
    bool          m_pendingAddVolume  = false;
    bool          m_pendingDuplicate  = false;

    // Pending env map load (set by inspector, consumed by App before beginFrame)
    std::string m_pendingEnvLoadPath;

    // Gizmo transform commit
    bool            m_transformCommitReady = false;
    TransformCommit m_transformCommit      = {};

    // Gizmo local matrix snapshot at drag start (for TransformCommit.before)
    glm::mat4 m_gizmoLocalStart = glm::mat4(1.f);

    // Pending reparent (set by hierarchy drag-and-drop, consumed by App between frames)
    bool           m_pendingReparentReady = false;
    PendingReparent m_pendingReparent      = {};

    // Loading overlay state
    std::string m_loadingStage;
    float       m_loadingProgress = 0.f;

    // Cached per-submesh scene stats — recomputed only when the scene changes
    struct CachedSceneStats
    {
        uint32_t totalVerts        = 0;
        uint32_t totalIndices      = 0;
        int      totalSubs         = 0;
        int      emissiveMeshCount = 0;
        int      uniqueTextureCount = 0;
        int      cachedNodeCount   = -1;  // sentinel: -1 = never computed
    };
    CachedSceneStats m_sceneStats;
};
