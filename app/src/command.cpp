#include "command.h"
#include "scene.h"
#include "scene_renderer.h"
#include "editor_ui.h"

#include <vex/core/log.h>

// ── CmdAddMeshGroup ──────────────────────────────────────────────────────────

void CmdAddMeshGroup::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    s.addMeshGroupFromSave(save);
    index = static_cast<int>(s.meshGroups.size()) - 1;
    ui.setSelection(Selection::Mesh, index);
}

void CmdAddMeshGroup::undo(Scene& s, SceneRenderer& r, EditorUI& ui)
{
    if (index < 0 || index >= static_cast<int>(s.meshGroups.size()))
    {
        ui.clearSelection();
        return;
    }
    r.waitIdle();  // GPU may still reference the mesh buffers
    vex::Log::info("Undo add: " + s.meshGroups[index].name);
    s.meshGroups.erase(s.meshGroups.begin() + index);
    s.geometryDirty = true;
    ui.clearSelection();
}

// ── CmdDeleteMeshGroup ───────────────────────────────────────────────────────

void CmdDeleteMeshGroup::redo(Scene& s, SceneRenderer& r, EditorUI& ui)
{
    if (index < 0 || index >= static_cast<int>(s.meshGroups.size()))
    {
        ui.clearSelection();
        return;
    }
    r.waitIdle();
    vex::Log::info("Deleted: " + s.meshGroups[index].name);
    s.meshGroups.erase(s.meshGroups.begin() + index);
    s.geometryDirty = true;
    ui.clearSelection();
}

void CmdDeleteMeshGroup::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    s.addMeshGroupFromSave(save, index);
    ui.setSelection(Selection::Mesh, index);
}

// ── CmdAddVolume ─────────────────────────────────────────────────────────────

void CmdAddVolume::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    s.volumes.push_back(vol);
    index = static_cast<int>(s.volumes.size()) - 1;
    ui.setSelection(Selection::Volume, index);
}

void CmdAddVolume::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    if (index < 0 || index >= static_cast<int>(s.volumes.size()))
    {
        ui.clearSelection();
        return;
    }
    s.volumes.erase(s.volumes.begin() + index);
    ui.clearSelection();
}

// ── CmdDeleteVolume ──────────────────────────────────────────────────────────

void CmdDeleteVolume::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    if (index < 0 || index >= static_cast<int>(s.volumes.size()))
    {
        ui.clearSelection();
        return;
    }
    vex::Log::info("Deleted volume: " + s.volumes[index].name);
    s.volumes.erase(s.volumes.begin() + index);
    ui.clearSelection();
}

void CmdDeleteVolume::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    if (index >= 0 && index <= static_cast<int>(s.volumes.size()))
        s.volumes.insert(s.volumes.begin() + index, vol);
    else
        s.volumes.push_back(vol);
    ui.setSelection(Selection::Volume, index);
}

// ── CmdSetTransform ──────────────────────────────────────────────────────────

static void applyTransformMat(Scene& s, int groupIdx, int submeshIdx,
                              const std::string& objectName, const glm::mat4& mat)
{
    if (groupIdx < 0 || groupIdx >= static_cast<int>(s.meshGroups.size())) return;
    auto& group = s.meshGroups[groupIdx];
    if (submeshIdx >= 0 && submeshIdx < static_cast<int>(group.submeshes.size()))
    {
        // mat is world-space; sm.modelMatrix is local (relative to group)
        group.submeshes[submeshIdx].modelMatrix = glm::inverse(group.modelMatrix) * mat;
    }
    else if (!objectName.empty())
    {
        glm::mat4 localMat = glm::inverse(group.modelMatrix) * mat;
        for (auto& sm : group.submeshes)
            if (sm.meshData.objectName == objectName)
                sm.modelMatrix = localMat;
    }
    else
    {
        group.modelMatrix = mat;
    }
    s.geometryDirty = true;
}

void CmdSetTransform::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    applyTransformMat(s, groupIdx, submeshIdx, objectName, after);
}

void CmdSetTransform::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    applyTransformMat(s, groupIdx, submeshIdx, objectName, before);
}
