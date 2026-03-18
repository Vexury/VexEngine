#include "command.h"
#include "scene.h"
#include "scene_renderer.h"
#include "editor_ui.h"

#include <vex/core/log.h>
#include <vex/raytracing/bvh.h>

#include <algorithm>

// ── Helper: detach a node from its parent's childIndices ─────────────────────

static void detachFromParent(Scene& s, int nodeIdx)
{
    int parentIdx = s.nodes[nodeIdx].parentIndex;
    if (parentIdx < 0) return;  // already a root
    auto& parentChildren = s.nodes[parentIdx].childIndices;
    parentChildren.erase(
        std::remove(parentChildren.begin(), parentChildren.end(), nodeIdx),
        parentChildren.end());
    s.nodes[nodeIdx].parentIndex = -1;
}

static void attachToParent(Scene& s, int nodeIdx, int parentIdx, int siblingPos)
{
    s.nodes[nodeIdx].parentIndex = parentIdx;
    if (parentIdx < 0) return;  // root — no parent childIndices to update
    auto& parentChildren = s.nodes[parentIdx].childIndices;
    if (siblingPos < 0 || siblingPos >= (int)parentChildren.size())
        parentChildren.push_back(nodeIdx);
    else
        parentChildren.insert(parentChildren.begin() + siblingPos, nodeIdx);
}

// ── CmdAddNode ────────────────────────────────────────────────────────────────

void CmdAddNode::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    s.addNodeFromSave(save, insertionIndex);
    // addNodeFromSave inserts at insertionIndex (or appends if out of range).
    // Update insertionIndex to reflect actual position.
    insertionIndex = (insertionIndex >= 0 && insertionIndex < (int)s.nodes.size() - 1)
        ? insertionIndex
        : static_cast<int>(s.nodes.size()) - 1;

    // Link into parent's childIndices
    if (parentIndex >= 0 && parentIndex < (int)s.nodes.size())
    {
        auto& parentChildren = s.nodes[parentIndex].childIndices;
        if (siblingPos < 0 || siblingPos >= (int)parentChildren.size())
            parentChildren.push_back(insertionIndex);
        else
            parentChildren.insert(parentChildren.begin() + siblingPos, insertionIndex);
        s.nodes[insertionIndex].parentIndex = parentIndex;
    }

    ui.setSelection(Selection::Mesh, insertionIndex);
}

void CmdAddNode::undo(Scene& s, SceneRenderer& r, EditorUI& ui)
{
    if (insertionIndex < 0 || insertionIndex >= (int)s.nodes.size())
    {
        ui.clearSelection();
        return;
    }
    r.waitIdle();
    vex::Log::info("Undo add node: " + s.nodes[insertionIndex].name);

    // Remove from parent's childIndices
    detachFromParent(s, insertionIndex);

    s.nodes.erase(s.nodes.begin() + insertionIndex);
    fixRefsAfterRemove(s, insertionIndex);
    s.geometryDirty = true;
    ui.clearSelection();
}

// ── CmdDeleteNode ─────────────────────────────────────────────────────────────

NodeSave CmdDeleteNode::captureNode(const Scene& scene, int nodeIdx)
{
    const auto& node = scene.nodes[nodeIdx];
    NodeSave save;
    save.name         = node.name;
    save.center       = node.center;
    save.radius       = node.radius;
    save.localMatrix  = node.localMatrix;
    save.parentIndex  = node.parentIndex;
    save.childIndices = node.childIndices;
    for (const auto& sm : node.submeshes)
    {
        SubmeshSave ss;
        ss.name        = sm.name;
        ss.meshData    = sm.meshData;
        ss.modelMatrix = sm.modelMatrix;
        save.submeshes.push_back(std::move(ss));
    }
    return save;
}

CmdDeleteNode::CmdDeleteNode(const Scene& scene, int rootNodeIdx)
{
    // Collect subtree in descending order (for redo erase), but save in ascending order (for undo insert)
    std::vector<int> descending = collectSubtree(scene, rootNodeIdx);

    // Convert to ascending for saves
    originalIndices = descending;
    std::sort(originalIndices.begin(), originalIndices.end());

    subtreeSaves.resize(originalIndices.size());
    for (int i = 0; i < (int)originalIndices.size(); ++i)
        subtreeSaves[i] = captureNode(scene, originalIndices[i]);

    rootParentIdx  = scene.nodes[rootNodeIdx].parentIndex;
    rootSiblingPos = 0;
    if (rootParentIdx >= 0)
    {
        const auto& pc = scene.nodes[rootParentIdx].childIndices;
        for (int i = 0; i < (int)pc.size(); ++i)
            if (pc[i] == rootNodeIdx) { rootSiblingPos = i; break; }
    }
}

void CmdDeleteNode::redo(Scene& s, SceneRenderer& r, EditorUI& ui)
{
    if (originalIndices.empty()) { ui.clearSelection(); return; }
    r.waitIdle();

    int rootNodeIdx = originalIndices[0];  // ascending: first = lowest = root
    vex::Log::info("Deleted node: " + s.nodes[rootNodeIdx].name);

    // Remove root from parent's childIndices
    detachFromParent(s, rootNodeIdx);

    // Erase in descending order
    std::vector<int> desc = originalIndices;
    std::sort(desc.rbegin(), desc.rend());
    for (int idx : desc)
    {
        s.nodes.erase(s.nodes.begin() + idx);
        fixRefsAfterRemove(s, idx);
    }

    s.geometryDirty = true;
    ui.clearSelection();
}

void CmdDeleteNode::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& ui)
{
    // Insert all subtree nodes back in ascending order.
    // After all insertions, re-link parentIndex/childIndices from saves
    // (fixRefsAfterInsert inside addNodeFromSave shifts existing nodes' refs,
    //  but the restored nodes' refs need to be set explicitly from saves).
    for (int i = 0; i < (int)originalIndices.size(); ++i)
        s.addNodeFromSave(subtreeSaves[i], originalIndices[i]);

    // Re-link: overwrite parentIndex/childIndices for all restored nodes from saves
    for (int i = 0; i < (int)originalIndices.size(); ++i)
    {
        auto& node = s.nodes[originalIndices[i]];
        node.parentIndex  = subtreeSaves[i].parentIndex;
        node.childIndices = subtreeSaves[i].childIndices;
    }

    // Re-attach root to its original parent's childIndices
    int rootNodeIdx = originalIndices[0];
    if (rootParentIdx >= 0 && rootParentIdx < (int)s.nodes.size())
    {
        auto& parentChildren = s.nodes[rootParentIdx].childIndices;
        // Ensure not already present (re-link above restored it from save; but parent's save
        // may not include this child — parent is NOT a subtree node)
        bool already = false;
        for (int c : parentChildren) if (c == rootNodeIdx) { already = true; break; }
        if (!already)
        {
            if (rootSiblingPos < (int)parentChildren.size())
                parentChildren.insert(parentChildren.begin() + rootSiblingPos, rootNodeIdx);
            else
                parentChildren.push_back(rootNodeIdx);
        }
        s.nodes[rootNodeIdx].parentIndex = rootParentIdx;
    }

    s.geometryDirty = true;
    ui.setSelection(Selection::Mesh, rootNodeIdx);
}

// ── CmdReparent ──────────────────────────────────────────────────────────────

void CmdReparent::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    if (nodeIdx < 0 || nodeIdx >= (int)s.nodes.size()) return;

    detachFromParent(s, nodeIdx);
    s.nodes[nodeIdx].localMatrix = newLocalMatrix;
    attachToParent(s, nodeIdx, newParentIdx, -1);  // append to new parent's children

    s.geometryDirty = true;
}

void CmdReparent::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    if (nodeIdx < 0 || nodeIdx >= (int)s.nodes.size()) return;

    detachFromParent(s, nodeIdx);
    s.nodes[nodeIdx].localMatrix = oldLocalMatrix;
    attachToParent(s, nodeIdx, oldParentIdx, oldSiblingPos);

    s.geometryDirty = true;
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

void CmdSetTransform::redo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    if (nodeIdx < 0 || nodeIdx >= (int)s.nodes.size()) return;
    s.nodes[nodeIdx].localMatrix = after;
    s.geometryDirty = true;
}

void CmdSetTransform::undo(Scene& s, SceneRenderer& /*r*/, EditorUI& /*ui*/)
{
    if (nodeIdx < 0 || nodeIdx >= (int)s.nodes.size()) return;
    s.nodes[nodeIdx].localMatrix = before;
    s.geometryDirty = true;
}
