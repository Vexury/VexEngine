#pragma once

#include "mesh_group_save.h"
#include "scene.h"  // SceneVolume is defined here

#include <glm/glm.hpp>
#include <deque>
#include <memory>
#include <string>
#include <vector>

struct Scene;
class SceneRenderer;
class EditorUI;

// ── Command interface ────────────────────────────────────────────────────────

struct ICommand
{
    virtual ~ICommand() = default;
    virtual void undo(Scene& s, SceneRenderer& r, EditorUI& ui) = 0;
    virtual void redo(Scene& s, SceneRenderer& r, EditorUI& ui) = 0;
};

// ── Concrete commands ────────────────────────────────────────────────────────

// Adds a single node (no children) at insertionIndex.
struct CmdAddNode : ICommand
{
    NodeSave save;
    int      insertionIndex;  // index in scene.nodes after redo
    int      parentIndex;     // -1 if root; parent is NOT in save (it's an existing node)
    int      siblingPos;      // insertion position in parent's childIndices (-1 = append)

    CmdAddNode(NodeSave s, int idx, int parent = -1, int sibPos = -1)
        : save(std::move(s)), insertionIndex(idx), parentIndex(parent), siblingPos(sibPos) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

// Deletes a subtree rooted at a node. Saves entire subtree for undo.
struct CmdDeleteNode : ICommand
{
    std::vector<NodeSave> subtreeSaves;      // one per node, in ascending originalIndex order
    std::vector<int>      originalIndices;   // ascending originalIndex order (matches subtreeSaves)
    int                   rootParentIdx;     // parent of root before deletion (-1 = root of scene)
    int                   rootSiblingPos;    // position of root in parent's childIndices

    CmdDeleteNode(const Scene& scene, int rootNodeIdx);
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;

private:
    static NodeSave captureNode(const Scene& scene, int nodeIdx);
};

// Reparents a node (no flat-vector changes, only parentIndex/childIndices + localMatrix).
struct CmdReparent : ICommand
{
    int       nodeIdx;
    int       oldParentIdx;     // -1 if was root
    int       newParentIdx;     // -1 if becoming root
    int       oldSiblingPos;    // position in old parent's childIndices (for undo)
    glm::mat4 oldLocalMatrix;
    glm::mat4 newLocalMatrix;   // preserves world position: inv(parent.world) * node.world

    CmdReparent(int node, int oldPar, int newPar, int oldSibPos,
                glm::mat4 oldLocal, glm::mat4 newLocal)
        : nodeIdx(node), oldParentIdx(oldPar), newParentIdx(newPar),
          oldSiblingPos(oldSibPos), oldLocalMatrix(oldLocal), newLocalMatrix(newLocal) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

struct CmdAddVolume : ICommand
{
    SceneVolume vol;
    int         index;

    CmdAddVolume(SceneVolume v, int idx) : vol(std::move(v)), index(idx) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

struct CmdDeleteVolume : ICommand
{
    SceneVolume vol;
    int         index;

    CmdDeleteVolume(SceneVolume v, int idx) : vol(std::move(v)), index(idx) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

// Undo/redo wrapper for OBJ import: swaps CmdDeleteNode's undo/redo so that
// undo() removes the imported subtree and redo() restores it.
struct CmdImportUndo : ICommand
{
    CmdDeleteNode delCmd;
    explicit CmdImportUndo(CmdDeleteNode d) : delCmd(std::move(d)) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override { delCmd.redo(s, r, ui); }
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override { delCmd.undo(s, r, ui); }
};

// Simplified transform command: stores nodeIdx + local matrix before/after.
struct CmdSetTransform : ICommand
{
    int       nodeIdx;
    glm::mat4 before;   // localMatrix before
    glm::mat4 after;    // localMatrix after

    CmdSetTransform(int idx, glm::mat4 b, glm::mat4 a)
        : nodeIdx(idx), before(b), after(a) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

// ── Command stack ────────────────────────────────────────────────────────────

class CommandStack
{
public:
    static constexpr int MAX_UNDO = 50;

    // Execute cmd immediately (calls redo), push to undo stack, clear redo stack.
    void execute(std::unique_ptr<ICommand> cmd, Scene& s, SceneRenderer& r, EditorUI& ui)
    {
        cmd->redo(s, r, ui);
        push(std::move(cmd));
    }

    // Push to undo stack without calling redo (for actions already applied externally).
    void pushUndoOnly(std::unique_ptr<ICommand> cmd)
    {
        push(std::move(cmd));
    }

    void undo(Scene& s, SceneRenderer& r, EditorUI& ui)
    {
        if (m_undoStack.empty()) return;
        auto cmd = std::move(m_undoStack.back());
        m_undoStack.pop_back();
        cmd->undo(s, r, ui);
        m_redoStack.push_back(std::move(cmd));
    }

    void redo(Scene& s, SceneRenderer& r, EditorUI& ui)
    {
        if (m_redoStack.empty()) return;
        auto cmd = std::move(m_redoStack.back());
        m_redoStack.pop_back();
        cmd->redo(s, r, ui);
        m_undoStack.push_back(std::move(cmd));
    }

    void clear()
    {
        m_undoStack.clear();
        m_redoStack.clear();
    }

    bool canUndo() const { return !m_undoStack.empty(); }
    bool canRedo() const { return !m_redoStack.empty(); }

private:
    void push(std::unique_ptr<ICommand> cmd)
    {
        m_undoStack.push_back(std::move(cmd));
        if (static_cast<int>(m_undoStack.size()) > MAX_UNDO)
            m_undoStack.pop_front();
        m_redoStack.clear();
    }

    std::deque<std::unique_ptr<ICommand>> m_undoStack;
    std::deque<std::unique_ptr<ICommand>> m_redoStack;
};
