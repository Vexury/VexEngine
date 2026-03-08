#pragma once

#include "mesh_group_save.h"
#include "scene.h"  // SceneVolume is defined here

#include <glm/glm.hpp>
#include <deque>
#include <memory>
#include <string>

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

struct CmdAddMeshGroup : ICommand
{
    MeshGroupSave save;
    int           index;  // meshGroups index the group occupies after redo

    CmdAddMeshGroup(MeshGroupSave s, int idx) : save(std::move(s)), index(idx) {}
    void undo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
    void redo(Scene& s, SceneRenderer& r, EditorUI& ui) override;
};

struct CmdDeleteMeshGroup : ICommand
{
    MeshGroupSave save;
    int           index;  // original position in meshGroups

    CmdDeleteMeshGroup(MeshGroupSave s, int idx) : save(std::move(s)), index(idx) {}
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

struct CmdSetTransform : ICommand
{
    int         groupIdx;
    int         submeshIdx;   // >= 0 = specific submesh index; -1 = use objectName or group
    std::string objectName;   // non-empty = apply to all submeshes with this objectName
    glm::mat4   before;       // local matrix before (group.modelMatrix or sm.modelMatrix)
    glm::mat4   after;        // local matrix after

    CmdSetTransform(int gIdx, int sIdx, std::string objName, glm::mat4 b, glm::mat4 a)
        : groupIdx(gIdx), submeshIdx(sIdx), objectName(std::move(objName)), before(b), after(a) {}
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
