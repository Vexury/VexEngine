#pragma once
#include <string>

enum class Selection { None, Mesh, Skybox, Light, Sun, Camera, Volume };

struct SelectionState
{
    Selection   type       = Selection::None;
    int         index      = -1;
    int         submeshIdx = -1;
    std::string objectName;

    void set(Selection t, int idx = -1, int sub = -1)
    {
        type = t; index = idx; submeshIdx = sub; objectName.clear();
    }
    void clear() { type = Selection::None; index = -1; submeshIdx = -1; objectName.clear(); }
};
