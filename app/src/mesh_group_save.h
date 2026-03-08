#pragma once

#include <vex/scene/mesh_data.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>

// Plain-data types used by the command system to save/restore mesh groups.
// Kept in a separate header to avoid circular includes between scene.h and command.h.

struct SubmeshSave
{
    std::string   name;
    vex::MeshData meshData;   // full CPU copy; texture paths stored inside
    glm::mat4     modelMatrix = glm::mat4(1.0f);
};

struct MeshGroupSave
{
    std::string              name;
    glm::vec3                center      = {};
    float                    radius      = 1.0f;
    glm::mat4                modelMatrix = glm::mat4(1.0f);
    std::vector<SubmeshSave> submeshes;
};
