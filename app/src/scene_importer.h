#pragma once

#include "scene.h"
#include "mesh_group_save.h"

#include <functional>
#include <string>

// Import and GPU-upload logic for Scene.
// Kept separate from scene.h/cpp so Scene stays a pure data container.
namespace SceneImporter
{
    using ProgressFn = std::function<void(const std::string& stage, float progress)>;

    bool importOBJ (Scene& scene, const std::string& path, const std::string& name,
                    ProgressFn onProgress = nullptr);
    bool importGLTF(Scene& scene, const std::string& path, const std::string& name,
                    ProgressFn onProgress = nullptr);

    // Recreate GPU resources from a CPU save and insert the node into the scene.
    // insertAt = -1 → append; otherwise inserts at that index.
    void addNodeFromSave(Scene& scene, const NodeSave& save, int insertAt = -1);

    // Parallel stbi_load of all unique texture paths referenced by current scene nodes.
    // Results land in scene.importedTexPixels; consumed (and cleared) by
    // SceneGeometryCache::rebuild() to avoid a second disk read per texture.
    void prefetchTextures(Scene& scene);
}
