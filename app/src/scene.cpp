#include "scene.h"

#include <vex/scene/mesh_data.h>
#include <vex/graphics/mesh.h>
#include <vex/core/log.h>

#include <cfloat>
#include <unordered_map>

bool Scene::importOBJ(const std::string& path, const std::string& name,
                      ProgressFn onProgress)
{
    auto submeshes = vex::MeshData::loadOBJ(path);
    if (submeshes.empty())
        return false;

    vex::Log::info("Uploading " + std::to_string(submeshes.size()) + " submesh(es) to GPU...");

    MeshGroup group;
    group.name = name;

    // Texture cache: each unique path is loaded once and shared across submeshes.
    std::unordered_map<std::string, std::shared_ptr<vex::Texture2D>> texCache;
    int texCount = 0;

    auto loadTex = [&](const std::string& p) -> std::shared_ptr<vex::Texture2D>
    {
        if (p.empty()) return nullptr;
        auto it = texCache.find(p);
        if (it != texCache.end()) return it->second;
        auto t = vex::Texture2D::createFromFile(p);
        if (t) ++texCount;
        auto sp = std::shared_ptr<vex::Texture2D>(std::move(t));
        texCache[p] = sp;
        return sp;
    };

    if (onProgress) onProgress("Uploading meshes and textures...", 0.3f);

    glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
    vex::Mesh::beginBatchUpload();
    for (size_t i = 0; i < submeshes.size(); ++i)
    {
        for (const auto& v : submeshes[i].vertices)
        {
            bboxMin = glm::min(bboxMin, v.position);
            bboxMax = glm::max(bboxMax, v.position);
        }

        auto mesh = vex::Mesh::create();
        mesh->upload(submeshes[i]);

        uint32_t vc = static_cast<uint32_t>(submeshes[i].vertices.size());
        uint32_t ic = static_cast<uint32_t>(submeshes[i].indices.size());

        SceneMesh sm;
        sm.name = submeshes[i].name.empty()
            ? "Submesh " + std::to_string(i)
            : submeshes[i].name;
        sm.mesh             = std::move(mesh);
        sm.diffuseTexture   = loadTex(submeshes[i].diffuseTexturePath);
        sm.normalTexture    = loadTex(submeshes[i].normalTexturePath);
        sm.roughnessTexture = loadTex(submeshes[i].roughnessTexturePath);
        sm.metallicTexture  = loadTex(submeshes[i].metallicTexturePath);
        sm.emissiveTexture  = loadTex(submeshes[i].emissiveTexturePath);
        sm.meshData         = submeshes[i];
        sm.vertexCount      = vc;
        sm.indexCount       = ic;
        group.submeshes.push_back(std::move(sm));
    }
    vex::Mesh::endBatchUpload();
    group.center = (bboxMin + bboxMax) * 0.5f;
    group.radius = glm::length(bboxMax - bboxMin) * 0.5f;
    meshGroups.push_back(std::move(group));
    geometryDirty = true;

    if (texCount > 0)
        vex::Log::info("  Loaded " + std::to_string(texCount) + " unique texture(s)"
                       + " (shared across " + std::to_string(submeshes.size()) + " submeshes)");

    return true;
}
