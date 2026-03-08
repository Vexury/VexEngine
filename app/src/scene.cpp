#include "scene.h"

#include <vex/scene/mesh_data.h>
#include <vex/graphics/mesh.h>
#include <vex/core/log.h>

#include <cfloat>
#include <unordered_map>

// ── Shared texture loader ─────────────────────────────────────────────────────

using TexCache = std::unordered_map<std::string, std::shared_ptr<vex::Texture2D>>;

static std::shared_ptr<vex::Texture2D> loadTex(const std::string& p, TexCache& cache, int& count)
{
    if (p.empty()) return nullptr;
    auto it = cache.find(p);
    if (it != cache.end()) return it->second;
    auto t = vex::Texture2D::createFromFile(p);
    if (t) ++count;
    return cache[p] = std::shared_ptr<vex::Texture2D>(std::move(t));
}

// ── importOBJ ────────────────────────────────────────────────────────────────

bool Scene::importOBJ(const std::string& path, const std::string& name,
                      ProgressFn onProgress)
{
    auto submeshes = vex::MeshData::loadOBJ(path);
    if (submeshes.empty())
        return false;

    vex::Log::info("Uploading " + std::to_string(submeshes.size()) + " submesh(es) to GPU...");

    MeshGroup group;
    group.name = name;

    TexCache texCache;
    int texCount = 0;

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
        sm.diffuseTexture   = loadTex(submeshes[i].diffuseTexturePath,   texCache, texCount);
        sm.normalTexture    = loadTex(submeshes[i].normalTexturePath,     texCache, texCount);
        sm.roughnessTexture = loadTex(submeshes[i].roughnessTexturePath,  texCache, texCount);
        sm.metallicTexture  = loadTex(submeshes[i].metallicTexturePath,   texCache, texCount);
        sm.emissiveTexture  = loadTex(submeshes[i].emissiveTexturePath,   texCache, texCount);
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

// ── addMeshGroupFromSave ──────────────────────────────────────────────────────

void Scene::addMeshGroupFromSave(const MeshGroupSave& save, int insertAt)
{
    MeshGroup group;
    group.name        = save.name;
    group.center      = save.center;
    group.radius      = save.radius;
    group.modelMatrix = save.modelMatrix;

    TexCache texCache;
    int texCount = 0;

    vex::Mesh::beginBatchUpload();
    for (size_t i = 0; i < save.submeshes.size(); ++i)
    {
        const auto& ss = save.submeshes[i];

        auto mesh = vex::Mesh::create();
        mesh->upload(ss.meshData);

        SceneMesh sm;
        sm.name             = ss.name;
        sm.mesh             = std::move(mesh);
        sm.diffuseTexture   = loadTex(ss.meshData.diffuseTexturePath,   texCache, texCount);
        sm.normalTexture    = loadTex(ss.meshData.normalTexturePath,     texCache, texCount);
        sm.roughnessTexture = loadTex(ss.meshData.roughnessTexturePath,  texCache, texCount);
        sm.metallicTexture  = loadTex(ss.meshData.metallicTexturePath,   texCache, texCount);
        sm.emissiveTexture  = loadTex(ss.meshData.emissiveTexturePath,   texCache, texCount);
        sm.meshData         = ss.meshData;
        sm.modelMatrix      = ss.modelMatrix;
        sm.vertexCount      = static_cast<uint32_t>(ss.meshData.vertices.size());
        sm.indexCount       = static_cast<uint32_t>(ss.meshData.indices.size());
        group.submeshes.push_back(std::move(sm));
    }
    vex::Mesh::endBatchUpload();

    if (insertAt < 0 || insertAt >= static_cast<int>(meshGroups.size()))
    {
        meshGroups.push_back(std::move(group));
    }
    else
    {
        meshGroups.insert(meshGroups.begin() + insertAt, std::move(group));
    }
    geometryDirty = true;
}
