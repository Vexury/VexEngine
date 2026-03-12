#include "scene.h"

#include <vex/scene/mesh_data.h>
#include <vex/graphics/mesh.h>
#include <vex/core/log.h>

#include <algorithm>
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

// ── getWorldMatrix ────────────────────────────────────────────────────────────

glm::mat4 Scene::getWorldMatrix(int nodeIdx) const
{
    if (nodeIdx < 0 || nodeIdx >= (int)nodes.size()) return glm::mat4(1.0f);
    const SceneNode& node = nodes[nodeIdx];
    if (node.parentIndex < 0) return node.localMatrix;
    return getWorldMatrix(node.parentIndex) * node.localMatrix;
}

// ── Index fixup helpers ───────────────────────────────────────────────────────

void fixRefsAfterRemove(Scene& scene, int removedIdx)
{
    for (auto& n : scene.nodes)
    {
        if (n.parentIndex > removedIdx) --n.parentIndex;
        for (auto& c : n.childIndices)
            if (c > removedIdx) --c;
    }
}

void fixRefsAfterInsert(Scene& scene, int insertedIdx)
{
    for (auto& n : scene.nodes)
    {
        if (n.parentIndex >= insertedIdx) ++n.parentIndex;
        for (auto& c : n.childIndices)
            if (c >= insertedIdx) ++c;
    }
}

std::vector<int> collectSubtree(const Scene& scene, int nodeIdx)
{
    std::vector<int> result;
    std::vector<int> stack = { nodeIdx };
    while (!stack.empty())
    {
        int cur = stack.back();
        stack.pop_back();
        result.push_back(cur);
        for (int child : scene.nodes[cur].childIndices)
            stack.push_back(child);
    }
    std::sort(result.rbegin(), result.rend()); // descending — safe erase order
    return result;
}

// ── importOBJ ────────────────────────────────────────────────────────────────

bool Scene::importOBJ(const std::string& path, const std::string& name,
                      ProgressFn onProgress)
{
    auto submeshes = vex::MeshData::loadOBJ(path);
    if (submeshes.empty())
        return false;

    vex::Log::info("Uploading " + std::to_string(submeshes.size()) + " submesh(es) to GPU...");

    // Collect ordered unique objectNames to decide whether to create a hierarchy
    std::vector<std::string> objNames;
    for (const auto& md : submeshes)
        if (std::find(objNames.begin(), objNames.end(), md.objectName) == objNames.end())
            objNames.push_back(md.objectName);

    TexCache texCache;
    int texCount = 0;

    if (onProgress) onProgress("Uploading meshes and textures...", 0.3f);

    // Helper: upload one vex::MeshData into a SceneMesh
    auto makeSM = [&](size_t i) -> SceneMesh
    {
        auto mesh = vex::Mesh::create();
        mesh->upload(submeshes[i]);
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
        sm.vertexCount      = static_cast<uint32_t>(submeshes[i].vertices.size());
        sm.indexCount       = static_cast<uint32_t>(submeshes[i].indices.size());
        return sm;
    };

    vex::Mesh::beginBatchUpload();

    if (objNames.size() <= 1)
    {
        // Single object (or no objectName tags) — one flat node
        SceneNode node;
        node.name        = name;
        node.parentIndex = -1;

        glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
        for (size_t i = 0; i < submeshes.size(); ++i)
        {
            for (const auto& v : submeshes[i].vertices)
            {
                bboxMin = glm::min(bboxMin, v.position);
                bboxMax = glm::max(bboxMax, v.position);
            }
            node.submeshes.push_back(makeSM(i));
        }
        vex::Mesh::endBatchUpload();
        node.center = (bboxMin + bboxMax) * 0.5f;
        node.radius = glm::length(bboxMax - bboxMin) * 0.5f;
        nodes.push_back(std::move(node));
    }
    else
    {
        // Multiple named objects — root node + one child per objectName
        int rootIdx = static_cast<int>(nodes.size());

        SceneNode root;
        root.name        = name;
        root.parentIndex = -1;

        glm::vec3 rootBBoxMin(FLT_MAX), rootBBoxMax(-FLT_MAX);

        std::vector<SceneNode> children(objNames.size());
        // Per-child bbox accumulators (proper min/max, no heuristic)
        std::vector<glm::vec3> childBBoxMin(objNames.size(), glm::vec3( FLT_MAX));
        std::vector<glm::vec3> childBBoxMax(objNames.size(), glm::vec3(-FLT_MAX));

        for (size_t oi = 0; oi < objNames.size(); ++oi)
        {
            children[oi].name        = objNames[oi].empty()
                ? "Object" + std::to_string(oi) : objNames[oi];
            children[oi].parentIndex = rootIdx;
            root.childIndices.push_back(rootIdx + 1 + static_cast<int>(oi));
        }

        for (size_t i = 0; i < submeshes.size(); ++i)
        {
            // Find which child this submesh belongs to
            int oi = static_cast<int>(
                std::find(objNames.begin(), objNames.end(), submeshes[i].objectName)
                - objNames.begin());

            for (const auto& v : submeshes[i].vertices)
            {
                childBBoxMin[oi] = glm::min(childBBoxMin[oi], v.position);
                childBBoxMax[oi] = glm::max(childBBoxMax[oi], v.position);
                rootBBoxMin      = glm::min(rootBBoxMin,      v.position);
                rootBBoxMax      = glm::max(rootBBoxMax,      v.position);
            }
            children[oi].submeshes.push_back(makeSM(i));
        }
        vex::Mesh::endBatchUpload();

        for (size_t oi = 0; oi < objNames.size(); ++oi)
        {
            if (childBBoxMin[oi].x <= childBBoxMax[oi].x) // has vertices
            {
                children[oi].center = (childBBoxMin[oi] + childBBoxMax[oi]) * 0.5f;
                children[oi].radius = glm::length(childBBoxMax[oi] - childBBoxMin[oi]) * 0.5f;
            }
        }

        root.center = (rootBBoxMin + rootBBoxMax) * 0.5f;
        root.radius = glm::length(rootBBoxMax - rootBBoxMin) * 0.5f;

        nodes.push_back(std::move(root));
        for (auto& child : children)
            nodes.push_back(std::move(child));
    }

    geometryDirty = true;

    if (texCount > 0)
        vex::Log::info("  Loaded " + std::to_string(texCount) + " unique texture(s)"
                       + " (shared across " + std::to_string(submeshes.size()) + " submeshes)");

    return true;
}

// ── addNodeFromSave ───────────────────────────────────────────────────────────

void Scene::addNodeFromSave(const NodeSave& save, int insertAt)
{
    SceneNode node;
    node.name         = save.name;
    node.center       = save.center;
    node.radius       = save.radius;
    node.localMatrix  = save.localMatrix;
    node.parentIndex  = save.parentIndex;
    node.childIndices = save.childIndices;

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
        node.submeshes.push_back(std::move(sm));
    }
    vex::Mesh::endBatchUpload();

    if (insertAt >= 0 && insertAt < (int)nodes.size())
    {
        fixRefsAfterInsert(*this, insertAt);
        nodes.insert(nodes.begin() + insertAt, std::move(node));
    }
    else
    {
        nodes.push_back(std::move(node));
    }
    geometryDirty = true;
}
