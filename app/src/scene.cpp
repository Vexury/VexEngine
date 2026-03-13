#include "scene.h"

#include <vex/scene/mesh_data.h>
#include <vex/graphics/mesh.h>
#include <vex/core/log.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>

#include <stb_image.h>

// ── Shared texture loader ─────────────────────────────────────────────────────

using TexCache = std::unordered_map<std::string, std::shared_ptr<vex::Texture2D>>;

// Load a texture, uploading it to GPU and caching CPU-side pixels in pixCache so
// rebuildRaytraceGeometry can reuse them without a second stbi_load.
// EXR files fall back to createFromFile (no pixel cache — rare in practice).
static std::shared_ptr<vex::Texture2D> loadTex(const std::string& p, TexCache& cache, int& count,
                                                std::unordered_map<std::string, TexPixels>& pixCache)
{
    if (p.empty()) return nullptr;
    auto it = cache.find(p);
    if (it != cache.end()) return it->second;

    // EXR: delegate entirely to createFromFile (no pixel cache support needed)
    if (p.size() >= 4 &&
        (p.compare(p.size() - 4, 4, ".exr") == 0 ||
         p.compare(p.size() - 4, 4, ".EXR") == 0))
    {
        auto t = vex::Texture2D::createFromFile(p);
        if (t) ++count;
        return cache[p] = std::shared_ptr<vex::Texture2D>(std::move(t));
    }

    // Fast path: parallel decode already filled pixCache — skip disk I/O entirely.
    {
        auto preIt = pixCache.find(p);
        if (preIt != pixCache.end() && !preIt->second.pixels.empty())
        {
            const auto& tp = preIt->second;
            int rowBytes = tp.width * 4;
            std::vector<uint8_t> flipped(tp.pixels); // copy; pixCache must stay unflipped for CPU RT
            std::vector<uint8_t> row(static_cast<size_t>(rowBytes));
            for (int r = 0; r < tp.height / 2; ++r)
            {
                uint8_t* top = flipped.data() + r               * rowBytes;
                uint8_t* bot = flipped.data() + (tp.height-1-r) * rowBytes;
                std::memcpy(row.data(), top, rowBytes);
                std::memcpy(top, bot, rowBytes);
                std::memcpy(bot, row.data(), rowBytes);
            }
            auto tex = vex::Texture2D::create(static_cast<uint32_t>(tp.width),
                                               static_cast<uint32_t>(tp.height), 4);
            tex->setData(flipped.data(), static_cast<uint32_t>(tp.width),
                         static_cast<uint32_t>(tp.height), 4);
            ++count;
            return cache[p] = std::shared_ptr<vex::Texture2D>(std::move(tex));
        }
    }

    // Slow path: decode from disk (only reached if parallel pre-decode didn't run, e.g. addNodeFromSave).
    int tw, th, tch;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* data = stbi_load(p.c_str(), &tw, &th, &tch, 4);
    if (!data)
        return cache[p] = nullptr;

    // Cache the unflipped pixels for reuse by rebuildRaytraceGeometry.
    {
        TexPixels tp;
        tp.width  = tw;
        tp.height = th;
        tp.pixels.assign(data, data + static_cast<size_t>(tw) * th * 4);
        pixCache[p] = std::move(tp);
    }

    // Flip rows in-place for GPU upload (GPU conventions expect stbi_flip=true order).
    {
        int rowBytes = tw * 4;
        std::vector<uint8_t> row(rowBytes);
        for (int r = 0; r < th / 2; ++r)
        {
            uint8_t* top = data + r           * rowBytes;
            uint8_t* bot = data + (th-1 - r) * rowBytes;
            std::memcpy(row.data(), top, rowBytes);
            std::memcpy(top, bot, rowBytes);
            std::memcpy(bot, row.data(), rowBytes);
        }
    }

    auto tex = vex::Texture2D::create(static_cast<uint32_t>(tw),
                                       static_cast<uint32_t>(th), 4);
    tex->setData(data, static_cast<uint32_t>(tw), static_cast<uint32_t>(th), 4);
    stbi_image_free(data);
    ++count;
    return cache[p] = std::shared_ptr<vex::Texture2D>(std::move(tex));
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

    // ── Parallel texture decode ───────────────────────────────────────────────
    // Collect unique non-EXR paths, then stbi_load them on multiple threads.
    // Results land in importedTexPixels; loadTex fast-path skips disk I/O for
    // all of these, leaving only the GPU setData calls on the main thread.
    {
        // Collect unique paths (skip EXRs — they use createFromFile)
        std::vector<std::string> uniquePaths;
        {
            std::unordered_map<std::string, bool> seen;
            auto addPath = [&](const std::string& p)
            {
                if (p.empty() || seen.count(p)) return;
                if (p.size() >= 4 &&
                    (p.compare(p.size()-4, 4, ".exr") == 0 ||
                     p.compare(p.size()-4, 4, ".EXR") == 0)) return;
                seen[p] = true;
                uniquePaths.push_back(p);
            };
            for (const auto& md : submeshes)
            {
                addPath(md.diffuseTexturePath);
                addPath(md.normalTexturePath);
                addPath(md.roughnessTexturePath);
                addPath(md.metallicTexturePath);
                addPath(md.emissiveTexturePath);
            }
        }

        if (!uniquePaths.empty())
        {
            struct Slot { std::string path; TexPixels pixels; };
            std::vector<Slot> slots(uniquePaths.size());
            for (size_t i = 0; i < uniquePaths.size(); ++i)
                slots[i].path = uniquePaths[i];

            // stbi_set_flip_vertically_on_load writes a global — set once here;
            // worker threads only call stbi_load (read side of that global).
            stbi_set_flip_vertically_on_load(false);

            const unsigned int hw = std::thread::hardware_concurrency();
            const int nThreads = static_cast<int>(
                std::min(static_cast<size_t>(hw ? hw : 4), uniquePaths.size()));

            auto t_decode = std::chrono::steady_clock::now();

            std::atomic<int> nextSlot{0};
            auto workerFn = [&]()
            {
                for (int idx = nextSlot.fetch_add(1, std::memory_order_relaxed);
                     idx < static_cast<int>(slots.size());
                     idx = nextSlot.fetch_add(1, std::memory_order_relaxed))
                {
                    int w, h, c;
                    unsigned char* data = stbi_load(slots[idx].path.c_str(), &w, &h, &c, 4);
                    if (data)
                    {
                        slots[idx].pixels.width  = w;
                        slots[idx].pixels.height = h;
                        slots[idx].pixels.pixels.assign(
                            data, data + static_cast<size_t>(w) * h * 4);
                        stbi_image_free(data);
                    }
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(nThreads);
            for (int t = 0; t < nThreads; ++t)
                workers.emplace_back(workerFn);
            for (auto& t : workers)
                t.join();

            float t_decode_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t_decode).count();

            // Move decoded pixels into importedTexPixels (main thread, no races)
            int decoded = 0;
            for (auto& slot : slots)
                if (!slot.pixels.pixels.empty())
                {
                    importedTexPixels[slot.path] = std::move(slot.pixels);
                    ++decoded;
                }

            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "  Parallel texture decode: %.0f ms  (%d/%zu textures, %d threads)",
                t_decode_ms, decoded, uniquePaths.size(), nThreads);
            vex::Log::info(buf);
        }
    }
    // ─────────────────────────────────────────────────────────────────────────

    auto t_gpu_upload = std::chrono::steady_clock::now();

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
        sm.diffuseTexture   = loadTex(submeshes[i].diffuseTexturePath,   texCache, texCount, importedTexPixels);
        sm.normalTexture    = loadTex(submeshes[i].normalTexturePath,     texCache, texCount, importedTexPixels);
        sm.roughnessTexture = loadTex(submeshes[i].roughnessTexturePath,  texCache, texCount, importedTexPixels);
        sm.metallicTexture  = loadTex(submeshes[i].metallicTexturePath,   texCache, texCount, importedTexPixels);
        sm.emissiveTexture  = loadTex(submeshes[i].emissiveTexturePath,   texCache, texCount, importedTexPixels);
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

    {
        float t_gpu_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_gpu_upload).count();
        size_t totalVerts = 0;
        for (const auto& md : submeshes) totalVerts += md.vertices.size();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  GPU mesh upload: %.0f ms  (%zu submeshes, %zu verts, %d textures)",
            t_gpu_ms, submeshes.size(), totalVerts, texCount);
        vex::Log::info(buf);
    }

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
        sm.diffuseTexture   = loadTex(ss.meshData.diffuseTexturePath,   texCache, texCount, importedTexPixels);
        sm.normalTexture    = loadTex(ss.meshData.normalTexturePath,     texCache, texCount, importedTexPixels);
        sm.roughnessTexture = loadTex(ss.meshData.roughnessTexturePath,  texCache, texCount, importedTexPixels);
        sm.metallicTexture  = loadTex(ss.meshData.metallicTexturePath,   texCache, texCount, importedTexPixels);
        sm.emissiveTexture  = loadTex(ss.meshData.emissiveTexturePath,   texCache, texCount, importedTexPixels);
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

// ── prefetchTextures ──────────────────────────────────────────────────────────

void Scene::prefetchTextures()
{
    // Collect unique non-EXR texture paths from all current scene nodes.
    std::vector<std::string> uniquePaths;
    {
        std::unordered_map<std::string, bool> seen;
        auto addPath = [&](const std::string& p)
        {
            if (p.empty() || seen.count(p)) return;
            if (p.size() >= 4 &&
                (p.compare(p.size()-4, 4, ".exr") == 0 ||
                 p.compare(p.size()-4, 4, ".EXR") == 0)) return;
            seen[p] = true;
            uniquePaths.push_back(p);
        };
        for (const auto& node : nodes)
            for (const auto& sm : node.submeshes)
            {
                addPath(sm.meshData.diffuseTexturePath);
                addPath(sm.meshData.normalTexturePath);
                addPath(sm.meshData.roughnessTexturePath);
                addPath(sm.meshData.metallicTexturePath);
                addPath(sm.meshData.emissiveTexturePath);
            }
    }

    if (uniquePaths.empty()) return;

    struct Slot { std::string path; TexPixels pixels; };
    std::vector<Slot> slots(uniquePaths.size());
    for (size_t i = 0; i < uniquePaths.size(); ++i)
        slots[i].path = uniquePaths[i];

    stbi_set_flip_vertically_on_load(false);

    const unsigned int hw = std::thread::hardware_concurrency();
    const int nThreads = static_cast<int>(
        std::min(static_cast<size_t>(hw ? hw : 4), uniquePaths.size()));

    auto t_decode = std::chrono::steady_clock::now();

    std::atomic<int> nextSlot{0};
    auto workerFn = [&]()
    {
        for (int idx = nextSlot.fetch_add(1, std::memory_order_relaxed);
             idx < static_cast<int>(slots.size());
             idx = nextSlot.fetch_add(1, std::memory_order_relaxed))
        {
            int w, h, c;
            unsigned char* data = stbi_load(slots[idx].path.c_str(), &w, &h, &c, 4);
            if (data)
            {
                slots[idx].pixels.width  = w;
                slots[idx].pixels.height = h;
                slots[idx].pixels.pixels.assign(
                    data, data + static_cast<size_t>(w) * h * 4);
                stbi_image_free(data);
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t)
        workers.emplace_back(workerFn);
    for (auto& t : workers)
        t.join();

    float t_decode_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_decode).count();

    int decoded = 0;
    for (auto& slot : slots)
        if (!slot.pixels.pixels.empty())
        {
            importedTexPixels[slot.path] = std::move(slot.pixels);
            ++decoded;
        }

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  Texture prefetch: %.0f ms  (%d/%zu textures, %d threads)",
        t_decode_ms, decoded, uniquePaths.size(), nThreads);
    vex::Log::info(buf);
}
