#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace vex
{

// ---------------------------------------------------------------------------
// Tangent helper (same math as loadOBJ)
// ---------------------------------------------------------------------------
static glm::vec4 computeTangent(const glm::vec3& pos0, const glm::vec3& pos1, const glm::vec3& pos2,
                                 const glm::vec2& uv0,  const glm::vec2& uv1,  const glm::vec2& uv2,
                                 const glm::vec3& faceNormal)
{
    glm::vec3 edge1 = pos1 - pos0;
    glm::vec3 edge2 = pos2 - pos0;
    glm::vec2 dUV1  = uv1  - uv0;
    glm::vec2 dUV2  = uv2  - uv0;
    float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;

    glm::vec3 T(1, 0, 0);
    float bSign = 1.0f;
    if (std::abs(det) > 1e-8f)
    {
        float ff = 1.0f / det;
        T = glm::normalize(ff * (dUV2.y * edge1 - dUV1.y * edge2));
        glm::vec3 B = ff * (-dUV2.x * edge1 + dUV1.x * edge2);
        bSign = (glm::dot(glm::cross(faceNormal, T), B) < 0.0f) ? -1.0f : 1.0f;
    }
    return glm::vec4(T, bSign);
}

// ---------------------------------------------------------------------------
// Resolve a GLTF texture index to a file path
// ---------------------------------------------------------------------------
static std::string resolveTexPath(const tinygltf::Model& model,
                                   int texIndex,
                                   const std::string& baseDir)
{
    if (texIndex < 0) return {};
    const auto& tex = model.textures[texIndex];
    if (tex.source < 0) return {};
    const auto& img = model.images[tex.source];
    if (img.uri.empty()) return {};
    return (std::filesystem::path(baseDir) / img.uri).string();
}

// ---------------------------------------------------------------------------
// Build a MeshData from a single GLTF primitive
// ---------------------------------------------------------------------------
static MeshData buildPrimitive(const tinygltf::Model& model,
                                const tinygltf::Primitive& prim,
                                const std::string& baseDir,
                                const std::string& meshName)
{
    MeshData md;
    md.name       = meshName;
    md.objectName = meshName;

    // ── Accessors ────────────────────────────────────────────────────────────
    auto getAccessor = [&](const std::string& attrib) -> const tinygltf::Accessor*
    {
        auto it = prim.attributes.find(attrib);
        if (it == prim.attributes.end()) return nullptr;
        return &model.accessors[it->second];
    };

    const tinygltf::Accessor* posAcc  = getAccessor("POSITION");
    const tinygltf::Accessor* nrmAcc  = getAccessor("NORMAL");
    const tinygltf::Accessor* uvAcc   = getAccessor("TEXCOORD_0");
    const tinygltf::Accessor* tanAcc  = getAccessor("TANGENT");

    if (!posAcc) return md; // no positions — skip

    size_t vertCount = posAcc->count;

    // Helper to get typed span from an accessor
    auto getRaw = [&](const tinygltf::Accessor* acc) -> const uint8_t*
    {
        if (!acc) return nullptr;
        const auto& bv  = model.bufferViews[acc->bufferView];
        const auto& buf = model.buffers[bv.buffer];
        return buf.data.data() + bv.byteOffset + acc->byteOffset;
    };
    auto getStride = [&](const tinygltf::Accessor* acc) -> size_t
    {
        if (!acc) return 0;
        const auto& bv = model.bufferViews[acc->bufferView];
        size_t compSize = tinygltf::GetComponentSizeInBytes(acc->componentType);
        size_t numComp  = tinygltf::GetNumComponentsInType(acc->type);
        return bv.byteStride ? bv.byteStride : compSize * numComp;
    };

    const uint8_t* posPtr = getRaw(posAcc);
    const uint8_t* nrmPtr = getRaw(nrmAcc);
    const uint8_t* uvPtr  = getRaw(uvAcc);
    const uint8_t* tanPtr = getRaw(tanAcc);

    size_t posStride = getStride(posAcc);
    size_t nrmStride = getStride(nrmAcc);
    size_t uvStride  = getStride(uvAcc);
    size_t tanStride = getStride(tanAcc);

    md.vertices.resize(vertCount);
    for (size_t i = 0; i < vertCount; ++i)
    {
        glm::vec3 pos(0.f), nrm(0.f, 1.f, 0.f);
        glm::vec2 uv(0.f);
        glm::vec4 tan(1.f, 0.f, 0.f, 1.f);

        if (posPtr)
        {
            const float* p = reinterpret_cast<const float*>(posPtr + i * posStride);
            pos = { p[0], p[1], p[2] };
        }
        if (nrmPtr)
        {
            const float* p = reinterpret_cast<const float*>(nrmPtr + i * nrmStride);
            nrm = { p[0], p[1], p[2] };
        }
        if (uvPtr)
        {
            const float* p = reinterpret_cast<const float*>(uvPtr + i * uvStride);
            // GLTF UV origin is top-left (V=0 at top); invert V to match GPU upload
            // convention where V=0 is the bottom of the flipped texture in memory.
            // The CPU/compute raytracers independently flip V when sampling importedTexPixels,
            // so storing 1-V here keeps all three paths consistent.
            uv = { p[0], 1.0f - p[1] };
        }
        if (tanPtr)
        {
            const float* p = reinterpret_cast<const float*>(tanPtr + i * tanStride);
            tan = { p[0], p[1], p[2], p[3] };
        }

        md.vertices[i] = { pos, nrm, glm::vec3(1.f), glm::vec3(0.f), uv, tan };
    }

    // ── Indices ──────────────────────────────────────────────────────────────
    if (prim.indices >= 0)
    {
        const auto& idxAcc = model.accessors[prim.indices];
        const auto& bv     = model.bufferViews[idxAcc.bufferView];
        const auto& buf    = model.buffers[bv.buffer];
        const uint8_t* idxPtr = buf.data.data() + bv.byteOffset + idxAcc.byteOffset;

        md.indices.reserve(idxAcc.count);
        for (size_t i = 0; i < idxAcc.count; ++i)
        {
            uint32_t idx = 0;
            if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                idx = reinterpret_cast<const uint32_t*>(idxPtr)[i];
            else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                idx = reinterpret_cast<const uint16_t*>(idxPtr)[i];
            else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                idx = idxPtr[i];
            md.indices.push_back(idx);
        }
    }
    else
    {
        // Non-indexed: generate sequential indices
        md.indices.resize(vertCount);
        for (size_t i = 0; i < vertCount; ++i)
            md.indices[i] = static_cast<uint32_t>(i);
    }

    // ── Compute tangents if not provided ─────────────────────────────────────
    if (!tanPtr && md.indices.size() >= 3)
    {
        size_t triCount = md.indices.size() / 3;
        for (size_t t = 0; t < triCount; ++t)
        {
            uint32_t i0 = md.indices[t * 3 + 0];
            uint32_t i1 = md.indices[t * 3 + 1];
            uint32_t i2 = md.indices[t * 3 + 2];

            glm::vec3 faceNormal = glm::normalize(
                glm::cross(md.vertices[i1].position - md.vertices[i0].position,
                           md.vertices[i2].position - md.vertices[i0].position));

            glm::vec4 tan4 = computeTangent(
                md.vertices[i0].position, md.vertices[i1].position, md.vertices[i2].position,
                md.vertices[i0].uv,       md.vertices[i1].uv,       md.vertices[i2].uv,
                faceNormal);

            md.vertices[i0].tangent = tan4;
            md.vertices[i1].tangent = tan4;
            md.vertices[i2].tangent = tan4;
        }
    }

    // ── Material ─────────────────────────────────────────────────────────────
    if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size()))
    {
        const auto& mat = model.materials[prim.material];
        md.name = mat.name.empty() ? meshName : mat.name;

        const auto& pbr = mat.pbrMetallicRoughness;

        // Base color tint
        if (pbr.baseColorFactor.size() == 4)
        {
            md.baseColor = {
                static_cast<float>(pbr.baseColorFactor[0]),
                static_cast<float>(pbr.baseColorFactor[1]),
                static_cast<float>(pbr.baseColorFactor[2])
            };
        }

        // Scalar roughness/metallic (overridden to 1.0 if texture present)
        md.roughness = static_cast<float>(pbr.roughnessFactor);
        md.metallic  = static_cast<float>(pbr.metallicFactor);

        // Alpha clip
        md.alphaClip = (mat.alphaMode == "MASK");

        // Emissive strength from emissiveFactor (max channel)
        if (mat.emissiveFactor.size() == 3)
        {
            float maxE = static_cast<float>(
                std::max({ mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2] }));
            if (maxE > 1e-5f)
                md.emissiveStrength = maxE;
        }

        // Texture paths
        int baseColorIdx     = pbr.baseColorTexture.index;
        int metallicRoughIdx = pbr.metallicRoughnessTexture.index;
        int normalIdx        = mat.normalTexture.index;
        int emissiveIdx      = mat.emissiveTexture.index;
        int occlusionIdx     = mat.occlusionTexture.index;

        md.diffuseTexturePath    = resolveTexPath(model, baseColorIdx,     baseDir);
        md.normalTexturePath     = resolveTexPath(model, normalIdx,         baseDir);
        md.emissiveTexturePath   = resolveTexPath(model, emissiveIdx,       baseDir);

        // ARM: metallicRoughness texture stores AO(R), Roughness(G), Metallic(B)
        std::string armPath = resolveTexPath(model, metallicRoughIdx, baseDir);
        if (!armPath.empty())
        {
            md.roughnessTexturePath = armPath;
            md.metallicTexturePath  = armPath;
            // AO is always in the R channel of the ARM texture — assign unconditionally.
            // A separate occlusionTexture (different index) overrides it if present.
            md.aoTexturePath = armPath;
            if (occlusionIdx >= 0 && occlusionIdx != metallicRoughIdx)
                md.aoTexturePath = resolveTexPath(model, occlusionIdx, baseDir);

            // Texture drives roughness/metallic — set scalar factors to neutral 1
            md.roughness = 1.0f;
            md.metallic  = 1.0f;
        }
        else
        {
            md.aoTexturePath = resolveTexPath(model, occlusionIdx, baseDir);
        }
    }

    return md;
}

// ---------------------------------------------------------------------------
// DFS walk of GLTF scene nodes
// ---------------------------------------------------------------------------
static void walkNode(const tinygltf::Model& model,
                     int nodeIdx,
                     int parentInfoIdx,
                     const std::string& baseDir,
                     std::vector<MeshData>& outMeshes,
                     std::vector<GLTFNodeInfo>& outNodes)
{
    const auto& node = model.nodes[nodeIdx];

    // Compute local transform
    glm::mat4 localTransform(1.0f);
    if (!node.matrix.empty())
    {
        // Column-major 4x4 in GLTF
        float m[16];
        for (int i = 0; i < 16; ++i) m[i] = static_cast<float>(node.matrix[i]);
        localTransform = glm::mat4(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15]);
    }
    else
    {
        glm::mat4 T(1.f), R(1.f), S(1.f);
        if (node.translation.size() == 3)
            T = glm::translate(glm::mat4(1.f), glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2])));
        if (node.rotation.size() == 4)
        {
            glm::quat q(
                static_cast<float>(node.rotation[3]), // w
                static_cast<float>(node.rotation[0]), // x
                static_cast<float>(node.rotation[1]), // y
                static_cast<float>(node.rotation[2])  // z
            );
            R = glm::mat4_cast(q);
        }
        if (node.scale.size() == 3)
            S = glm::scale(glm::mat4(1.f), glm::vec3(
                static_cast<float>(node.scale[0]),
                static_cast<float>(node.scale[1]),
                static_cast<float>(node.scale[2])));
        localTransform = T * R * S;
    }

    // Record this node
    int myInfoIdx = static_cast<int>(outNodes.size());
    GLTFNodeInfo info;
    info.nodeName       = node.name.empty() ? ("Node" + std::to_string(nodeIdx)) : node.name;
    info.localTransform = localTransform;
    info.parentIndex    = parentInfoIdx;
    outNodes.push_back(info); // placeholder; meshDataIndices filled below

    // Emit primitives for this node's mesh (if any)
    if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size()))
    {
        const auto& gltfMesh = model.meshes[node.mesh];
        std::string meshName = gltfMesh.name.empty()
            ? ("Mesh" + std::to_string(node.mesh))
            : gltfMesh.name;

        for (size_t pi = 0; pi < gltfMesh.primitives.size(); ++pi)
        {
            std::string primName = gltfMesh.primitives.size() == 1
                ? meshName
                : (meshName + "_" + std::to_string(pi));

            MeshData md = buildPrimitive(model, gltfMesh.primitives[pi], baseDir, primName);
            if (md.vertices.empty()) continue;

            md.objectName = info.nodeName;
            int meshIdx = static_cast<int>(outMeshes.size());
            outMeshes.push_back(std::move(md));
            outNodes[myInfoIdx].meshDataIndices.push_back(meshIdx);
        }
    }

    // Recurse into children
    for (int childNodeIdx : node.children)
        walkNode(model, childNodeIdx, myInfoIdx, baseDir, outMeshes, outNodes);
}

// ---------------------------------------------------------------------------
// loadGLTF
// ---------------------------------------------------------------------------
std::vector<MeshData> MeshData::loadGLTF(const std::string& path,
                                          std::vector<GLTFNodeInfo>& outNodes)
{
    auto t_start = std::chrono::steady_clock::now();

    std::string filename = std::filesystem::path(path).filename().string();
    Log::info("Loading GLTF: " + filename + "...");

    tinygltf::TinyGLTF loader;

    // Register a no-op image loader — we load textures ourselves via stbi
    loader.SetImageLoader(
        [](tinygltf::Image*, const int, std::string*, std::string*,
           int, int, const unsigned char*, int, void*) -> bool { return true; },
        nullptr);

    tinygltf::Model model;
    std::string warn, err;

    std::string baseDir = std::filesystem::path(path).parent_path().string();
    if (!baseDir.empty() && baseDir.back() != '/' && baseDir.back() != '\\')
        baseDir += '/';

    bool ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty()) Log::warn(warn);
    if (!err.empty())  Log::error(err);
    if (!ok) return {};

    float t_parse_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "  GLTF parsed in %.0f ms  (%zu meshes, %zu materials, %zu nodes)",
        t_parse_ms,
        model.meshes.size(),
        model.materials.size(),
        model.nodes.size());
    Log::info(buf);

    std::vector<MeshData> result;
    outNodes.clear();

    // Walk all root nodes of the default scene (or scene 0)
    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIdx >= static_cast<int>(model.scenes.size())) return {};

    for (int rootNodeIdx : model.scenes[sceneIdx].nodes)
        walkNode(model, rootNodeIdx, -1, baseDir, result, outNodes);

    // Stats
    size_t totalVerts = 0, totalTris = 0;
    for (const auto& m : result)
    {
        totalVerts += m.vertices.size();
        totalTris  += m.indices.size() / 3;
    }
    std::snprintf(buf, sizeof(buf),
        "  GLTF loaded: %zu submeshes, %zu verts, %zu tris, %zu nodes",
        result.size(), totalVerts, totalTris, outNodes.size());
    Log::info(buf);

    return result;
}

} // namespace vex
