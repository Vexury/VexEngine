#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>
#include <tiny_obj_loader.h>
#include <stb_image.h>
#include <filesystem>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>

namespace vex
{

// ---------------------------------------------------------------------------
// Auto-smooth normals: recompute vertex normals from face normals, averaging
// only across faces whose normals differ by less than `angleDeg`.  This fixes
// the diagonal-seam artifact on hard-edged models exported with smooth normals
// while preserving smooth shading on curved surfaces.
// ---------------------------------------------------------------------------
[[maybe_unused]] static void autoSmoothNormals(MeshData& mesh, float angleDeg = 80.0f)
{
    size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0) return;

    float cosThreshold = std::cos(glm::radians(angleDeg));

    // 1. Compute face normals
    std::vector<glm::vec3> faceNormals(triCount);
    for (size_t t = 0; t < triCount; ++t)
    {
        glm::vec3 v0 = mesh.vertices[mesh.indices[t * 3 + 0]].position;
        glm::vec3 v1 = mesh.vertices[mesh.indices[t * 3 + 1]].position;
        glm::vec3 v2 = mesh.vertices[mesh.indices[t * 3 + 2]].position;
        glm::vec3 n  = glm::cross(v1 - v0, v2 - v0);
        float len = glm::length(n);
        faceNormals[t] = (len > 1e-8f) ? n / len : glm::vec3(0, 1, 0);
    }

    // 2. Spatial map: quantized position -> triangle indices sharing that position
    struct PosKey {
        int32_t x, y, z;
        bool operator==(const PosKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct PosHash {
        size_t operator()(const PosKey& k) const {
            size_t h = std::hash<int32_t>{}(k.x);
            h ^= std::hash<int32_t>{}(k.y) * 2654435761u;
            h ^= std::hash<int32_t>{}(k.z) * 40503u;
            return h;
        }
    };
    auto toKey = [](const glm::vec3& p) -> PosKey {
        return { static_cast<int32_t>(std::round(p.x * 10000.0f)),
                 static_cast<int32_t>(std::round(p.y * 10000.0f)),
                 static_cast<int32_t>(std::round(p.z * 10000.0f)) };
    };

    std::unordered_map<PosKey, std::vector<size_t>, PosHash> posMap;
    for (size_t t = 0; t < triCount; ++t)
    {
        for (int v = 0; v < 3; ++v)
        {
            PosKey key = toKey(mesh.vertices[mesh.indices[t * 3 + v]].position);
            auto& list = posMap[key];
            if (list.empty() || list.back() != t)
                list.push_back(t);
        }
    }

    // 3. For each vertex, average face normals of adjacent triangles within threshold
    for (size_t t = 0; t < triCount; ++t)
    {
        for (int v = 0; v < 3; ++v)
        {
            uint32_t vi = mesh.indices[t * 3 + v];
            PosKey key = toKey(mesh.vertices[vi].position);

            glm::vec3 smoothN(0.0f);
            for (size_t adj : posMap[key])
            {
                if (glm::dot(faceNormals[t], faceNormals[adj]) >= cosThreshold)
                    smoothN += faceNormals[adj];
            }

            float len = glm::length(smoothN);
            mesh.vertices[vi].normal = (len > 1e-8f) ? smoothN / len : faceNormals[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Returns true if the texture at `path` contains any pixel with alpha < 253.
// Uses stbi_info to cheaply skip non-RGBA files, then subsamples up to 4096
// pixels for speed.  Results are cached (mutex-protected) so each unique path
// is loaded at most once — safe for the parallel shape-dedup workers.
// ---------------------------------------------------------------------------
static std::mutex                            s_alphaCacheMtx;
static std::unordered_map<std::string, bool> s_alphaPresenceCache;

static bool textureHasTransparency(const std::string& path)
{
    {
        std::lock_guard<std::mutex> lk(s_alphaCacheMtx);
        auto it = s_alphaPresenceCache.find(path);
        if (it != s_alphaPresenceCache.end()) return it->second;
    }

    bool result = false;
    int w = 0, h = 0, ch = 0;
    if (stbi_info(path.c_str(), &w, &h, &ch) && ch == 4)
    {
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
        if (data)
        {
            int total = w * h;
            int step  = std::max(1, total / 4096);
            for (int i = 0; i < total && !result; i += step)
                if (data[i * 4 + 3] < 253) result = true;
            stbi_image_free(data);
        }
    }

    {
        std::lock_guard<std::mutex> lk(s_alphaCacheMtx);
        s_alphaPresenceCache[path] = result;
    }
    return result;
}

std::vector<MeshData> MeshData::loadOBJ(const std::string& path)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string filename = std::filesystem::path(path).filename().string();
    Log::info("Loading OBJ: " + filename + "...");

    std::string mtlDir = std::filesystem::path(path).parent_path().string() + "/";

    auto t_parse = std::chrono::steady_clock::now();
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                               path.c_str(), mtlDir.c_str(), true);
    float t_parse_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_parse).count();

    if (!warn.empty()) Log::warn(warn);
    if (!err.empty())  Log::error(err);
    if (!ok) return {};

    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  OBJ parsed in %.0f ms  (%zu shapes, %zu materials, %zu verts)",
            t_parse_ms, shapes.size(), materials.size(),
            attrib.vertices.size() / 3);
        Log::info(buf);
    }

    // Group faces by material ID within each shape so that each named OBJ object
    // (o / g tag) becomes its own submesh rather than being merged with other
    // objects that happen to share the same material.
    std::vector<MeshData> result;

    // Vertex deduplication key: the three tinyobj indices that uniquely identify
    // a corner (position, normal, texcoord can each have independent indices in OBJ).
    struct VertKey {
        int vi, ni, ti;
        bool operator==(const VertKey& o) const { return vi==o.vi && ni==o.ni && ti==o.ti; }
    };
    struct VertKeyHash {
        size_t operator()(const VertKey& k) const {
            size_t h = std::hash<int>{}(k.vi);
            h ^= std::hash<int>{}(k.ni) * 2654435761u;
            h ^= std::hash<int>{}(k.ti) * 40503u;
            return h;
        }
    };

    auto t_verts = std::chrono::steady_clock::now();

    // Per-shape results stored at fixed indices — workers write to exclusive slots.
    struct ShapeResult { std::vector<MeshData> submeshes; };
    std::vector<ShapeResult> perShapeResults(shapes.size());

    {
        std::atomic<int> nextShape{0};
        const int numThreads = std::max(1, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> workers;
        workers.reserve(numThreads);

        for (int t = 0; t < numThreads; ++t)
        {
            workers.emplace_back([&]()
            {
                for (;;)
                {
                    int si = nextShape.fetch_add(1, std::memory_order_relaxed);
                    if (si >= (int)shapes.size()) break;

                    const auto& shape = shapes[si];
                    std::unordered_map<int, MeshData> matGroups;
                    std::unordered_map<int, std::unordered_map<VertKey, uint32_t, VertKeyHash>> matVertCache;
                    size_t indexOffset = 0;

                    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
                    {
                        int fv = shape.mesh.num_face_vertices[f];
                        int matId = shape.mesh.material_ids[f];
                        if (matId < 0) matId = -1; // normalize "no material"

                        auto& group = matGroups[matId];

            glm::vec3 color(0.7f);
            glm::vec3 emissive(0.0f);
            if (matId >= 0 && matId < static_cast<int>(materials.size()))
            {
                const auto& m = materials[matId];
                if (group.name.empty())
                    group.name = m.name;
                color    = { m.diffuse[0],  m.diffuse[1],  m.diffuse[2] };
                emissive = { m.emission[0], m.emission[1], m.emission[2] };
                group.emissiveColor = emissive;

                // Material type from illum model
                if (m.illum == 3 || m.illum == 5)
                {
                    group.materialType = 1; // Mirror
                    // Use specular color (Ks) for mirror tint instead of diffuse
                    color = { m.specular[0], m.specular[1], m.specular[2] };
                }
                else if (m.illum == 4 || m.illum == 6 || m.illum == 7)
                {
                    group.materialType = 2; // Dielectric
                    group.ior = m.ior > 0.0f ? m.ior : 1.5f;
                    // Use transmission filter (Tf) for dielectric tint
                    glm::vec3 tf = { m.transmittance[0], m.transmittance[1], m.transmittance[2] };
                    if (glm::length(tf) > 0.001f)
                        color = tf;
                }

                if (group.diffuseTexturePath.empty() && !m.diffuse_texname.empty())
                {
                    group.diffuseTexturePath =
                        (std::filesystem::path(mtlDir) / m.diffuse_texname).string();

                }

                // map_d: opacity mask.
                if (group.alphaTexturePath.empty() && !m.alpha_texname.empty())
                {
                    auto alphaPath =
                        (std::filesystem::path(mtlDir) / m.alpha_texname).string();
                    if (alphaPath == group.diffuseTexturePath)
                    {
                        // map_d aliases map_Kd — alpha lives in diffuse .a channel.
                        // Only enable alpha-clip if pixels are actually transparent;
                        // many Blender exports set map_d = map_Kd even for fully-opaque
                        // materials (e.g. the Bistro scene).
                        group.alphaClip = textureHasTransparency(alphaPath);
                        // Leave alphaTexturePath empty; the shader falls back to
                        // texColor.a when hasAlphaMap is false.
                    }
                    else
                    {
                        // Dedicated separate mask — always clip.
                        group.alphaTexturePath = alphaPath;
                        group.alphaClip = true;
                    }
                }

                // tinyobjloader defaults Kd to (0.6,0.6,0.6) when map_Kd exists
                // without an explicit Kd line. Use white so the texture is the
                // sole albedo source. Must be outside the path-empty guard so it
                // applies to every face, not just the first one.
                if (!m.diffuse_texname.empty())
                    color = glm::vec3(1.0f);

                // Roughness: prefer Pr scalar (PBR extension), otherwise invert Blender's
                // Ns formula: Ns = (1 - roughness)^2 * 1000  =>  roughness = 1 - sqrt(Ns/1000).
                // This correctly round-trips roughness through Blender's OBJ exporter.
                if (m.roughness > 0.0f)
                    group.roughness = glm::clamp(m.roughness, 0.0f, 1.0f);
                else
                    group.roughness = glm::clamp(
                        1.0f - std::sqrt(std::max(m.shininess, 0.0f) / 1000.0f), 0.0f, 1.0f);

                // Metallic: prefer Pm scalar (PBR extension), override to 1 for mirror illum.
                if (m.illum == 3 || m.illum == 5)
                    group.metallic = 1.0f;
                else
                    group.metallic = glm::clamp(m.metallic, 0.0f, 1.0f);

                // d/Tr: constant dissolve < 1 without illum-based transparency or texture alpha clip.
                // Treat as Dielectric so the path tracer transmits through it.
                // Materials with alphaClip (per-pixel alpha from map_Kd) are left as-is.
                if (group.materialType == 0 && !group.alphaClip && m.dissolve < 1.0f - 1e-4f)
                {
                    group.materialType = 2; // Dielectric
                    group.ior = m.ior > 0.0f ? m.ior : 1.5f;
                }

                // Dielectric materials transmit light — alpha clip would punch holes instead of
                // refracting. Clear it unconditionally, regardless of how Dielectric was detected
                // (illum 4/6/7 or dissolve).
                if (group.materialType == 2)
                    group.alphaClip = false;

                if (group.emissiveTexturePath.empty() && !m.emissive_texname.empty())
                {
                    group.emissiveTexturePath =
                        (std::filesystem::path(mtlDir) / m.emissive_texname).string();
                }

                if (group.normalTexturePath.empty())
                {
                    std::string normTex = m.bump_texname;
                    if (normTex.empty()) normTex = m.normal_texname;
                    if (!normTex.empty())
                        group.normalTexturePath =
                            (std::filesystem::path(mtlDir) / normTex).string();
                }

                // map_Pr (PBR roughness) takes priority; fall back to map_Ns (Blinn-Phong highlight).
                if (group.roughnessTexturePath.empty() && !m.roughness_texname.empty())
                    group.roughnessTexturePath = (std::filesystem::path(mtlDir) / m.roughness_texname).string();
                if (group.roughnessTexturePath.empty() && !m.specular_highlight_texname.empty())
                    group.roughnessTexturePath = (std::filesystem::path(mtlDir) / m.specular_highlight_texname).string();

                if (group.metallicTexturePath.empty() && !m.metallic_texname.empty())
                    group.metallicTexturePath = (std::filesystem::path(mtlDir) / m.metallic_texname).string();
                // map_refl is commonly used for metallic by exporters predating PBR's map_Pm.
                // tinyobj only recognises bare "refl" → reflection_texname; "map_refl" is unknown
                // and lands in unknown_parameter. Use ParseTextureNameAndOption so options like
                // "-type sphere" are stripped and only the filename is kept.
                if (group.metallicTexturePath.empty())
                {
                    auto it = m.unknown_parameter.find("map_refl");
                    if (it != m.unknown_parameter.end())
                    {
                        std::string texName;
                        tinyobj::texture_option_t texOpt;
                        if (tinyobj::ParseTextureNameAndOption(&texName, &texOpt, it->second.c_str()) && !texName.empty())
                            group.metallicTexturePath = (std::filesystem::path(mtlDir) / texName).string();
                    }
                }
            }

            glm::vec3 pos[3];
            glm::vec3 vtxNormals[3];
            glm::vec2 uv[3] = {};
            tinyobj::index_t tidx[3] = {};
            bool hasVertexNormals = true;
            for (int v = 0; v < fv && v < 3; v++)
            {
                tidx[v] = shape.mesh.indices[indexOffset + v];
                pos[v] =
                {
                    attrib.vertices[3 * tidx[v].vertex_index + 0],
                    attrib.vertices[3 * tidx[v].vertex_index + 1],
                    attrib.vertices[3 * tidx[v].vertex_index + 2]
                };
                if (tidx[v].normal_index >= 0)
                {
                    vtxNormals[v] =
                    {
                        attrib.normals[3 * tidx[v].normal_index + 0],
                        attrib.normals[3 * tidx[v].normal_index + 1],
                        attrib.normals[3 * tidx[v].normal_index + 2]
                    };
                }
                else
                {
                    hasVertexNormals = false;
                }
                if (tidx[v].texcoord_index >= 0)
                {
                    uv[v] =
                    {
                        attrib.texcoords[2 * tidx[v].texcoord_index + 0],
                        attrib.texcoords[2 * tidx[v].texcoord_index + 1]
                    };
                }
            }

            glm::vec3 faceNormal = glm::normalize(glm::cross(pos[1] - pos[0], pos[2] - pos[0]));
            if (!hasVertexNormals)
                vtxNormals[0] = vtxNormals[1] = vtxNormals[2] = faceNormal;

            // Compute tangent from UV gradients
            glm::vec3 edge1 = pos[1] - pos[0];
            glm::vec3 edge2 = pos[2] - pos[0];
            glm::vec2 dUV1  = uv[1] - uv[0];
            glm::vec2 dUV2  = uv[2] - uv[0];
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
            glm::vec4 tangent(T, bSign);

            auto& vcache = matVertCache[matId];
            for (int v = 0; v < fv && v < 3; v++)
            {
                VertKey key { tidx[v].vertex_index, tidx[v].normal_index, tidx[v].texcoord_index };
                auto it = vcache.find(key);
                if (it != vcache.end())
                {
                    group.indices.push_back(it->second);
                }
                else
                {
                    uint32_t newIdx = static_cast<uint32_t>(group.vertices.size());
                    group.vertices.push_back({ pos[v], vtxNormals[v], color, emissive, uv[v], tangent });
                    group.indices.push_back(newIdx);
                    vcache[key] = newIdx;
                }
            }

                        indexOffset += fv;
                    }

                    // autoSmoothNormals disabled — use OBJ normals as exported

                    // Assign names and collect into this shape's exclusive result slot.
                    std::string shapeName = shape.name.empty()
                        ? ("Shape " + std::to_string(si))
                        : shape.name;

                    for (auto& [matId, group] : matGroups)
                    {
                        if (group.name.empty())
                            group.name = (matId >= 0 && matId < (int)materials.size())
                                ? materials[matId].name : "default";
                        group.objectName = shapeName;
                        perShapeResults[si].submeshes.push_back(std::move(group));
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    // Flatten per-shape results in shape order (preserves original submesh ordering).
    for (auto& sr : perShapeResults)
        for (auto& sm : sr.submeshes)
            result.push_back(std::move(sm));

    float t_verts_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_verts).count();

    size_t totalVerts = 0;
    size_t totalTris = 0;
    int mirrorCount = 0, dielectricCount = 0, microfacetCount = 0;
    for (const auto& m : result)
    {
        totalVerts += m.vertices.size();
        totalTris += m.indices.size() / 3;
        if (m.materialType == 1) ++mirrorCount;
        else if (m.materialType == 2) ++dielectricCount;
        else ++microfacetCount;
    }

    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  Vertex/material processing: %.0f ms  (%zu verts, %zu tris, %zu submeshes)",
            t_verts_ms, totalVerts, totalTris, result.size());
        Log::info(buf);
    }

    std::string matInfo = "  Materials: " + std::to_string(microfacetCount) + " microfacet";
    if (mirrorCount > 0)
        matInfo += ", " + std::to_string(mirrorCount) + " mirror";
    if (dielectricCount > 0)
        matInfo += ", " + std::to_string(dielectricCount) + " dielectric";
    Log::info(matInfo);

    return result;
}

} // namespace vex
