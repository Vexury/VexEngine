#include <vex/scene/mesh_data.h>
#include <vex/core/log.h>
#include <tiny_obj_loader.h>
#include <stb_image.h>
#include <filesystem>
#include <unordered_map>
#include <cmath>
#include <functional>

namespace vex
{

// ---------------------------------------------------------------------------
// Auto-smooth normals: recompute vertex normals from face normals, averaging
// only across faces whose normals differ by less than `angleDeg`.  This fixes
// the diagonal-seam artifact on hard-edged models exported with smooth normals
// while preserving smooth shading on curved surfaces.
// ---------------------------------------------------------------------------
static void autoSmoothNormals(MeshData& mesh, float angleDeg = 80.0f)
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

std::vector<MeshData> MeshData::loadOBJ(const std::string& path)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string filename = std::filesystem::path(path).filename().string();
    Log::info("Loading OBJ: " + filename + "...");

    std::string mtlDir = std::filesystem::path(path).parent_path().string() + "/";

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                               path.c_str(), mtlDir.c_str(), true);

    if (!warn.empty()) Log::warn(warn);
    if (!err.empty())  Log::error(err);
    if (!ok) return {};

    Log::info("  Parsed " + std::to_string(shapes.size()) + " shape(s), "
              + std::to_string(materials.size()) + " material(s)");

    // Group faces by material ID across all shapes
    std::unordered_map<int, MeshData> matGroups;

    for (const auto& shape : shapes)
    {
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

                    int tw, th, tch;
                    if (stbi_info(group.diffuseTexturePath.c_str(), &tw, &th, &tch) && tch == 4)
                        group.alphaClip = true;
                }

                // tinyobjloader defaults Kd to (0.6,0.6,0.6) when map_Kd exists
                // without an explicit Kd line. Use white so the texture is the
                // sole albedo source. Must be outside the path-empty guard so it
                // applies to every face, not just the first one.
                if (!m.diffuse_texname.empty())
                    color = glm::vec3(1.0f);

                // Ns → roughness: sqrt(2 / (Ns + 2)) maps Blinn-Phong exponent to GGX roughness.
                // Roughness textures (map_Ns) are expected in the 0-1 PBR range, not as Ns exponents.
                group.roughness = glm::clamp(
                    std::sqrt(2.0f / (std::max(m.shininess, 0.0f) + 2.0f)), 0.0f, 1.0f);

                // illum → metallic (mirror materials are metallic)
                if (m.illum == 3 || m.illum == 5)
                    group.metallic = 1.0f;

                // d/Tr: constant dissolve < 1 without illum-based transparency or texture alpha clip.
                // Treat as Dielectric so the path tracer transmits through it.
                // Materials with alphaClip (per-pixel alpha from map_Kd) are left as-is.
                if (group.materialType == 0 && !group.alphaClip && m.dissolve < 1.0f - 1e-4f)
                {
                    group.materialType = 2; // Dielectric
                    group.ior = m.ior > 0.0f ? m.ior : 1.5f;
                }

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

                if (group.roughnessTexturePath.empty() && !m.specular_highlight_texname.empty())
                    group.roughnessTexturePath = (std::filesystem::path(mtlDir) / m.specular_highlight_texname).string();

                if (group.metallicTexturePath.empty() && !m.metallic_texname.empty())
                    group.metallicTexturePath = (std::filesystem::path(mtlDir) / m.metallic_texname).string();
            }

            glm::vec3 pos[3];
            glm::vec3 vtxNormals[3];
            glm::vec2 uv[3] = {};
            bool hasVertexNormals = true;
            for (int v = 0; v < fv && v < 3; v++)
            {
                tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
                pos[v] =
                {
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                };
                if (idx.normal_index >= 0)
                {
                    vtxNormals[v] =
                    {
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    };
                }
                else
                {
                    hasVertexNormals = false;
                }
                if (idx.texcoord_index >= 0)
                {
                    uv[v] =
                    {
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
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

            uint32_t baseIdx = static_cast<uint32_t>(group.vertices.size());
            for (int v = 0; v < fv && v < 3; v++)
            {
                group.vertices.push_back({ pos[v], vtxNormals[v], color, emissive, uv[v], tangent });
                group.indices.push_back(baseIdx + v);
            }

            indexOffset += fv;
        }
    }

    // autoSmoothNormals disabled — use OBJ normals as exported

    // Collect into vector
    std::vector<MeshData> result;
    result.reserve(matGroups.size());
    for (auto& [id, group] : matGroups)
    {
        result.push_back(std::move(group));
    }

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

    Log::info("  " + std::to_string(totalVerts) + " vertices, "
              + std::to_string(totalTris) + " triangles, "
              + std::to_string(result.size()) + " submesh(es)");

    std::string matInfo = "  Materials: " + std::to_string(microfacetCount) + " microfacet";
    if (mirrorCount > 0)
        matInfo += ", " + std::to_string(mirrorCount) + " mirror";
    if (dielectricCount > 0)
        matInfo += ", " + std::to_string(dielectricCount) + " dielectric";
    Log::info(matInfo);

    return result;
}

} // namespace vex
