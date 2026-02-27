#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace vex
{

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec3 emissive;
    glm::vec2 uv{0.0f, 0.0f};
    glm::vec4 tangent{0.0f};  // xyz = tangent, w = bitangent sign (+1 or -1)
};

struct MeshData
{
    std::string name;       // material group name (from MTL usemtl)
    std::string objectName; // parent object/shape name (from OBJ o/g tag)
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::string diffuseTexturePath;
    std::string emissiveTexturePath;
    std::string normalTexturePath;
    std::string roughnessTexturePath;
    std::string metallicTexturePath;
    bool alphaClip = false;
    int materialType = 0;   // 0=Microfacet (GGX), 1=Mirror, 2=Dielectric
    float ior = 1.5f;       // index of refraction (dielectric only)
    float roughness = 0.5f;
    float metallic = 0.0f;

    static std::vector<MeshData> loadOBJ(const std::string& path);
};

} // namespace vex
